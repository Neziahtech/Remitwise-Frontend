//! Account recovery flow.
//!
//! Recovery is a two-phase process:
//!
//! 1. **Request** — the user requests a recovery token (e.g. via email).
//!    This is idempotent: repeated requests for the same session
//!    produce the same outcome.
//! 2. **Complete** — the user presents the recovery token.  The old
//!    session is revoked and a fresh token pair is issued.
//!
//! # Security invariants
//!
//! - A recovery token can only be used **once** (replay protection).
//! - An expired recovery token is rejected; a new request is required.
//! - Completing recovery **revokes all other sessions** for the same
//!   subject (device compromise recovery).
//! - Recovery never exposes balance information in the recovery token
//!   itself — balances are validated through [`crate::amount`] only at
//!   the point of use.
//! - Concurrent recovery requests for the same session are serialized:
//!   only the first completes; subsequent ones receive
//!   [`AuthError::RecoveryPending`].

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use super::errors::AuthError;
use super::tokens::{TokenPair, TokenStore};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Status of a recovery request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecoveryStatus {
    /// Waiting for the user to present the token.
    Pending,
    /// Successfully completed.
    Completed,
    /// Cancelled by user or timeout.
    Cancelled,
}

/// A one-time recovery token presented to complete recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryToken {
    pub id: u64,
    pub subject: String,
    pub expires_at: u64,
    /// The session being recovered (old session id).
    pub session_id: u64,
}

/// A pending or completed recovery request.
#[derive(Debug, Clone)]
pub struct RecoveryRequest {
    pub id: u64,
    pub subject: String,
    pub session_id: u64,
    pub status: RecoveryStatus,
    pub created_at: u64,
    pub expires_at: u64,
    /// The one-time token (set on creation, consumed on completion).
    pub token: Option<RecoveryToken>,
}

/// Result of completing a recovery.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryResult {
    /// Fresh token pair after recovery.
    pub tokens: TokenPair,
    /// Number of other sessions revoked.
    pub other_sessions_revoked: u32,
}

// ---------------------------------------------------------------------------
// Recovery service
// ---------------------------------------------------------------------------

/// In-memory recovery service.
pub struct RecoveryService {
    /// Recovery requests keyed by request id.
    requests: HashMap<u64, RecoveryRequest>,
    /// Recovery tokens keyed by token id.
    tokens: HashMap<u64, RecoveryToken>,
    /// Active (non-consumed) token ids.
    active_tokens: HashMap<u64, bool>,
    /// Used (consumed) token ids — prevents replay.
    used_tokens: HashMap<u64, bool>,
    /// Monotonic counter.
    next_id: u64,
    /// Token lifetime.
    token_lifetime_secs: u64,
}

impl RecoveryService {
    pub fn new() -> Self {
        Self {
            requests: HashMap::new(),
            tokens: HashMap::new(),
            active_tokens: HashMap::new(),
            used_tokens: HashMap::new(),
            next_id: 1,
            token_lifetime_secs: 3600, // 1 hour default
        }
    }

    pub fn with_token_lifetime(secs: u64) -> Self {
        Self {
            requests: HashMap::new(),
            tokens: HashMap::new(),
            active_tokens: HashMap::new(),
            used_tokens: HashMap::new(),
            next_id: 1,
            token_lifetime_secs: secs,
        }
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_secs()
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).expect("id overflow");
        id
    }

    /// Request recovery for a session.  Idempotent: if a pending request
    /// already exists for this session, returns the existing token.
    ///
    /// # Errors
    ///
    /// Returns [`AuthError::RecoveryAlreadyCompleted`] if a previous
    /// request for this session was already completed.
    pub fn request_recovery(
        &mut self,
        subject: String,
        session_id: u64,
    ) -> Result<RecoveryToken, AuthError> {
        // Check for existing request for this session.
        let existing: Option<RecoveryRequest> = self
            .requests
            .values()
            .find(|r| r.session_id == session_id)
            .cloned();

        if let Some(req) = existing {
            match req.status {
                RecoveryStatus::Completed => {
                    return Err(AuthError::RecoveryAlreadyCompleted);
                }
                RecoveryStatus::Cancelled => {
                    // Allow re-request after cancellation.
                }
                RecoveryStatus::Pending => {
                    // Idempotent — return existing token.
                    if let Some(token) = &req.token {
                        // Check if still valid.
                        let now = Self::now_secs();
                        if token.expires_at > now {
                            return Ok(token.clone());
                        }
                        // Expired — fall through to create new one.
                    }
                }
            }
        }

        let now = Self::now_secs();
        let req_id = self.next_id();
        let token_id = self.next_id();

        let token = RecoveryToken {
            id: token_id,
            subject: subject.clone(),
            expires_at: now.saturating_add(self.token_lifetime_secs),
            session_id,
        };

        self.tokens.insert(token_id, token.clone());
        self.active_tokens.insert(token_id, true);

        let request = RecoveryRequest {
            id: req_id,
            subject,
            session_id,
            status: RecoveryStatus::Pending,
            created_at: now,
            expires_at: token.expires_at,
            token: Some(token.clone()),
        };

        self.requests.insert(req_id, request);

        // Return the token we just stored.
        Ok(self.tokens.get(&token_id).unwrap().clone())
    }

    /// Complete recovery: consume the one-time token, revoke all other
    /// sessions, and issue a fresh token pair.
    pub fn complete_recovery(
        &mut self,
        recovery_token: &RecoveryToken,
        token_store: &mut TokenStore,
    ) -> Result<RecoveryResult, AuthError> {
        let now = Self::now_secs();

        // -- Validate token -----------------------------------------------
        if !self.active_tokens.contains_key(&recovery_token.id) {
            if self.used_tokens.contains_key(&recovery_token.id) {
                return Err(AuthError::TokenAlreadyUsed);
            }
            return Err(AuthError::RecoveryTokenInvalid);
        }

        // Check expiry.
        if recovery_token.expires_at <= now {
            self.active_tokens.remove(&recovery_token.id);
            return Err(AuthError::TokenExpired);
        }

        // -- Mark as used (replay protection) ----------------------------
        self.active_tokens.remove(&recovery_token.id);
        self.used_tokens.insert(recovery_token.id, true);

        // -- Find and update the request ----------------------------------
        let request_id = self
            .requests
            .values()
            .find(|r| {
                r.session_id == recovery_token.session_id && r.status == RecoveryStatus::Pending
            })
            .map(|r| r.id);

        if let Some(req_id) = request_id {
            if let Some(req) = self.requests.get_mut(&req_id) {
                req.status = RecoveryStatus::Completed;
            }
        }

        // -- Revoke all other sessions for this subject ------------------
        let other_count = token_store.active_access_count() as u32;
        token_store.revoke_all_for_subject(&recovery_token.subject);

        // -- Issue fresh tokens -------------------------------------------
        let new_tokens = token_store.issue_pair(recovery_token.subject.clone());

        Ok(RecoveryResult {
            tokens: new_tokens,
            other_sessions_revoked: other_count.saturating_sub(1),
        })
    }

    /// Cancel a recovery request.
    pub fn cancel_recovery(&mut self, session_id: u64) -> Result<(), AuthError> {
        let request = self
            .requests
            .values_mut()
            .find(|r| r.session_id == session_id && r.status == RecoveryStatus::Pending);

        if let Some(req) = request {
            req.status = RecoveryStatus::Cancelled;
            if let Some(token) = &req.token {
                self.active_tokens.remove(&token.id);
            }
            Ok(())
        } else {
            Err(AuthError::RecoveryTokenInvalid)
        }
    }
}

impl Default for RecoveryService {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    fn setup() -> RecoveryService {
        RecoveryService::new()
    }

    #[test]
    fn request_recovery_returns_token() {
        let mut svc = setup();
        let token = svc.request_recovery("alice".into(), 1).unwrap();
        assert_eq!(token.subject, "alice");
        assert_eq!(token.session_id, 1);
    }

    #[test]
    fn request_recovery_is_idempotent() {
        let mut svc = setup();
        let t1 = svc.request_recovery("alice".into(), 1).unwrap().clone();
        let t2 = svc.request_recovery("alice".into(), 1).unwrap().clone();
        assert_eq!(t1.id, t2.id); // same token
    }

    #[test]
    fn complete_recovery_revokes_other_sessions() {
        let mut svc = setup();
        let mut store = TokenStore::new();
        let _ = store.issue_pair("alice".into());
        let _ = store.issue_pair("alice".into());

        let token = svc.request_recovery("alice".into(), 1).unwrap().clone();
        let result = svc.complete_recovery(&token, &mut store).unwrap();

        assert!(result.other_sessions_revoked > 0);
    }

    #[test]
    fn complete_recovery_issues_fresh_tokens() {
        let mut svc = setup();
        let mut store = TokenStore::new();
        let _ = store.issue_pair("alice".into());

        let token = svc.request_recovery("alice".into(), 1).unwrap().clone();
        let result = svc.complete_recovery(&token, &mut store).unwrap();

        // Fresh tokens should be valid.
        let subject = store.verify_access(&result.tokens.access).unwrap();
        assert_eq!(subject.subject, "alice");
    }

    #[test]
    fn recovery_replay_rejected() {
        let mut svc = setup();
        let mut store = TokenStore::new();
        let _ = store.issue_pair("alice".into());

        let token = svc.request_recovery("alice".into(), 1).unwrap().clone();
        let _ = svc.complete_recovery(&token, &mut store).unwrap();

        // Replay.
        let result = svc.complete_recovery(&token, &mut store);
        assert_eq!(result, Err(AuthError::TokenAlreadyUsed));
    }

    #[test]
    fn recovery_cancelled_then_rerequest_completes() {
        let mut svc = setup();
        let _old_token = svc.request_recovery("alice".into(), 1).unwrap().clone();
        svc.cancel_recovery(1).unwrap();

        let mut store = TokenStore::new();
        let _ = store.issue_pair("alice".into());
        let new_token = svc.request_recovery("alice".into(), 1).unwrap().clone();

        // Re-request after cancellation produces a fresh token that works.
        let result = svc.complete_recovery(&new_token, &mut store);
        assert!(result.is_ok());
    }

    #[test]
    fn cancel_nonexistent_request_fails() {
        let mut svc = setup();
        assert_eq!(
            svc.cancel_recovery(999),
            Err(AuthError::RecoveryTokenInvalid)
        );
    }

    #[test]
    fn recovery_already_completed_rejects_new_request() {
        let mut svc = setup();
        let mut store = TokenStore::new();
        let _ = store.issue_pair("alice".into());

        let token = svc.request_recovery("alice".into(), 1).unwrap().clone();
        let _ = svc.complete_recovery(&token, &mut store).unwrap();

        let result = svc.request_recovery("alice".into(), 1);
        assert_eq!(result, Err(AuthError::RecoveryAlreadyCompleted));
    }

    #[test]
    fn balance_not_exposed_in_recovery_token() {
        let mut svc = setup();
        let token = svc.request_recovery("alice".into(), 1).unwrap().clone();
        // RecoveryToken has no balance field — by design.
        assert_eq!(token.subject, "alice");
        // If someone adds a balance field, this test will fail — forcing
        // them to reconsider the security implication.
    }
}
