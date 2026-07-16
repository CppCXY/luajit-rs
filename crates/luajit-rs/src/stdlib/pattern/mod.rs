//! High-performance Lua pattern matching — byte-oriented, zero-AST design
//!
//! Modeled after C Lua's lstrlib.c.  Operates on `&[u8]` (raw bytes).
//! In Lua, all string operations are byte-oriented — each byte is a "character".
//!
//! 1. NO AST / parse phase — pattern string is interpreted directly during matching
//! 2. Fixed-size capture array (`LUA_MAXCAPTURES` = 32 slots)
//! 3. MatchState struct tracks all state on the stack
//! 4. Pattern is `&[u8]`, walked with index arithmetic (like C pointers)
//! 5. Recursion-limited to prevent stack overflow on pathological patterns

mod class;
mod engine;

/// Maximum number of captures, from C Lua's `LUA_MAXCAPTURES`.
pub const LUA_MAXCAPTURES: usize = 32;

/// Maximum repetition count when matching `*` / `+` quantifiers.
pub const MAX_REPETITION: usize = usize::MAX;

/// Recursion depth limit for the backtracking matcher.
pub const RECURSION_LIMIT: usize = 200;

/// Maximum match-call nesting depth (C Lua's `MAXCCALLS`).
pub const MAXCCALLS_PATTERN: usize = 200;

pub use engine::{
    CaptureValue, find, find_all_matches, find_assume_valid, gsub, is_plain_pattern, validate,
};
