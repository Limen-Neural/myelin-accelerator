// Copyright 2026 Raul Montoya Cardenas
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonicalize JSON and strip local usernames, home paths, and secrets.

use serde_json::{Map, Value};

/// Local identity used when redacting strings.
#[derive(Clone, Debug, Default)]
pub struct RedactionContext {
    pub home: Option<String>,
    pub user: Option<String>,
}

impl RedactionContext {
    /// Read `HOME` / `USER` (with `USERNAME` / `LOGNAME` fallbacks).
    pub fn from_env() -> Self {
        let home = std::env::var("HOME")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var("USERPROFILE").ok().filter(|s| !s.is_empty()));
        let user = std::env::var("USER")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var("USERNAME").ok().filter(|s| !s.is_empty()))
            .or_else(|| std::env::var("LOGNAME").ok().filter(|s| !s.is_empty()));
        Self { home, user }
    }

    /// Redact a single string field.
    pub fn redact_str(&self, input: &str) -> String {
        let mut s = input.to_string();
        if let Some(home) = self.home.as_deref()
            && !home.is_empty()
        {
            s = s.replace(home, "$HOME");
        }
        if let Some(user) = self.user.as_deref()
            && !user.is_empty()
        {
            for prefix in ["/home/", "/Users/", "C:\\Users\\", "C:/Users/"] {
                s = s.replace(&format!("{prefix}{user}"), "$HOME");
            }
            s = replace_path_component(&s, user, "$USER");
        }
        s = redact_secret_tokens(&s);
        if looks_like_windows_path(&s) {
            normalize_slashes(&s)
        } else {
            s
        }
    }
}

/// Recursively redact strings, replace secret-keyed values, and sort object keys.
pub fn canonicalize_json_value(value: Value, ctx: &RedactionContext) -> Value {
    match value {
        Value::String(s) => Value::String(ctx.redact_str(&s)),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|v| canonicalize_json_value(v, ctx))
                .collect(),
        ),
        Value::Object(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            let mut out = Map::new();
            for key in keys {
                let val = map.get(&key).cloned().unwrap_or(Value::Null);
                if key_looks_secret(&key) {
                    out.insert(key, redact_secret_value_preserving_types(val));
                } else {
                    out.insert(key, canonicalize_json_value(val, ctx));
                }
            }
            Value::Object(out)
        }
        other => other,
    }
}

fn redact_secret_value_preserving_types(value: Value) -> Value {
    match value {
        Value::String(_) => Value::String("$REDACTED".into()),
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(redact_secret_value_preserving_types)
                .collect(),
        ),
        Value::Object(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            let mut out = Map::new();
            for key in keys {
                let val = map.get(&key).cloned().unwrap_or(Value::Null);
                out.insert(key, redact_secret_value_preserving_types(val));
            }
            Value::Object(out)
        }
        other => other,
    }
}

/// Serialize `value` to pretty JSON after redaction and key sorting.
pub fn redact_and_canonicalize<T: serde::Serialize>(
    value: &T,
    ctx: &RedactionContext,
) -> Result<String, serde_json::Error> {
    let raw = serde_json::to_value(value)?;
    let canonical = canonicalize_json_value(raw, ctx);
    serde_json::to_string_pretty(&canonical)
}

fn key_looks_secret(key: &str) -> bool {
    let upper = key.to_ascii_uppercase();
    const HINTS: &[&str] = &[
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "API_KEY",
        "PRIVATE_KEY",
        "CREDENTIAL",
        "AUTHORIZATION",
        "AUTH_HEADER",
        "BEARER",
    ];
    HINTS.iter().any(|h| upper.contains(h))
}

fn replace_path_component(input: &str, user: &str, replacement: &str) -> String {
    if user.is_empty() {
        return input.to_string();
    }
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(idx) = rest.find(user) {
        let after_idx = idx + user.len();
        let prev_ok = idx == 0
            || rest[..idx]
                .chars()
                .next_back()
                .is_some_and(is_path_delim_char);
        let next_ok = after_idx == rest.len()
            || rest[after_idx..]
                .chars()
                .next()
                .is_some_and(is_path_delim_char);
        if prev_ok && next_ok {
            out.push_str(&rest[..idx]);
            out.push_str(replacement);
            rest = &rest[after_idx..];
        } else {
            out.push_str(&rest[..after_idx]);
            rest = &rest[after_idx..];
        }
    }
    out.push_str(rest);
    out
}

fn is_path_delim_char(c: char) -> bool {
    matches!(c, '/' | '\\' | ':' | '@')
}

fn redact_secret_tokens(input: &str) -> String {
    let prefixes = [
        ("ghp_", 36),
        ("gho_", 36),
        ("ghu_", 36),
        ("ghs_", 36),
        ("ghr_", 36),
        ("github_pat_", 40),
        ("sk-", 20),
    ];
    let mut s = input.to_string();
    for (prefix, min_payload_len) in prefixes {
        s = redact_prefix_token(&s, prefix, min_payload_len);
    }
    redact_aws_access_key_ids(&s)
}

fn redact_prefix_token(input: &str, prefix: &str, min_payload_len: usize) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(idx) = rest.find(prefix) {
        let after = &rest[idx + prefix.len()..];
        let consume = after.chars().take_while(|c| is_token_char(*c)).count();
        let has_left_boundary = idx == 0
            || rest[..idx]
                .chars()
                .next_back()
                .is_some_and(|c| !is_token_char(c));
        if has_left_boundary && consume >= min_payload_len {
            out.push_str(&rest[..idx]);
            out.push_str("$REDACTED");
            rest = &after[consume..];
        } else {
            let prefix_end = idx + prefix.len();
            out.push_str(&rest[..prefix_end]);
            rest = &rest[prefix_end..];
        }
    }
    out.push_str(rest);
    out
}

fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn redact_aws_access_key_ids(input: &str) -> String {
    ["AKIA", "ASIA"]
        .into_iter()
        .fold(input.to_string(), |value, prefix| {
            redact_aws_access_key_prefix(&value, prefix)
        })
}

fn redact_aws_access_key_prefix(input: &str, prefix: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(idx) = rest.find(prefix) {
        out.push_str(&rest[..idx]);
        let candidate = &rest[idx..];
        let token: String = candidate.chars().take(20).collect();
        let token_end = idx + 20;
        let has_left_boundary = idx == 0
            || rest[..idx]
                .chars()
                .next_back()
                .is_some_and(|c| !is_token_char(c));
        let has_right_boundary = token.len() == 20
            && (token_end == rest.len()
                || rest[token_end..]
                    .chars()
                    .next()
                    .is_some_and(|c| !is_token_char(c)));
        if has_left_boundary
            && has_right_boundary
            && token.len() == 20
            && token
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        {
            out.push_str("$REDACTED");
            rest = &rest[idx + 20..];
        } else {
            out.push_str(prefix);
            rest = &rest[idx + prefix.len()..];
        }
    }
    out.push_str(rest);
    out
}

fn normalize_slashes(input: &str) -> String {
    let replaced = input.replace('\\', "/");
    let mut out = String::with_capacity(replaced.len());
    let mut prev_slash = false;
    for ch in replaced.chars() {
        if ch == '/' {
            if !prev_slash {
                out.push('/');
            }
            prev_slash = true;
        } else {
            prev_slash = false;
            out.push(ch);
        }
    }
    out
}

fn looks_like_windows_path(input: &str) -> bool {
    input.starts_with("\\\\")
        || input.contains("$HOME\\")
        || input.as_bytes().windows(3).any(|window| {
            window[0].is_ascii_alphabetic() && window[1] == b':' && window[2] == b'\\'
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn alice() -> RedactionContext {
        RedactionContext {
            home: Some("/home/alice".into()),
            user: Some("alice".into()),
        }
    }

    #[test]
    fn redacts_home_and_user_path_components() {
        let ctx = alice();
        assert_eq!(
            ctx.redact_str("/home/alice/.rustup/toolchains"),
            "$HOME/.rustup/toolchains"
        );
        assert_eq!(
            ctx.redact_str("nvcc=/home/alice/cuda/bin/nvcc"),
            "nvcc=$HOME/cuda/bin/nvcc"
        );
        assert_eq!(ctx.redact_str("/tmp/alice/out"), "/tmp/$USER/out");
        // Do not rewrite the username inside unrelated words.
        assert_eq!(ctx.redact_str("malice"), "malice");
    }

    #[test]
    fn redacts_single_character_user_path_component() {
        let ctx = RedactionContext {
            home: None,
            user: Some("x".into()),
        };
        assert_eq!(ctx.redact_str("/tmp/x/out"), "/tmp/$USER/out");
    }

    #[test]
    fn preserves_non_path_slashes_and_backslashes() {
        let ctx = alice();
        assert_eq!(
            ctx.redact_str("https://example.com/a//b"),
            "https://example.com/a//b"
        );
        assert_eq!(ctx.redact_str(r"escaped\\value"), r"escaped\\value");
    }

    #[test]
    fn redacts_github_and_openai_style_tokens() {
        let ctx = alice();
        assert_eq!(
            ctx.redact_str(&format!("token=ghp_{}", "a".repeat(36))),
            "token=$REDACTED"
        );
        let openai_key = format!("key=sk-{}", "a".repeat(32));
        assert_eq!(ctx.redact_str(&openai_key), "key=$REDACTED");
        assert_eq!(ctx.redact_str("id=AKIAIOSFODNN7EXAMPLE"), "id=$REDACTED");
    }

    #[test]
    fn preserves_embedded_or_implausibly_short_openai_prefixes() {
        let ctx = alice();
        assert_eq!(ctx.redact_str("mask-kernel"), "mask-kernel");
        assert_eq!(ctx.redact_str("key=sk-short"), "key=sk-short");
    }

    #[test]
    fn preserves_implausibly_short_github_token_prefixes() {
        let ctx = alice();
        for value in [
            "ghp_kernel",
            "gho_kernel",
            "ghu_kernel",
            "ghs_kernel",
            "ghr_kernel",
            "github_pat_kernel",
        ] {
            assert_eq!(ctx.redact_str(value), value);
        }
    }

    #[test]
    fn redacts_temporary_aws_access_key_ids_under_ordinary_keys() {
        let ctx = alice();
        let temporary_key = format!("ASIA{}", "A".repeat(16));
        let out = canonicalize_json_value(json!({ "identity": temporary_key }), &ctx);

        assert_eq!(out["identity"], json!("$REDACTED"));
    }

    #[test]
    fn preserves_aws_key_shapes_embedded_in_larger_tokens() {
        let ctx = alice();
        let access_key = format!("AKIA{}", "A".repeat(16));
        let left_embedded = format!("labelX{access_key}");
        let right_embedded = format!("{access_key}Y");

        assert_eq!(ctx.redact_str(&left_embedded), left_embedded);
        assert_eq!(ctx.redact_str(&right_embedded), right_embedded);
    }

    #[test]
    fn secret_keys_are_stripped_in_json() {
        let ctx = alice();
        let value = json!({ "password": "hunter2", "ok": 1 });
        let out = canonicalize_json_value(value, &ctx);
        assert_eq!(out["password"], json!("$REDACTED"));
        assert_eq!(out["ok"], json!(1));
    }
}
