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
                    out.insert(key, Value::String("$REDACTED".into()));
                } else {
                    out.insert(key, canonicalize_json_value(val, ctx));
                }
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
    let prefixes = ["ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_", "sk-"];
    let mut s = input.to_string();
    for prefix in prefixes {
        s = redact_prefix_token(&s, prefix);
    }
    redact_akia(&s)
}

fn redact_prefix_token(input: &str, prefix: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(idx) = rest.find(prefix) {
        out.push_str(&rest[..idx]);
        out.push_str("$REDACTED");
        let after = &rest[idx + prefix.len()..];
        let consume = after
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
            .count();
        rest = &after[consume..];
    }
    out.push_str(rest);
    out
}

fn redact_akia(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(idx) = rest.find("AKIA") {
        out.push_str(&rest[..idx]);
        let candidate = &rest[idx..];
        let token: String = candidate.chars().take(20).collect();
        if token.len() == 20
            && token
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        {
            out.push_str("$REDACTED");
            rest = &rest[idx + 20..];
        } else {
            out.push_str("AKIA");
            rest = &rest[idx + 4..];
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
            ctx.redact_str("token=ghp_abcdefghijklmnopqrstuvwxyz012345"),
            "token=$REDACTED"
        );
        assert_eq!(ctx.redact_str("key=sk-abcDEF123"), "key=$REDACTED");
        assert_eq!(ctx.redact_str("id=AKIAIOSFODNN7EXAMPLE"), "id=$REDACTED");
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
