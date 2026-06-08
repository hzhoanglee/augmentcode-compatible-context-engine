pub mod google;
pub mod openai;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use anyhow::{Result, bail};
use reqwest::Client;
use tracing::warn;
use crate::config::LlmConfig;

#[derive(Clone)]
pub struct LlmClient {
    provider: String,
    model: String,
    api_keys: Vec<String>,
    http: Client,
    key_cursor: std::sync::Arc<AtomicUsize>,
    use_structured_output: bool,
}

/// Whether `provider` has a native JSON output mode the reranker can request.
/// Anything not listed here uses the XML tag-wrapping path regardless of the
/// `use_structured_output` setting.
fn provider_supports_structured_output(provider: &str) -> bool {
    matches!(provider, "google" | "openai")
}

impl LlmClient {
    /// Create a new client. Returns None if api_keys is empty.
    pub fn new(config: &LlmConfig) -> Option<Self> {
        if config.api_keys.is_empty() {
            return None;
        }
        let http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .ok()?;
        Some(Self {
            provider: config.provider.clone(),
            model: config.rerank_model.clone(),
            api_keys: config.api_keys.clone(),
            http,
            key_cursor: std::sync::Arc::new(AtomicUsize::new(0)),
            use_structured_output: config.use_structured_output,
        })
    }

    /// Whether this client will request native JSON output for reranking.
    /// True only when the setting is enabled AND the provider supports it; when
    /// the setting is on but the provider lacks a JSON mode, logs a warning once
    /// per call decision so operators see why the XML path is used.
    pub fn structured_output_active(&self) -> bool {
        if !self.use_structured_output {
            return false;
        }
        if provider_supports_structured_output(&self.provider) {
            true
        } else {
            tracing::warn!(
                provider = %self.provider,
                "use_structured_output is enabled but provider has no native JSON mode; \
                 falling back to XML rerank path"
            );
            false
        }
    }

    /// Dispatch to the provider-specific completion function.
    async fn call_provider(&self, system: &str, user: &str, temperature: f32, structured: bool, key: &str) -> Result<String> {
        match self.provider.as_str() {
            "google" => google::complete(&self.http, &self.model, key, system, user, temperature, structured).await,
            "openai" => openai::complete(&self.http, &self.model, key, system, user, temperature, structured).await,
            other => bail!("unsupported LLM provider: {other}"),
        }
    }

    /// Send a completion request to the configured LLM provider.
    /// Rotates through all keys on failure; backs off 2s and retries once more
    /// before returning the last error.
    pub async fn complete(&self, system: &str, user: &str, temperature: f32, structured: bool) -> Result<String> {
        let n_keys = self.api_keys.len();
        let start_cursor = self.key_cursor.fetch_add(1, Ordering::Relaxed) % n_keys;

        let mut last_err = None;

        // First pass — try each key once.
        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.api_keys[key_idx];
            match self.call_provider(system, user, temperature, structured, key).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    warn!(key_index = key_idx, error = %e, "LLM call failed — trying next key");
                    last_err = Some(e);
                }
            }
        }

        // All keys failed — backoff 2s and retry once more.
        tokio::time::sleep(Duration::from_secs(2)).await;

        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.api_keys[key_idx];
            match self.call_provider(system, user, temperature, structured, key).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap())
    }
}
