use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::config::{EmbeddingConfig, RerankConfig};
use crate::embedding::InputType;

const VOYAGE_ENDPOINT: &str = "https://api.voyageai.com/v1/embeddings";
pub const MAX_BATCH_SIZE: usize = 128;
// ─── Request / response shapes ────────────────────────────────────────────

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
    /// Voyage-specific; omitted for OpenAI-compatible endpoints, which reject
    /// unknown fields or ignore the distinction.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<&'a str>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

// ─── Client ───────────────────────────────────────────────────────────────

/// VoyageAI embedding client with round-robin key rotation and retry on 429.
#[derive(Clone)]
pub struct VoyageClient {
    inner: Arc<VoyageInner>,
}

struct VoyageInner {
    http: Client,
    /// Tighter-timeout client for user-facing query embedding (30s vs 120s).
    query_http: Client,
    model: String,
    api_keys: Vec<String>,
    /// Resolved embeddings endpoint URL. Voyage's official endpoint by default;
    /// any OpenAI-compatible `…/v1/embeddings` when `provider == "openai"`.
    endpoint: String,
    /// Voyage accepts an `input_type` hint (document vs query); OpenAI-compatible
    /// servers don't, so the field is omitted there.
    send_input_type: bool,
    /// Max texts per request — providers enforce different input caps
    /// (Voyage 128, BytePlus Ark 10, …).
    batch_size: usize,
    /// Round-robin cursor — atomically advanced on each batch call.
    key_cursor: AtomicUsize,
}

/// Normalize a user-supplied base URL into a full embeddings endpoint.
/// Accepts the base form (`http://host:1234/v1`, with or without trailing
/// slash) or the full `…/embeddings` URL.
fn embeddings_url(base: &str) -> String {
    let trimmed = base.trim().trim_end_matches('/');
    if trimmed.ends_with("/embeddings") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/embeddings")
    }
}

impl VoyageClient {
    /// Create a client from the embedding settings. Returns `Err` if `api_keys`
    /// is empty. `provider == "openai"` (with `base_url`) targets any
    /// OpenAI-compatible /v1/embeddings endpoint; anything else is Voyage.
    pub fn from_config(config: &EmbeddingConfig) -> Result<Self> {
        let openai = config.provider == "openai";
        let endpoint =
            if openai {
                let base = config
                .base_url
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| anyhow::anyhow!(
                    "embedding provider 'openai' requires base_url (e.g. http://host:1234/v1)"
                ))?;
                embeddings_url(base)
            } else {
                VOYAGE_ENDPOINT.to_string()
            };
        // OpenAI-compatible local servers may run without auth — substitute one
        // empty key, which suppresses the Authorization header per request.
        let keys = if openai && config.api_keys.is_empty() {
            vec![String::new()]
        } else {
            config.api_keys.clone()
        };
        Self::with_endpoint(
            config.model.clone(),
            keys,
            endpoint,
            !openai,
            config.embed_batch_size.clamp(1, MAX_BATCH_SIZE),
        )
    }

    /// Back-compat constructor: official Voyage endpoint.
    pub fn new(model: String, api_keys: Vec<String>) -> Result<Self> {
        Self::with_endpoint(
            model,
            api_keys,
            VOYAGE_ENDPOINT.to_string(),
            true,
            MAX_BATCH_SIZE,
        )
    }

    /// Client for the embedding-similarity reranker (LM Studio / any
    /// OpenAI-compatible /v1/embeddings server). Unlike the main embedding
    /// client, API keys may be empty — local servers often run without auth —
    /// in which case requests carry an empty bearer token.
    pub fn from_rerank_config(config: &RerankConfig) -> Result<Self> {
        let base = config
            .base_url
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!("rerank requires base_url (e.g. http://host:1234/v1)")
            })?;
        let keys = if config.api_keys.is_empty() {
            vec![String::new()]
        } else {
            config.api_keys.clone()
        };
        // Conservative batch size: rerank candidate sets are small (~top_k), and
        // some gateways cap embeddings input at 10 — a few extra requests cost
        // little next to a wrong 400.
        Self::with_endpoint(config.model.clone(), keys, embeddings_url(base), false, 10)
    }

    fn with_endpoint(
        model: String,
        api_keys: Vec<String>,
        endpoint: String,
        send_input_type: bool,
        batch_size: usize,
    ) -> Result<Self> {
        if api_keys.is_empty() {
            bail!("embedding client requires at least one API key");
        }
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("build reqwest client")?;
        let query_http = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("build query reqwest client")?;
        Ok(Self {
            inner: Arc::new(VoyageInner {
                http,
                query_http,
                model,
                api_keys,
                endpoint,
                send_input_type,
                batch_size: batch_size.max(1),
                key_cursor: AtomicUsize::new(0),
            }),
        })
    }

    /// Return the configured embedding model name.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// Embed a single query string with bounded retry.
    ///
    /// Uses `input_type: "query"`. On 429 from all keys, waits 2 s and retries
    /// once. A second 429 wave returns `Err`. Non-429 errors return `Err` immediately.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let n_keys = self.inner.api_keys.len();
        let start_cursor = self.inner.key_cursor.fetch_add(1, Ordering::Relaxed) % n_keys;

        // First pass — try each key once (30s timeout per attempt).
        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.inner.api_keys[key_idx];
            match self
                .try_embed_query_with_key(key, &texts, InputType::Query)
                .await
            {
                Ok(mut embeddings) => {
                    return embeddings
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("embedding API returned empty embeddings"));
                }
                Err(EmbedError::RateLimited) => {
                    warn!(
                        key_index = key_idx,
                        "embedding API 429 on query embed — trying next key"
                    );
                }
                Err(EmbedError::Other(e)) => return Err(e),
            }
        }

        // All keys 429 — one backoff attempt (2 s), then return Err.
        warn!("all embedding API keys rate-limited on query embed; backing off 2s");
        tokio::time::sleep(Duration::from_secs(2)).await;

        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.inner.api_keys[key_idx];
            match self
                .try_embed_query_with_key(key, &texts, InputType::Query)
                .await
            {
                Ok(mut embeddings) => {
                    return embeddings
                        .pop()
                        .ok_or_else(|| anyhow::anyhow!("embedding API returned empty embeddings"));
                }
                Err(EmbedError::RateLimited) => continue,
                Err(EmbedError::Other(e)) => return Err(e),
            }
        }

        anyhow::bail!("embedding API query embed still rate-limited after backoff")
    }

    /// Embed texts in batches of up to the configured batch size.
    /// Returns one Vec<f32> per input.
    pub async fn embed(&self, texts: &[String], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.inner.batch_size) {
            let embeddings = self.embed_batch(batch, input_type).await?;
            all_embeddings.extend(embeddings);
        }
        Ok(all_embeddings)
    }

    /// Embed one provider-request batch, splitting internally if `texts`
    /// exceeds the configured batch size. Public so the pipeline can drive
    /// batching manually and report per-batch progress between awaits.
    pub async fn embed_batch(
        &self,
        texts: &[String],
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.len() > self.inner.batch_size {
            // Callers may chunk by the old 128 constant — re-split to the
            // provider's real cap rather than 400ing.
            let mut all = Vec::with_capacity(texts.len());
            for sub in texts.chunks(self.inner.batch_size) {
                all.extend(self.embed_one_request(sub, input_type).await?);
            }
            return Ok(all);
        }
        self.embed_one_request(texts, input_type).await
    }

    async fn embed_one_request(
        &self,
        texts: &[String],
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        let n_keys = self.inner.api_keys.len();
        let start_cursor = self.inner.key_cursor.fetch_add(1, Ordering::Relaxed) % n_keys;

        // Try each key once before falling back to exponential backoff.
        for offset in 0..n_keys {
            let key_idx = (start_cursor + offset) % n_keys;
            let key = &self.inner.api_keys[key_idx];

            match self.try_embed_with_key(key, texts, input_type).await {
                Ok(embeddings) => return Ok(embeddings),
                Err(EmbedError::RateLimited) => {
                    warn!(key_index = key_idx, "embedding API 429 — trying next key");
                }
                // Non-429 error: abort immediately, old data untouched.
                Err(EmbedError::Other(e)) => return Err(e),
            }
        }

        // All keys returned 429 — exponential backoff, retry indefinitely.
        let mut delay_secs: u64 = 2;
        loop {
            warn!(
                delay_secs = delay_secs,
                "all embedding API keys rate-limited; backing off"
            );
            tokio::time::sleep(Duration::from_secs(delay_secs)).await;

            for offset in 0..n_keys {
                let key_idx = (start_cursor + offset) % n_keys;
                let key = &self.inner.api_keys[key_idx];
                match self.try_embed_with_key(key, texts, input_type).await {
                    Ok(embeddings) => {
                        info!("embedding API call succeeded after backoff");
                        return Ok(embeddings);
                    }
                    Err(EmbedError::RateLimited) => continue,
                    Err(EmbedError::Other(e)) => return Err(e),
                }
            }

            delay_secs = (delay_secs * 2).min(60);
        }
    }

    async fn try_embed_with_key(
        &self,
        key: &str,
        texts: &[String],
        input_type: InputType,
    ) -> std::result::Result<Vec<Vec<f32>>, EmbedError> {
        self.try_embed_with_key_using(&self.inner.http, key, texts, input_type)
            .await
    }

    async fn try_embed_query_with_key(
        &self,
        key: &str,
        texts: &[String],
        input_type: InputType,
    ) -> std::result::Result<Vec<Vec<f32>>, EmbedError> {
        self.try_embed_with_key_using(&self.inner.query_http, key, texts, input_type)
            .await
    }

    async fn try_embed_with_key_using(
        &self,
        client: &Client,
        key: &str,
        texts: &[String],
        input_type: InputType,
    ) -> std::result::Result<Vec<Vec<f32>>, EmbedError> {
        let body = EmbedRequest {
            model: &self.inner.model,
            input: texts,
            input_type: self.inner.send_input_type.then(|| input_type.as_str()),
        };

        let mut req = client.post(&self.inner.endpoint).json(&body);
        // Skip the Authorization header for empty keys (unauthenticated local
        // servers reject malformed bearer tokens rather than ignoring them).
        if !key.is_empty() {
            req = req.bearer_auth(key);
        }
        let response = req.send().await.map_err(|e| EmbedError::Other(e.into()))?;

        let status = response.status();

        if status.as_u16() == 429 {
            return Err(EmbedError::RateLimited);
        }

        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(EmbedError::Other(anyhow::anyhow!(
                "embedding API error {}: {}",
                status,
                text
            )));
        }

        let resp: EmbedResponse = response
            .json()
            .await
            .map_err(|e| EmbedError::Other(e.into()))?;

        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

enum EmbedError {
    RateLimited,
    Other(anyhow::Error),
}
