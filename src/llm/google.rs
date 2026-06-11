use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use super::ToolDef;

// ─── Single-turn types ───────────────────────────────────────────────────

#[derive(Serialize)]
struct GeminiRequest {
    system_instruction: SystemInstruction,
    contents: Vec<Content>,
    generation_config: GenerationConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolsBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<ToolConfig>,
}

#[derive(Serialize)]
struct SystemInstruction {
    parts: Vec<Part>,
}

#[derive(Serialize, Clone)]
struct Content {
    role: String,
    parts: Vec<Part>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
enum Part {
    // Order matters for untagged (de)serialization: the FunctionCall variant is
    // listed before Text so a part carrying `functionCall` is never misparsed.
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: FunctionCallPart,
        /// Part-level sibling of `functionCall` (Gemini 2.5/3.x thinking). Echoed
        /// back verbatim in history; omitted from the wire when absent.
        #[serde(
            rename = "thoughtSignature",
            skip_serializing_if = "Option::is_none",
            default
        )]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: FunctionResponsePart,
    },
    Text {
        text: String,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FunctionCallPart {
    pub name: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub args: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone)]
struct FunctionResponsePart {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    response: serde_json::Value,
}

#[derive(Serialize)]
struct GenerationConfig {
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
}

#[derive(Serialize)]
struct ToolsBlock {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Serialize)]
struct FunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
struct ToolConfig {
    #[serde(rename = "functionCallingConfig")]
    function_calling_config: FunctionCallingConfig,
}

#[derive(Serialize)]
struct FunctionCallingConfig {
    mode: String,
}

// ─── Response types ──────────────────────────────────────────────────────

fn deserialize_null_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(|opt| opt.unwrap_or_default())
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<Candidate>>,
    error: Option<GeminiError>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Deserialize)]
struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
    #[serde(rename = "cachedContentTokenCount")]
    cached_content_token_count: Option<u32>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Option<CandidateContent>,
}

#[derive(Deserialize)]
struct CandidateContent {
    #[serde(default, deserialize_with = "deserialize_null_default")]
    parts: Vec<ResponsePart>,
}

#[derive(Deserialize)]
struct ResponsePart {
    text: Option<String>,
    #[serde(rename = "functionCall")]
    function_call: Option<FunctionCallPart>,
    /// Part-level signature attached to a `functionCall` when thinking is active.
    #[serde(rename = "thoughtSignature", default)]
    thought_signature: Option<String>,
}

#[derive(Deserialize)]
struct GeminiError {
    message: String,
}

// ─── Public API ──────────────────────────────────────────────────────────

fn log_cache_metrics(usage: &Option<UsageMetadata>) {
    if let Some(u) = usage {
        let prompt = u.prompt_token_count.unwrap_or(0);
        let completion = u.candidates_token_count.unwrap_or(0);
        let cached = u.cached_content_token_count.unwrap_or(0);
        info!(
            prompt_tokens = prompt,
            completion_tokens = completion,
            cached_tokens = cached,
            cache_hit_pct = if prompt > 0 {
                (cached as f64 / prompt as f64 * 100.0) as u32
            } else {
                0
            },
            "gemini cache metrics"
        );
    }
}

pub async fn complete(
    http: &Client,
    model: &str,
    api_key: &str,
    system: &str,
    user: &str,
    temperature: f32,
    structured: bool,
) -> Result<String> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let body = GeminiRequest {
        system_instruction: SystemInstruction {
            parts: vec![Part::Text {
                text: system.to_owned(),
            }],
        },
        contents: vec![Content {
            role: "user".to_owned(),
            parts: vec![Part::Text {
                text: user.to_owned(),
            }],
        }],
        generation_config: GenerationConfig {
            temperature,
            response_mime_type: structured.then(|| "application/json".to_owned()),
        },
        tools: None,
        tool_config: None,
    };

    let resp = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("Gemini HTTP request failed")?;

    let status = resp.status();
    let text = resp
        .text()
        .await
        .context("failed to read Gemini response body")?;

    if !status.is_success() {
        bail!("Gemini API returned HTTP {status}: {text}");
    }

    let parsed: GeminiResponse = serde_json::from_str(&text).with_context(|| {
        let preview = if text.len() > 512 {
            &text[..512]
        } else {
            &text
        };
        warn!(body_preview = %preview, "Gemini response parse failed");
        "failed to parse Gemini response JSON"
    })?;

    if let Some(err) = parsed.error {
        bail!("Gemini API error: {}", err.message);
    }

    log_cache_metrics(&parsed.usage_metadata);

    let result_text = parsed
        .candidates
        .and_then(|c| c.into_iter().next())
        .and_then(|c| c.content)
        .and_then(|c| c.parts.into_iter().next())
        .and_then(|p| p.text)
        .unwrap_or_default();

    Ok(result_text)
}

/// A function call returned by Gemini, paired with its part-level
/// `thoughtSignature` (if any) so the caller can echo it back in history.
pub struct GeminiToolCall {
    pub call: FunctionCallPart,
    pub thought_signature: Option<String>,
}

/// Result of a single turn in the tool-calling loop.
pub enum ToolTurnResult {
    /// Model returned text (done).
    Text(String),
    /// Model requested function calls.
    ToolCalls(Vec<GeminiToolCall>),
}

/// Send a multi-turn tool-calling request. `contents` is the full conversation
/// history (user/model/functionResponse turns). Returns either a text response
/// or a list of function calls the model wants executed.
#[allow(clippy::too_many_arguments)]
pub async fn complete_with_tools(
    http: &Client,
    model: &str,
    api_key: &str,
    system: &str,
    contents: &[super::ChatMessage],
    tools: &[ToolDef],
    temperature: f32,
    force_tool_use: bool,
) -> Result<ToolTurnResult> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let gemini_contents: Vec<Content> = contents
        .iter()
        .map(|m| match m {
            super::ChatMessage::User(text) => Content {
                role: "user".to_owned(),
                parts: vec![Part::Text { text: text.clone() }],
            },
            super::ChatMessage::ModelToolCalls(calls) => Content {
                role: "model".to_owned(),
                parts: calls
                    .iter()
                    .map(|c| Part::FunctionCall {
                        function_call: FunctionCallPart {
                            name: c.name.clone(),
                            id: c.id.clone(),
                            args: c.args.clone(),
                        },
                        // Echo the signature verbatim — Gemini requires it on replay.
                        thought_signature: c.thought_signature.clone(),
                    })
                    .collect(),
            },
            super::ChatMessage::ToolResults(results) => Content {
                role: "user".to_owned(),
                parts: results
                    .iter()
                    .map(|r| Part::FunctionResponse {
                        function_response: FunctionResponsePart {
                            name: r.name.clone(),
                            id: r.id.clone(),
                            response: serde_json::json!({ "result": &r.content }),
                        },
                    })
                    .collect(),
            },
        })
        .collect();

    let declarations: Vec<FunctionDeclaration> = tools
        .iter()
        .map(|t| FunctionDeclaration {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.parameters.clone(),
        })
        .collect();

    let body = GeminiRequest {
        system_instruction: SystemInstruction {
            parts: vec![Part::Text {
                text: system.to_owned(),
            }],
        },
        contents: gemini_contents,
        generation_config: GenerationConfig {
            temperature,
            response_mime_type: None,
        },
        tools: Some(vec![ToolsBlock {
            function_declarations: declarations,
        }]),
        tool_config: Some(ToolConfig {
            function_calling_config: FunctionCallingConfig {
                // ANY = the model MUST call a function (cannot answer with prose);
                // AUTO = it may finish with text. Forced while no chunk is yet
                // committed so the agent can't reply to the query directly.
                mode: if force_tool_use { "ANY" } else { "AUTO" }.to_owned(),
            },
        }),
    };

    let resp = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("Gemini tool-calling HTTP request failed")?;

    let status = resp.status();
    let text = resp
        .text()
        .await
        .context("failed to read Gemini response body")?;

    if !status.is_success() {
        bail!("Gemini API returned HTTP {status}: {text}");
    }

    let parsed: GeminiResponse = serde_json::from_str(&text).with_context(|| {
        let preview = if text.len() > 512 {
            &text[..512]
        } else {
            &text
        };
        warn!(body_preview = %preview, "Gemini tool-call response parse failed");
        "failed to parse Gemini response JSON"
    })?;

    if let Some(err) = parsed.error {
        bail!("Gemini API error: {}", err.message);
    }

    log_cache_metrics(&parsed.usage_metadata);

    let parts = parsed
        .candidates
        .and_then(|c| c.into_iter().next())
        .and_then(|c| c.content)
        .map(|c| c.parts)
        .unwrap_or_default();

    let mut function_calls: Vec<GeminiToolCall> = Vec::new();
    let mut text_parts: Vec<String> = Vec::new();

    for part in parts {
        if let Some(fc) = part.function_call {
            function_calls.push(GeminiToolCall {
                call: fc,
                thought_signature: part.thought_signature,
            });
        } else if let Some(t) = part.text {
            text_parts.push(t);
        }
    }

    if !function_calls.is_empty() {
        Ok(ToolTurnResult::ToolCalls(function_calls))
    } else {
        Ok(ToolTurnResult::Text(text_parts.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `functionCall` response part carrying a part-level `thoughtSignature`
    /// (Gemini 2.5/3.x thinking) must deserialize with the signature captured.
    #[test]
    fn response_part_captures_thought_signature() {
        let json = r#"{
            "functionCall": { "name": "add_chunks", "args": { "chunks": [] } },
            "thoughtSignature": "abc123=="
        }"#;
        let part: ResponsePart = serde_json::from_str(json).expect("parse part");
        assert_eq!(part.thought_signature.as_deref(), Some("abc123=="));
        assert!(part.function_call.is_some());
    }

    /// A response part without a signature parses with `None` (non-thinking
    /// models, or text parts).
    #[test]
    fn response_part_without_signature_is_none() {
        let json = r#"{ "functionCall": { "name": "q", "args": {} } }"#;
        let part: ResponsePart = serde_json::from_str(json).expect("parse part");
        assert!(part.thought_signature.is_none());
    }

    /// When we replay a model turn, the `Part::FunctionCall` MUST serialize the
    /// signature back under `thoughtSignature` — this is the fix for the 400
    /// "Function call is missing a thought_signature".
    #[test]
    fn function_call_part_echoes_thought_signature() {
        let part = Part::FunctionCall {
            function_call: FunctionCallPart {
                name: "add_chunks".to_owned(),
                id: None,
                args: serde_json::json!({ "chunks": [] }),
            },
            thought_signature: Some("sig-xyz".to_owned()),
        };
        let v = serde_json::to_value(&part).expect("serialize");
        assert_eq!(
            v.get("thoughtSignature").and_then(|s| s.as_str()),
            Some("sig-xyz")
        );
        assert!(v.get("functionCall").is_some());
    }

    /// Absent signature must be OMITTED from the wire (not serialized as null) so
    /// non-thinking turns produce the same payload as before.
    #[test]
    fn function_call_part_omits_absent_signature() {
        let part = Part::FunctionCall {
            function_call: FunctionCallPart {
                name: "q".to_owned(),
                id: None,
                args: serde_json::json!({}),
            },
            thought_signature: None,
        };
        let v = serde_json::to_value(&part).expect("serialize");
        assert!(
            v.get("thoughtSignature").is_none(),
            "absent signature must not appear on the wire"
        );
    }

    /// End-to-end of the type bridge: a parsed `GeminiToolCall` with a signature
    /// round-trips through a rebuilt `Part::FunctionCall` and re-emits the same
    /// signature — mirrors response → ToolCall → request-echo.
    #[test]
    fn signature_round_trips_response_to_request() {
        let resp = r#"{
            "functionCall": { "name": "add_chunks", "args": { "chunks": [] } },
            "thoughtSignature": "ROUND-TRIP"
        }"#;
        let part: ResponsePart = serde_json::from_str(resp).expect("parse");
        let fc = part.function_call.expect("has call");
        let sig = part.thought_signature;

        // Rebuild the request part exactly as complete_with_tools does.
        let echoed = Part::FunctionCall {
            function_call: fc,
            thought_signature: sig,
        };
        let v = serde_json::to_value(&echoed).expect("serialize");
        assert_eq!(
            v.get("thoughtSignature").and_then(|s| s.as_str()),
            Some("ROUND-TRIP")
        );
    }

    #[test]
    fn candidate_without_content_parses() {
        let json = r#"{"candidates":[{"finishReason":"SAFETY"}]}"#;
        let resp: GeminiResponse = serde_json::from_str(json).expect("parse");
        let parts = resp
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content)
            .map(|c| c.parts)
            .unwrap_or_default();
        assert!(parts.is_empty());
    }

    #[test]
    fn content_with_null_parts_parses() {
        let json = r#"{"candidates":[{"content":{"parts":null,"role":"model"}}]}"#;
        let resp: GeminiResponse = serde_json::from_str(json).expect("parse");
        let parts = resp
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content)
            .map(|c| c.parts)
            .unwrap_or_default();
        assert!(parts.is_empty());
    }

    #[test]
    fn function_call_without_args_parses() {
        let json = r#"{"functionCall":{"name":"get_info"}}"#;
        let part: ResponsePart = serde_json::from_str(json).expect("parse");
        let fc = part.function_call.expect("has call");
        assert_eq!(fc.name, "get_info");
        assert!(fc.args.is_null());
    }

    #[test]
    fn usage_metadata_with_cached_tokens_deserializes() {
        let json = r#"{
            "candidates": [{"content": {"parts": [{"text": "hello"}], "role": "model"}}],
            "usageMetadata": {
                "promptTokenCount": 2000,
                "candidatesTokenCount": 150,
                "cachedContentTokenCount": 1800
            }
        }"#;
        let resp: GeminiResponse = serde_json::from_str(json).expect("parse");
        let usage = resp.usage_metadata.expect("has usage");
        assert_eq!(usage.prompt_token_count, Some(2000));
        assert_eq!(usage.candidates_token_count, Some(150));
        assert_eq!(usage.cached_content_token_count, Some(1800));
    }

    #[test]
    fn usage_metadata_without_cache_deserializes() {
        let json = r#"{
            "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}}],
            "usageMetadata": {
                "promptTokenCount": 500,
                "candidatesTokenCount": 50
            }
        }"#;
        let resp: GeminiResponse = serde_json::from_str(json).expect("parse");
        let usage = resp.usage_metadata.expect("has usage");
        assert_eq!(usage.prompt_token_count, Some(500));
        assert!(usage.cached_content_token_count.is_none());
    }

    #[test]
    fn response_without_usage_metadata_deserializes() {
        let json = r#"{"candidates": [{"content": {"parts": [{"text": "hi"}], "role": "model"}}]}"#;
        let resp: GeminiResponse = serde_json::from_str(json).expect("parse");
        assert!(resp.usage_metadata.is_none());
        let text = resp
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content)
            .and_then(|c| c.parts.into_iter().next())
            .and_then(|p| p.text);
        assert_eq!(text.as_deref(), Some("hi"));
    }
}
