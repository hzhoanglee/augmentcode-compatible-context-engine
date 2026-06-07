use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct GeminiRequest {
    system_instruction: SystemInstruction,
    contents: Vec<Content>,
    generation_config: GenerationConfig,
}

#[derive(Serialize)]
struct SystemInstruction {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Content {
    role: String,
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Serialize)]
struct GenerationConfig {
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<Candidate>>,
    error: Option<GeminiError>,
}

#[derive(Deserialize)]
struct Candidate {
    content: CandidateContent,
}

#[derive(Deserialize)]
struct CandidateContent {
    parts: Vec<ResponsePart>,
}

#[derive(Deserialize)]
struct ResponsePart {
    text: Option<String>,
}

#[derive(Deserialize)]
struct GeminiError {
    message: String,
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
            parts: vec![Part { text: system.to_owned() }],
        },
        contents: vec![Content {
            role: "user".to_owned(),
            parts: vec![Part { text: user.to_owned() }],
        }],
        generation_config: GenerationConfig {
            temperature,
            response_mime_type: structured.then(|| "application/json".to_owned()),
        },
    };

    let resp = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("Gemini HTTP request failed")?;

    let status = resp.status();
    let text = resp.text().await.context("failed to read Gemini response body")?;

    if !status.is_success() {
        bail!("Gemini API returned HTTP {status}: {text}");
    }

    let parsed: GeminiResponse = serde_json::from_str(&text)
        .context("failed to parse Gemini response JSON")?;

    if let Some(err) = parsed.error {
        bail!("Gemini API error: {}", err.message);
    }

    let result_text = parsed.candidates
        .and_then(|c| c.into_iter().next())
        .and_then(|c| c.content.parts.into_iter().next())
        .and_then(|p| p.text)
        .unwrap_or_default();

    Ok(result_text)
}
