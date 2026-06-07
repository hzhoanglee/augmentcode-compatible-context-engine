use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Option<Vec<Choice>>,
    error: Option<OpenAIError>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIError {
    message: String,
}

/// Guarantee the literal token "json" is present in the messages when
/// `structured` json_object mode is on (OpenAI rejects the request otherwise).
/// Returns the system message to send: the original when the token is already
/// present in either message (the common case — the structured prompt says
/// "JSON"), or the original plus a minimal directive when it is somehow absent.
fn ensure_json_token(structured: bool, system: &str, user: &str) -> String {
    let missing = structured
        && !system.to_lowercase().contains("json")
        && !user.to_lowercase().contains("json");
    if missing {
        format!("{system}\nRespond in JSON.")
    } else {
        system.to_owned()
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
    let url = "https://api.openai.com/v1/chat/completions";

    // OpenAI's json_object mode rejects (HTTP 400) any request whose messages do
    // not contain the literal word "json". The reranker's structured prompt
    // already says "JSON" repeatedly, but guarantee it structurally here so a
    // future prompt edit can never silently 400 every rerank call.
    let system_owned = ensure_json_token(structured, system, user);

    let body = ChatRequest {
        model: model.to_owned(),
        messages: vec![
            Message { role: "system".to_owned(), content: system_owned },
            Message { role: "user".to_owned(), content: user.to_owned() },
        ],
        temperature,
        response_format: structured.then(|| ResponseFormat { kind: "json_object".to_owned() }),
    };

    let resp = http
        .post(url)
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&body)
        .send()
        .await
        .context("OpenAI HTTP request failed")?;

    let status = resp.status();
    let text = resp.text().await.context("failed to read OpenAI response body")?;

    if !status.is_success() {
        bail!("OpenAI API returned HTTP {status}: {text}");
    }

    let parsed: ChatResponse = serde_json::from_str(&text)
        .context("failed to parse OpenAI response JSON")?;

    if let Some(err) = parsed.error {
        bail!("OpenAI API error: {}", err.message);
    }

    let result_text = parsed.choices
        .and_then(|c| c.into_iter().next())
        .and_then(|c| c.message.content)
        .unwrap_or_default();

    Ok(result_text)
}

#[cfg(test)]
mod tests {
    use super::ensure_json_token;

    #[test]
    fn token_present_in_system_is_unchanged() {
        let s = ensure_json_token(true, "Respond with a JSON object.", "rank these");
        assert_eq!(s, "Respond with a JSON object.");
    }

    #[test]
    fn token_present_in_user_leaves_system_unchanged() {
        let s = ensure_json_token(true, "You are a ranker.", "reply as json please");
        assert_eq!(s, "You are a ranker.");
    }

    #[test]
    fn token_absent_appends_directive() {
        let s = ensure_json_token(true, "You are a ranker.", "rank these chunks");
        assert!(s.to_lowercase().contains("json"), "must inject the json token");
    }

    #[test]
    fn not_structured_never_modifies() {
        // XML mode: no json_object request, so no token requirement.
        let s = ensure_json_token(false, "You are a ranker.", "rank these chunks");
        assert_eq!(s, "You are a ranker.");
    }
}
