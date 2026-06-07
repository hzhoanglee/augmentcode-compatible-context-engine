use std::time::Instant;
use regex::Regex;
use tracing::warn;
use crate::llm::LlmClient;
use crate::query::merger::MergeChunk;

#[derive(Debug, Clone, serde::Serialize)]
pub struct RerankOutput {
    pub reranked_indices: Vec<usize>,
    /// Per-position line selections, aligned 1:1 with `reranked_indices`.
    /// `Some(ranges)` = LLM chose absolute line ranges to keep for that chunk
    /// (sorted, merged, clamped to chunk bounds); `None` = keep whole chunk.
    pub line_selections: Vec<Option<Vec<(u32, u32)>>>,
    pub raw_request: String,
    pub raw_response: String,
    pub elapsed_ms: u64,
    pub fallback_used: bool,
    pub skip_reason: Option<String>,
}

/// Padding added on each side of an LLM-selected range before clamping to the
/// chunk bounds — absorbs off-by-a-line selections.
const RANGE_PAD: u32 = 2;

pub async fn rerank(
    query: &str,
    chunks: &[MergeChunk],
    numbered: &[Option<String>],
    min_prune_lines: u32,
    llm_client: Option<&LlmClient>,
) -> RerankOutput {
    let n = chunks.len();
    let all_indices: Vec<usize> = (0..n).collect();

    let client = match llm_client {
        Some(c) => c,
        None => return RerankOutput {
            reranked_indices: all_indices,
            line_selections: vec![None; n],
            raw_request: String::new(),
            raw_response: String::new(),
            elapsed_ms: 0,
            fallback_used: false,
            skip_reason: Some("no LLM API key configured".to_owned()),
        },
    };

    if chunks.is_empty() {
        return RerankOutput {
            reranked_indices: vec![],
            line_selections: vec![],
            raw_request: String::new(),
            raw_response: String::new(),
            elapsed_ms: 0,
            fallback_used: false,
            skip_reason: None,
        };
    }

    let structured = client.structured_output_active();

    let common_intro = "You are a code search relevance ranker. \
        Given a query and numbered code chunks with metadata (relevance score, callers count, \
        callees count, flow membership), your job is to rank the chunks by relevance to the query. \
        OMIT chunks that are not relevant to the query. \
        Higher score, more callers, and flow membership indicate higher structural importance. \
        When both source code and documentation chunks are relevant to the query, \
        prefer source code over documentation because code is the source of truth. \
        Documentation can be outdated or inaccurate, but the code always reflects actual behavior. \
        Each code line is prefixed with its absolute line number (\"123: code\"). ";

    let element_spec = "Each element MUST be an object identified by `i` (the chunk index), in ONE of two forms: \
        to narrow a large chunk to the relevant parts, use \
        {\"i\": <index>, \"lines\": [[start, end], ...]} where `lines` are absolute line-number \
        ranges to keep from that chunk; \
        to keep an entire chunk (small chunks, or chunks that are wholly relevant), use \
        {\"i\": <index>, \"keep\": \"full\"}. \
        Only include chunks that are actually relevant to the query.";

    let system = if structured {
        format!(
            "{common_intro}\
            Respond with a single JSON object with exactly one key, `ranked_indices`, whose value \
            is a JSON array of objects ordered from most relevant to least relevant. \
            {element_spec} \
            Output only the JSON object — no prose, no code fences."
        )
    } else {
        format!(
            "{common_intro}\
            Your output MUST contain a pair of XML tags called ranked_indices. \
            Between the opening <ranked_indices> tag and the closing </ranked_indices> tag, \
            place a JSON array of objects, ordered from most relevant to least relevant. \
            {element_spec} \
            Do not include any other text between the tags, only the JSON array."
        )
    };
    let system = system.as_str();

    // Build user prompt with chunk entries. Use the disk-numbered content
    // (same text the server will slice) so the line numbers the LLM selects map
    // exactly to what is returned. Fall back to stored content when the file
    // could not be read — such chunks are NOT line-prunable downstream.
    let mut entries = Vec::with_capacity(n);
    for (i, chunk) in chunks.iter().enumerate() {
        let meta_str = format!("score={:.2}", chunk.score);
        let symbol_display = chunk.symbol.as_deref().unwrap_or("no symbol");

        let raw = numbered
            .get(i)
            .and_then(|c| c.as_deref())
            .unwrap_or(&chunk.content);
        // Truncate content at 100 lines
        let content = truncate_content(raw, 100);

        let entry = format!(
            "[{i}] {meta_str} | {}:{}-{} ({symbol_display})\n<content chunk-index=\"{i}\">\n{content}\n</content>",
            chunk.file, chunk.line_start, chunk.line_end
        );
        entries.push(entry);
    }

    let chunks_text = entries.join("\n---\n");
    let user_prompt = if structured {
        format!(
            "Query: {query}\n\nChunks:\n{chunks_text}\n\n\
             Now rank the chunks by relevance. Respond with a JSON object \
             {{\"ranked_indices\": [ ... ]}} whose array holds objects — \
             {{\"i\":index,\"lines\":[[start,end]]}} to narrow, or {{\"i\":index,\"keep\":\"full\"}} \
             to keep the whole chunk — from most to least relevant."
        )
    } else {
        format!(
            "Query: {query}\n\nChunks:\n{chunks_text}\n\n\
             Now rank the chunks by relevance. \
             Write the opening tag <ranked_indices>, then a JSON array of objects — \
             {{\"i\":index,\"lines\":[[start,end]]}} to narrow, or {{\"i\":index,\"keep\":\"full\"}} \
             to keep the whole chunk — from most to least relevant, then the closing tag \
             </ranked_indices>."
        )
    };

    let raw_request = format!("[System]\n{system}\n\n[User]\n{user_prompt}");

    let start = Instant::now();
    let result = client.complete(system, &user_prompt, 0.0, structured).await;
    let elapsed_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(response) => {
            let mut output = parse_rerank_response(&response, chunks, min_prune_lines, elapsed_ms, structured);
            output.raw_request = raw_request;
            output
        }
        Err(e) => {
            warn!(error = %e, "LLM rerank call failed, using fallback order");
            RerankOutput {
                reranked_indices: all_indices,
                line_selections: vec![None; n],
                raw_request,
                raw_response: String::new(),
                elapsed_ms,
                fallback_used: true,
                skip_reason: Some(format!("LLM request failed: {e}")),
            }
        }
    }
}

fn parse_rerank_response(
    response: &str,
    chunks: &[MergeChunk],
    min_prune_lines: u32,
    elapsed_ms: u64,
    structured: bool,
) -> RerankOutput {
    let n = chunks.len();
    let all_indices: Vec<usize> = (0..n).collect();

    // Unwrap the JSON array of ranking entries from the response.
    //
    // Structured (native JSON) mode: the whole response is a JSON object
    // `{"ranked_indices": [ ... ]}`. Parse it and pull the array out.
    // XML mode: the array lives between <ranked_indices> tags (with a
    // markdown-fence fallback), then is parsed as a bare JSON array.
    //
    // Either way, a missing/non-array/unparseable payload converges on the SAME
    // fallback below: original order with `fallback_used: true`.
    let parsed: Vec<serde_json::Value> = if structured {
        match serde_json::from_str::<serde_json::Value>(response.trim()) {
            Ok(serde_json::Value::Object(map)) => match map.get("ranked_indices") {
                Some(serde_json::Value::Array(arr)) => arr.clone(),
                _ => {
                    warn!(raw = %response, "structured rerank response missing `ranked_indices` array");
                    return RerankOutput {
                        reranked_indices: all_indices,
                        line_selections: vec![None; n],
                        raw_request: String::new(),
                        raw_response: response.to_owned(),
                        elapsed_ms,
                        fallback_used: true,
                        skip_reason: Some("structured response missing ranked_indices".to_owned()),
                    };
                }
            },
            _ => {
                warn!(raw = %response, "failed to parse structured rerank response as a JSON object");
                return RerankOutput {
                    reranked_indices: all_indices,
                    line_selections: vec![None; n],
                    raw_request: String::new(),
                    raw_response: response.to_owned(),
                    elapsed_ms,
                    fallback_used: true,
                    skip_reason: Some("failed to parse structured response".to_owned()),
                };
            }
        }
    } else {
        // Try XML tags first
        let re = Regex::new(r"(?s)<ranked_indices>\s*(.*?)\s*</ranked_indices>").unwrap();
        let text = if let Some(caps) = re.captures(response) {
            caps.get(1).unwrap().as_str().to_owned()
        } else {
            // Fallback: try raw response, strip markdown fences
            let trimmed = response.trim();
            if trimmed.starts_with("```") {
                trimmed.lines()
                    .filter(|line| !line.starts_with("```"))
                    .collect::<Vec<_>>()
                    .join("\n")
                    .trim()
                    .to_owned()
            } else {
                trimmed.to_owned()
            }
        };

        match serde_json::from_str(&text) {
            Ok(arr) => arr,
            Err(_) => {
                warn!(raw = %response, "failed to parse rerank response as JSON array");
                return RerankOutput {
                    reranked_indices: all_indices,
                    line_selections: vec![None; n],
                    raw_request: String::new(),
                    raw_response: response.to_owned(),
                    elapsed_ms,
                    fallback_used: true,
                    skip_reason: Some("failed to parse LLM response".to_owned()),
                };
            }
        }
    };

    let mut reranked_indices: Vec<usize> = Vec::new();
    let mut line_selections: Vec<Option<Vec<(u32, u32)>>> = Vec::new();

    for entry in &parsed {
        // Element is a bare integer index (back-compat safety net), or an object:
        //   {"i": idx, "lines": [[s,e],...]}  → narrow to ranges
        //   {"i": idx, "keep": "full"}        → whole chunk (no `lines` field)
        let (idx, raw_lines) = if let Some(i) = entry.as_u64() {
            (i as usize, None)
        } else if let Some(obj) = entry.as_object() {
            let Some(i) = obj.get("i").and_then(|v| v.as_u64()) else { continue };
            (i as usize, obj.get("lines").and_then(|v| v.as_array()))
        } else {
            continue;
        };

        // Drop indices outside the candidate set.
        if idx >= n {
            continue;
        }
        let chunk = &chunks[idx];

        let selection = match raw_lines {
            // Whole chunk: bare int, {"keep":"full"} (no `lines`), or empty `lines`.
            None => None,
            Some(arr) if arr.is_empty() => None,
            Some(arr) => {
                // Small chunks are never line-pruned (1C policy).
                if chunk.line_end.saturating_sub(chunk.line_start) < min_prune_lines {
                    None
                } else {
                    sanitize_ranges(arr, chunk.line_start, chunk.line_end)
                }
            }
        };

        reranked_indices.push(idx);
        line_selections.push(selection);
    }

    if reranked_indices.is_empty() {
        // LLM legitimately judged nothing relevant — honor that (empty result),
        // matching prior behavior. Not a fallback.
        return RerankOutput {
            reranked_indices: vec![],
            line_selections: vec![],
            raw_request: String::new(),
            raw_response: response.to_owned(),
            elapsed_ms,
            fallback_used: false,
            skip_reason: None,
        };
    }

    RerankOutput {
        reranked_indices,
        line_selections,
        raw_request: String::new(),
        raw_response: response.to_owned(),
        elapsed_ms,
        fallback_used: false,
        skip_reason: None,
    }
}

/// Validate and normalize the LLM's `lines` array for one chunk:
/// - parse each [start, end] pair (skip malformed / start>end),
/// - pad by RANGE_PAD and clamp to [chunk_start, chunk_end],
/// - sort by start and merge overlapping/adjacent ranges (gap <= 1).
///
/// Returns `None` if no valid range survives (caller keeps the whole chunk).
fn sanitize_ranges(
    arr: &[serde_json::Value],
    chunk_start: u32,
    chunk_end: u32,
) -> Option<Vec<(u32, u32)>> {
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for pair in arr {
        let Some(p) = pair.as_array() else { continue };
        if p.len() != 2 {
            continue;
        }
        let (Some(s), Some(e)) = (p[0].as_u64(), p[1].as_u64()) else { continue };
        let (s, e) = (s as u32, e as u32);
        if s > e {
            continue;
        }
        // Pad then clamp to chunk bounds.
        let s = s.saturating_sub(RANGE_PAD).max(chunk_start);
        let e = e.saturating_add(RANGE_PAD).min(chunk_end);
        if s > e {
            continue;
        }
        ranges.push((s, e));
    }

    if ranges.is_empty() {
        return None;
    }

    // Sort by start, then merge overlapping/adjacent ranges (gap <= 1).
    ranges.sort_unstable_by_key(|&(s, _)| s);
    let mut merged: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
    for (s, e) in ranges {
        if let Some(last) = merged.last_mut()
            && s <= last.1 + 1
        {
            last.1 = last.1.max(e);
            continue;
        }
        merged.push((s, e));
    }
    Some(merged)
}

fn truncate_content(content: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= max_lines {
        return content.to_owned();
    }
    let half = max_lines / 2;
    let truncated_count = lines.len() - max_lines;
    let mut result = lines[..half].join("\n");
    result.push_str(&format!("\n... ({truncated_count} lines truncated) ...\n"));
    result.push_str(&lines[lines.len() - half..].join("\n"));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chunk(line_start: u32, line_end: u32) -> MergeChunk {
        MergeChunk {
            file: "/x.rs".to_owned(),
            line_start,
            line_end,
            score: 1.0,
            content: "stored".to_owned(),
            symbol: None,
        }
    }

    // ── sanitize_ranges ──────────────────────────────────────────────────

    #[test]
    fn sanitize_merges_overlapping_and_adjacent() {
        // Chunk 100..200. Ranges [110,120] and [121,130] are gap==1 → merge.
        // With pad ±2: [108,122] and [119,132] overlap → one [108,132].
        let arr = vec![json!([110, 120]), json!([121, 130])];
        let out = sanitize_ranges(&arr, 100, 200).expect("some");
        assert_eq!(out, vec![(108, 132)]);
    }

    #[test]
    fn sanitize_drops_start_gt_end() {
        let arr = vec![json!([150, 140])];
        assert_eq!(sanitize_ranges(&arr, 100, 200), None);
    }

    #[test]
    fn sanitize_range_fully_outside_chunk_is_none() {
        // Chunk 100..120, range 300..310 → after pad+clamp s=max(298,100)=100... but
        // e=min(312,120)=120, s=298 padded -> 296.. wait: clamp pins s to chunk_start.
        // To truly land outside: range below the chunk, e < chunk_start.
        let arr = vec![json!([10, 20])]; // entirely below chunk_start=100
        assert_eq!(sanitize_ranges(&arr, 100, 200), None);
    }

    #[test]
    fn sanitize_pads_and_clamps_to_bounds() {
        // Range [101,199] padded ±2 → [99,201], clamped to [100,200].
        let arr = vec![json!([101, 199])];
        let out = sanitize_ranges(&arr, 100, 200).expect("some");
        assert_eq!(out, vec![(100, 200)]);
    }

    #[test]
    fn sanitize_all_malformed_is_none() {
        let arr = vec![json!([1, 2, 3]), json!(["a", "b"]), json!(5), json!([9, 4])];
        assert_eq!(sanitize_ranges(&arr, 100, 200), None);
    }

    // ── parse_rerank_response ────────────────────────────────────────────

    fn parse(resp: &str, chunks: &[MergeChunk]) -> RerankOutput {
        // 16 = default rerank_min_prune_lines. XML mode (structured = false).
        parse_rerank_response(resp, chunks, 16, 0, false)
    }

    fn parse_structured(resp: &str, chunks: &[MergeChunk]) -> RerankOutput {
        // 16 = default rerank_min_prune_lines. JSON object-root mode.
        parse_rerank_response(resp, chunks, 16, 0, true)
    }

    #[test]
    fn parse_bare_int_is_whole_chunk() {
        let chunks = vec![chunk(1, 10), chunk(1, 10)];
        let out = parse("<ranked_indices>[1, 0]</ranked_indices>", &chunks);
        assert_eq!(out.reranked_indices, vec![1, 0]);
        assert_eq!(out.line_selections, vec![None, None]);
        assert!(!out.fallback_used);
    }

    #[test]
    fn parse_object_with_lines_yields_ranges() {
        // Chunk 100..200 is large (>30 lines) so pruning applies.
        let chunks = vec![chunk(100, 200)];
        let out = parse(
            "<ranked_indices>[{\"i\":0,\"lines\":[[110,120]]}]</ranked_indices>",
            &chunks,
        );
        assert_eq!(out.reranked_indices, vec![0]);
        // [110,120] padded ±2 → [108,122], within bounds.
        assert_eq!(out.line_selections, vec![Some(vec![(108, 122)])]);
    }

    #[test]
    fn parse_object_without_i_is_skipped() {
        let chunks = vec![chunk(1, 10)];
        let out = parse(
            "<ranked_indices>[{\"lines\":[[1,2]]}, 0]</ranked_indices>",
            &chunks,
        );
        assert_eq!(out.reranked_indices, vec![0]);
        assert_eq!(out.line_selections, vec![None]);
    }

    #[test]
    fn parse_out_of_range_index_dropped() {
        let chunks = vec![chunk(1, 10)];
        let out = parse("<ranked_indices>[5, 0]</ranked_indices>", &chunks);
        assert_eq!(out.reranked_indices, vec![0]);
    }

    #[test]
    fn parse_keep_full_is_whole_chunk() {
        // Canonical "keep whole chunk" form: {"i":idx,"keep":"full"} — no `lines` field.
        let chunks = vec![chunk(100, 200)];
        let out = parse(
            "<ranked_indices>[{\"i\":0,\"keep\":\"full\"}]</ranked_indices>",
            &chunks,
        );
        assert_eq!(out.reranked_indices, vec![0]);
        assert_eq!(out.line_selections, vec![None]);
    }

    #[test]
    fn parse_empty_lines_is_whole_chunk() {
        // Safety net: a stray empty `lines` array still degrades to whole chunk.
        let chunks = vec![chunk(100, 200)];
        let out = parse(
            "<ranked_indices>[{\"i\":0,\"lines\":[]}]</ranked_indices>",
            &chunks,
        );
        assert_eq!(out.reranked_indices, vec![0]);
        assert_eq!(out.line_selections, vec![None]);
    }

    #[test]
    fn parse_small_chunk_never_pruned() {
        // Chunk span 9 < min_prune_lines (16) → selection forced to None.
        let chunks = vec![chunk(1, 10)];
        let out = parse(
            "<ranked_indices>[{\"i\":0,\"lines\":[[3,5]]}]</ranked_indices>",
            &chunks,
        );
        assert_eq!(out.line_selections, vec![None]);
    }

    #[test]
    fn parse_broken_json_falls_back_to_all_indices() {
        let chunks = vec![chunk(1, 10), chunk(1, 10), chunk(1, 10)];
        let out = parse("<ranked_indices>not json</ranked_indices>", &chunks);
        assert!(out.fallback_used);
        assert_eq!(out.reranked_indices, vec![0, 1, 2]);
        assert_eq!(out.line_selections, vec![None, None, None]);
    }

    // ── parse_rerank_response (structured object-root mode) ──────────────

    #[test]
    fn parse_structured_object_root_ranks_chunks() {
        // {"ranked_indices":[...]} object root, reusing the same element forms.
        let chunks = vec![chunk(1, 10), chunk(100, 200)];
        let out = parse_structured(
            "{\"ranked_indices\":[{\"i\":1,\"lines\":[[110,120]]},{\"i\":0,\"keep\":\"full\"}]}",
            &chunks,
        );
        assert!(!out.fallback_used);
        assert_eq!(out.reranked_indices, vec![1, 0]);
        // Chunk 1 (100..200) is large → [110,120] padded ±2 → [108,122].
        // Chunk 0 keep:full → None.
        assert_eq!(out.line_selections, vec![Some(vec![(108, 122)]), None]);
    }

    #[test]
    fn parse_structured_missing_key_falls_back_to_all_indices() {
        // Valid JSON object, but no `ranked_indices` key → original-order fallback.
        let chunks = vec![chunk(1, 10), chunk(1, 10)];
        let out = parse_structured("{\"something_else\":[0,1]}", &chunks);
        assert!(out.fallback_used);
        assert_eq!(out.reranked_indices, vec![0, 1]);
        assert_eq!(out.line_selections, vec![None, None]);
    }

    #[test]
    fn parse_structured_key_not_array_falls_back() {
        // `ranked_indices` present but not an array → original-order fallback.
        let chunks = vec![chunk(1, 10), chunk(1, 10)];
        let out = parse_structured("{\"ranked_indices\":\"oops\"}", &chunks);
        assert!(out.fallback_used);
        assert_eq!(out.reranked_indices, vec![0, 1]);
    }

    #[test]
    fn parse_structured_non_object_root_falls_back() {
        // A bare array (not the expected object root) in structured mode → fallback.
        let chunks = vec![chunk(1, 10), chunk(1, 10)];
        let out = parse_structured("[0, 1]", &chunks);
        assert!(out.fallback_used);
        assert_eq!(out.reranked_indices, vec![0, 1]);
    }

    #[test]
    fn parse_structured_unknown_entry_keys_skipped() {
        // Element-level tolerance is preserved: an object without `i` is skipped,
        // the valid one is kept. (Same loop as XML mode.)
        let chunks = vec![chunk(1, 10), chunk(1, 10)];
        let out = parse_structured(
            "{\"ranked_indices\":[{\"lines\":[[1,2]]},{\"i\":1,\"keep\":\"full\"}]}",
            &chunks,
        );
        assert!(!out.fallback_used);
        assert_eq!(out.reranked_indices, vec![1]);
        assert_eq!(out.line_selections, vec![None]);
    }
}
