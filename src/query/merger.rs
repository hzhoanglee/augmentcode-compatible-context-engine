use std::collections::HashMap;

/// A code chunk that participates in the merge pipeline.
#[derive(Debug, Clone)]
pub struct MergeChunk {
    pub file: String,
    pub line_start: u32,
    pub line_end: u32,
    pub score: f32,
    /// Pre-fetched content (may be empty string; will be re-read after merge).
    pub content: String,
    pub symbol: Option<String>,
    /// Full qualified name from the symbol table (e.g. "src/foo.rs::Mod::func").
    pub symbol_fqn: Option<String>,
    /// Symbol kind (e.g. "function", "class", "struct") — populated from the
    /// symbol table during fetch_chunk_content. Used for kind: filter application.
    pub symbol_kind: Option<String>,
}

/// Dedup + merge adjacent chunks, then cap to `top_k`.
///
/// Steps (as specified):
/// 1. Dedup by (file, line_start, line_end) — keep highest score.
/// 2. Group by file.
/// 3. Sort within each group by line_start ASC.
/// 4. Merge pass: merge when next.line_start <= current.line_end + 2 (gap ≤ 1),
///    capping merged range at 60 lines.
/// 5. Drop contained: within each file group, sort by width DESC, skip any chunk
///    fully inside a wider one.
/// 6. Collect all groups, sort by score DESC, cap to top_k.
pub fn merge_chunks(chunks: Vec<MergeChunk>, top_k: usize) -> Vec<MergeChunk> {
    if chunks.is_empty() {
        return vec![];
    }

    // Step 1: dedup — (file, line_start, line_end) → best score entry.
    let mut dedup_map: HashMap<(String, u32, u32), MergeChunk> = HashMap::new();
    for chunk in chunks {
        let key = (chunk.file.clone(), chunk.line_start, chunk.line_end);
        dedup_map
            .entry(key)
            .and_modify(|existing| {
                if chunk.score > existing.score {
                    *existing = chunk.clone();
                }
            })
            .or_insert(chunk);
    }

    // Step 2: group by file.
    let mut by_file: HashMap<String, Vec<MergeChunk>> = HashMap::new();
    for chunk in dedup_map.into_values() {
        by_file.entry(chunk.file.clone()).or_default().push(chunk);
    }

    let mut result = Vec::new();

    for (_file, mut file_chunks) in by_file {
        // Step 3: sort by line_start ASC.
        file_chunks.sort_unstable_by_key(|c| c.line_start);

        // Step 4: merge pass.
        let mut merged: Vec<MergeChunk> = Vec::new();
        for next in file_chunks {
            if let Some(current) = merged.last_mut() {
                // Merge condition: next starts within gap of 1 line from current end.
                if next.line_start <= current.line_end + 2 {
                    let new_end = current.line_end.max(next.line_end);
                    let new_start = current.line_start;
                    // Cap at 60 lines.
                    if new_end - new_start + 1 > 60 {
                        merged.push(next);
                    } else {
                        // Compute overlap BEFORE updating line_end.
                        let old_end = current.line_end;
                        let overlap_lines = if next.line_start <= old_end {
                            (old_end - next.line_start + 1) as usize
                        } else {
                            0
                        };

                        current.line_end = new_end;
                        current.score = current.score.max(next.score);

                        // Append only the non-overlapping tail of next's content as fallback.
                        if !next.content.is_empty() {
                            let tail: String = next
                                .content
                                .lines()
                                .skip(overlap_lines)
                                .collect::<Vec<_>>()
                                .join("\n");
                            if !tail.is_empty() {
                                current.content.push('\n');
                                current.content.push_str(&tail);
                            }
                        }
                        if current.symbol.is_none() && next.symbol.is_some() {
                            current.symbol = next.symbol;
                        }
                        if current.symbol_fqn.is_none() && next.symbol_fqn.is_some() {
                            current.symbol_fqn = next.symbol_fqn;
                        }
                        if current.symbol_kind.is_none() && next.symbol_kind.is_some() {
                            current.symbol_kind = next.symbol_kind;
                        }
                    }
                } else {
                    merged.push(next);
                }
            } else {
                merged.push(next);
            }
        }

        // Step 5: drop contained chunks.
        // Sort by width DESC (wider chunks survive).
        merged.sort_unstable_by(|a, b| {
            let wa = a.line_end.saturating_sub(a.line_start);
            let wb = b.line_end.saturating_sub(b.line_start);
            wb.cmp(&wa)
        });
        let mut survivors: Vec<MergeChunk> = Vec::new();
        for candidate in merged {
            let contained = survivors.iter().any(|s| {
                s.file == candidate.file
                    && s.line_start <= candidate.line_start
                    && s.line_end >= candidate.line_end
            });
            if !contained {
                survivors.push(candidate);
            }
        }

        result.extend(survivors);
    }

    // Step 6: sort by score DESC and cap to top_k.
    result.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result.truncate(top_k);
    result
}
