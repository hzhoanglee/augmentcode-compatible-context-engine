use crate::parsing::symbols::Symbol;

/// A text chunk ready for embedding.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Absolute path of the source file.
    pub file: String,
    pub line_start: u32,
    pub line_end: u32,
    pub content: String,
    /// FQN of the containing symbol, if this chunk came from a symbol body.
    pub symbol_ref: Option<String>,
}

const WINDOW: u32 = 50;
const STRIDE: u32 = 25;

/// Produce chunks for a source file.
///
/// Strategy:
/// 1. **Symbol chunks** — each symbol body becomes one chunk (linked via `symbol_ref`).
/// 2. **Coverage chunks** — sliding window (50 lines, 25-line stride) for lines NOT
///    already fully covered by a symbol chunk. `symbol_ref = None`.
///
/// Lines are 1-indexed.
pub fn chunk_file(file: &str, source: &str, symbols: &[Symbol]) -> Vec<Chunk> {
    let lines: Vec<&str> = source.lines().collect();
    let total_lines = lines.len() as u32;
    if total_lines == 0 {
        return vec![];
    }

    let mut chunks = Vec::new();

    // Build symbol chunks and collect covered line ranges.
    // A "covered" line is one whose range is fully within a symbol chunk.
    let mut symbol_covered: Vec<bool> = vec![false; total_lines as usize];

    for sym in symbols {
        let start = sym.line_start.saturating_sub(1); // 0-indexed
        let end = (sym.line_end).min(total_lines).saturating_sub(1); // 0-indexed inclusive
        if start > end || start >= total_lines {
            continue;
        }
        let content = lines[start as usize..=(end as usize)].join("\n");
        chunks.push(Chunk {
            file: file.to_string(),
            line_start: start + 1,
            line_end: end + 1,
            content,
            symbol_ref: Some(sym.qualified.fqn()),
        });
        for i in start..=end {
            if (i as usize) < symbol_covered.len() {
                symbol_covered[i as usize] = true;
            }
        }
    }

    // Sliding window over uncovered lines.
    let mut window_start: u32 = 0;
    while window_start < total_lines {
        let window_end = (window_start + WINDOW - 1).min(total_lines - 1);

        // Check if this window overlaps any uncovered line.
        let has_uncovered = (window_start..=window_end).any(|i| !symbol_covered[i as usize]);

        if has_uncovered {
            let content = lines[window_start as usize..=window_end as usize].join("\n");
            chunks.push(Chunk {
                file: file.to_string(),
                line_start: window_start + 1,
                line_end: window_end + 1,
                content,
                symbol_ref: None,
            });
        }

        window_start += STRIDE;
    }

    // Guard: never emit a chunk whose content is empty or whitespace-only.
    // This mirrors the Python reference implementation in
    // local-context-engine/corbell/core/embeddings/extractor.py:187.
    // The fix MUST live here (not in embed_all_chunks) because the pipeline
    // relies on a strict 1:1 positional alignment between the chunk list and
    // the returned embedding vectors; filtering at the embed layer would desync
    // that zip and corrupt stored data.
    chunks
        .into_iter()
        .filter(|c| !c.content.trim().is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing::symbols::{QualifiedSymbol, Symbol, SymbolKind};

    fn make_symbol(file: &str, name: &str, line_start: u32, line_end: u32) -> Symbol {
        Symbol {
            qualified: QualifiedSymbol {
                file: file.to_string(),
                scope_path: vec![],
                name: name.to_string(),
            },
            kind: SymbolKind::Function,
            line_start,
            line_end,
            signature: None,
            parent_fqn: None,
        }
    }

    /// A sliding window that lands entirely on blank lines must not produce a
    /// chunk with empty/whitespace content.  Construct a source string where a
    /// real function at the top and a real statement at the bottom are separated
    /// by 60+ blank lines, so at least one 50-line window falls entirely inside
    /// the blank region.
    #[test]
    fn no_blank_content_chunks_from_blank_line_regions() {
        let mut source = String::from("void foo() {\n    return;\n}\n");
        // 60 blank lines — enough to guarantee a full window of blanks
        for _ in 0..60 {
            source.push('\n');
        }
        source.push_str("int x = 1;\n");

        let chunks = chunk_file("test.cpp", &source, &[]);

        for chunk in &chunks {
            assert!(
                !chunk.content.trim().is_empty(),
                "chunk at lines {}-{} has blank/empty content",
                chunk.line_start,
                chunk.line_end,
            );
        }
    }

    /// A completely blank / whitespace-only file must yield an empty chunk vec.
    #[test]
    fn blank_file_yields_no_chunks() {
        let source = "\n   \n\t\n\n";
        let chunks = chunk_file("empty.cpp", source, &[]);
        assert!(
            chunks.is_empty(),
            "expected no chunks for a blank file, got {}",
            chunks.len()
        );
    }

    /// A symbol whose body is entirely whitespace must not produce a symbol chunk.
    #[test]
    fn blank_symbol_body_yields_no_chunk() {
        // Source has a "symbol" on lines 1-3 but the body is all blank lines.
        let source = "\n\n\n";
        let sym = make_symbol("blank_sym.cpp", "blank_fn", 1, 3);
        let chunks = chunk_file("blank_sym.cpp", source, &[sym]);
        assert!(
            chunks.is_empty(),
            "expected no chunks for a symbol with blank body, got {}",
            chunks.len()
        );
    }
}
