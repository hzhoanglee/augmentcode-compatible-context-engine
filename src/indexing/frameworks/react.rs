//! React framework resolver: detects React and extracts component-rendering edges.
//!
//! Detection: `package.json` contains "react" in dependencies.
//! Edge extraction: Capitalized JSX tag names in .tsx/.jsx files → Calls edge
//! from containing function to the component name.

use regex::Regex;
use std::sync::LazyLock;

use crate::indexing::frameworks::{DetectionContext, FrameworkResolver};
use crate::parsing::relations::{EdgeKind, EdgeTarget, RawEdge};
use crate::parsing::symbols::{QualifiedSymbol, Symbol};

pub struct ReactResolver;

/// Matches capitalized JSX tags like `<UserProfile` or `<App.Header`
static JSX_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<([A-Z][A-Za-z0-9]*(?:\.[A-Z][A-Za-z0-9]*)*)[\s/>]").unwrap());

impl FrameworkResolver for ReactResolver {
    fn name(&self) -> &str {
        "react"
    }

    fn detect(&self, ctx: &DetectionContext) -> bool {
        // Look for package.json with react dependency
        for file in ctx.file_set.iter() {
            if file.ends_with("package.json")
                && !file.contains("node_modules")
                && let Some(content) = (ctx.read_file)(file)
                && (content.contains("\"react\"") || content.contains("'react'"))
            {
                return true;
            }
        }
        false
    }

    fn extract_edges(&self, file_path: &str, source: &str, symbols: &[Symbol]) -> Vec<RawEdge> {
        // Only process JSX/TSX files
        if !file_path.ends_with(".tsx")
            && !file_path.ends_with(".jsx")
            && !file_path.ends_with(".js")
            && !file_path.ends_with(".ts")
        {
            return vec![];
        }

        let mut edges = Vec::new();

        // Find the enclosing function/component for context
        let from_symbol = find_enclosing_component(symbols, file_path);

        for cap in JSX_TAG_RE.captures_iter(source) {
            let tag_name = &cap[1];
            // Skip HTML-native tags (all lowercase won't match our regex anyway)
            // Skip self-references
            if let Some(ref from) = from_symbol
                && from.qualified.name == tag_name
            {
                continue;
            }

            let line = source[..cap.get(0).unwrap().start()]
                .chars()
                .filter(|&c| c == '\n')
                .count() as u32
                + 1;

            let from_qualified = from_symbol
                .as_ref()
                .map(|s| s.qualified.clone())
                .unwrap_or_else(|| QualifiedSymbol {
                    file: file_path.to_string(),
                    scope_path: vec![],
                    name: "<module>".to_string(),
                });

            edges.push(RawEdge {
                from: from_qualified,
                to: EdgeTarget::Unresolved {
                    name: tag_name.to_string(),
                    import_path: None,
                    qualifier: None,
                },
                kind: EdgeKind::Calls,
                line,
            });
        }

        edges
    }
}

/// Find the first function/component symbol in the file (best-effort enclosing context).
fn find_enclosing_component(symbols: &[Symbol], file_path: &str) -> Option<Symbol> {
    symbols
        .iter()
        .find(|s| {
            s.qualified.file == file_path
                && matches!(
                    s.kind,
                    crate::parsing::symbols::SymbolKind::Function
                        | crate::parsing::symbols::SymbolKind::Method
                )
        })
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn detect_react_from_package_json() {
        let mut file_set = HashSet::new();
        file_set.insert("package.json".to_string());
        let ctx = DetectionContext {
            file_set: &file_set,
            read_file: &|_| Some(r#"{"dependencies": {"react": "^18.0.0"}}"#.to_string()),
        };
        assert!(ReactResolver.detect(&ctx));
    }

    #[test]
    fn no_react_without_dependency() {
        let mut file_set = HashSet::new();
        file_set.insert("package.json".to_string());
        let ctx = DetectionContext {
            file_set: &file_set,
            read_file: &|_| Some(r#"{"dependencies": {"express": "^4.0.0"}}"#.to_string()),
        };
        assert!(!ReactResolver.detect(&ctx));
    }

    #[test]
    fn extract_jsx_component_edges() {
        let source = r#"
function App() {
    return (
        <div>
            <UserProfile name="test" />
            <Header />
            <span>text</span>
        </div>
    );
}
"#;
        let symbols = vec![Symbol {
            qualified: QualifiedSymbol {
                file: "src/App.tsx".to_string(),
                scope_path: vec![],
                name: "App".to_string(),
            },
            kind: crate::parsing::symbols::SymbolKind::Function,
            line_start: 2,
            line_end: 10,
            signature: None,
            parent_fqn: None,
        }];

        let edges = ReactResolver.extract_edges("src/App.tsx", source, &symbols);
        let names: Vec<&str> = edges
            .iter()
            .map(|e| match &e.to {
                EdgeTarget::Unresolved { name, .. } => name.as_str(),
                _ => "",
            })
            .collect();
        assert!(names.contains(&"UserProfile"));
        assert!(names.contains(&"Header"));
        // Should not contain lowercase tags
        assert!(!names.contains(&"div"));
        assert!(!names.contains(&"span"));
    }

    #[test]
    fn skip_self_reference() {
        let source = "<App />";
        let symbols = vec![Symbol {
            qualified: QualifiedSymbol {
                file: "src/App.tsx".to_string(),
                scope_path: vec![],
                name: "App".to_string(),
            },
            kind: crate::parsing::symbols::SymbolKind::Function,
            line_start: 1,
            line_end: 1,
            signature: None,
            parent_fqn: None,
        }];
        let edges = ReactResolver.extract_edges("src/App.tsx", source, &symbols);
        assert!(edges.is_empty());
    }
}
