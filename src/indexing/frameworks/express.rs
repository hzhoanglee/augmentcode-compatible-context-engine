//! Express framework resolver: detects Express.js and extracts route handler edges.
//!
//! Detection: `package.json` contains "express" in dependencies.
//! Edge extraction: `router.get/post/put/delete/use(path, handler)` patterns → edge
//! from route registration to handler function.

use regex::Regex;
use std::sync::LazyLock;

use crate::indexing::frameworks::{DetectionContext, FrameworkResolver};
use crate::parsing::relations::{EdgeKind, EdgeTarget, RawEdge};
use crate::parsing::symbols::{QualifiedSymbol, Symbol};

pub struct ExpressResolver;

/// Matches Express route patterns: `app.get('/path', handler)` or `router.use(middleware)`
/// Captures the handler name (skips inline arrow functions/anonymous functions).
static ROUTE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?:app|router|server)\s*\.\s*(?:get|post|put|delete|patch|use|all)\s*\(\s*(?:'[^']*'|`[^`]*`|"[^"]*")\s*,\s*([a-zA-Z_]\w*)"#
    ).unwrap()
});

/// Also match direct handler references without path:
/// `app.use(authMiddleware)`, `router.use(cors())`
static USE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:app|router|server)\s*\.\s*use\s*\(\s*([a-zA-Z_]\w*)\s*[,)]").unwrap()
});

impl FrameworkResolver for ExpressResolver {
    fn name(&self) -> &str {
        "express"
    }

    fn detect(&self, ctx: &DetectionContext) -> bool {
        for file in ctx.file_set.iter() {
            if file.ends_with("package.json")
                && !file.contains("node_modules")
                && let Some(content) = (ctx.read_file)(file)
                && (content.contains("\"express\"") || content.contains("'express'"))
            {
                return true;
            }
        }
        false
    }

    fn extract_edges(&self, file_path: &str, source: &str, symbols: &[Symbol]) -> Vec<RawEdge> {
        if !file_path.ends_with(".js")
            && !file_path.ends_with(".ts")
            && !file_path.ends_with(".mjs")
            && !file_path.ends_with(".cjs")
        {
            return vec![];
        }

        let mut edges = Vec::new();
        let from_qualified = find_module_symbol(symbols, file_path);

        // Route handlers with path
        for cap in ROUTE_RE.captures_iter(source) {
            let handler_name = &cap[1];
            let line = source[..cap.get(0).unwrap().start()]
                .chars()
                .filter(|&c| c == '\n')
                .count() as u32
                + 1;

            edges.push(RawEdge {
                from: from_qualified.clone(),
                to: EdgeTarget::Unresolved {
                    name: handler_name.to_string(),
                    import_path: None,
                    qualifier: None,
                },
                kind: EdgeKind::Calls,
                line,
            });
        }

        // Middleware usage without path
        for cap in USE_RE.captures_iter(source) {
            let handler_name = &cap[1];
            // Skip if already captured by ROUTE_RE
            let line = source[..cap.get(0).unwrap().start()]
                .chars()
                .filter(|&c| c == '\n')
                .count() as u32
                + 1;

            edges.push(RawEdge {
                from: from_qualified.clone(),
                to: EdgeTarget::Unresolved {
                    name: handler_name.to_string(),
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

fn find_module_symbol(symbols: &[Symbol], file_path: &str) -> QualifiedSymbol {
    symbols
        .iter()
        .find(|s| s.qualified.file == file_path)
        .map(|s| s.qualified.clone())
        .unwrap_or_else(|| QualifiedSymbol {
            file: file_path.to_string(),
            scope_path: vec![],
            name: "<module>".to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn detect_express_from_package_json() {
        let mut file_set = HashSet::new();
        file_set.insert("package.json".to_string());
        let ctx = DetectionContext {
            file_set: &file_set,
            read_file: &|_| Some(r#"{"dependencies": {"express": "^4.18.0"}}"#.to_string()),
        };
        assert!(ExpressResolver.detect(&ctx));
    }

    #[test]
    fn extract_route_handler_edges() {
        let source = r#"
const express = require('express');
const router = express.Router();
router.get('/users', getUsers);
router.post('/users', createUser);
app.use(authMiddleware);
"#;
        let edges = ExpressResolver.extract_edges("routes/users.js", source, &[]);
        let names: Vec<&str> = edges
            .iter()
            .map(|e| match &e.to {
                EdgeTarget::Unresolved { name, .. } => name.as_str(),
                _ => "",
            })
            .collect();
        assert!(names.contains(&"getUsers"));
        assert!(names.contains(&"createUser"));
        assert!(names.contains(&"authMiddleware"));
    }

    #[test]
    fn skip_inline_arrow_functions() {
        // Inline functions like `router.get('/path', (req, res) => {})` should not produce edges
        let source = r#"router.get('/path', (req, res) => { res.send('ok'); });"#;
        let edges = ExpressResolver.extract_edges("routes/api.js", source, &[]);
        // The regex won't match `(req` as a valid identifier start
        assert!(edges.is_empty());
    }
}
