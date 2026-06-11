//! Go Gin framework resolver: detects Gin and extracts route handler edges.
//!
//! Detection: `go.mod` contains `github.com/gin-gonic/gin`.
//! Edge extraction: `.GET`/`.POST`/`.PUT`/`.DELETE`/`.Use` method calls → edge
//! to the handler function.

use regex::Regex;
use std::sync::LazyLock;

use crate::indexing::frameworks::{DetectionContext, FrameworkResolver};
use crate::parsing::relations::{EdgeKind, EdgeTarget, RawEdge};
use crate::parsing::symbols::{QualifiedSymbol, Symbol};

pub struct GoGinResolver;

/// Matches Gin route registrations: `r.GET("/path", handlerFunc)` or `group.POST("/path", handler)`
/// Also handles qualified references like `handlers.ListUsers`.
static GIN_ROUTE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"\w+\s*\.\s*(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|Any|Use)\s*\(\s*(?:"[^"]*"\s*,\s*)?([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)"#
    ).unwrap()
});

impl FrameworkResolver for GoGinResolver {
    fn name(&self) -> &str {
        "go-gin"
    }

    fn detect(&self, ctx: &DetectionContext) -> bool {
        for file in ctx.file_set.iter() {
            if file.ends_with("go.mod")
                && let Some(content) = (ctx.read_file)(file)
                && content.contains("github.com/gin-gonic/gin")
            {
                return true;
            }
        }
        false
    }

    fn extract_edges(&self, file_path: &str, source: &str, symbols: &[Symbol]) -> Vec<RawEdge> {
        if !file_path.ends_with(".go") {
            return vec![];
        }

        let mut edges = Vec::new();
        let from_qualified = symbols
            .iter()
            .find(|s| s.qualified.file == file_path)
            .map(|s| s.qualified.clone())
            .unwrap_or_else(|| QualifiedSymbol {
                file: file_path.to_string(),
                scope_path: vec![],
                name: "<module>".to_string(),
            });

        for cap in GIN_ROUTE_RE.captures_iter(source) {
            let handler_name = &cap[1];
            let line = source[..cap.get(0).unwrap().start()]
                .chars()
                .filter(|&c| c == '\n')
                .count() as u32
                + 1;

            // Handle qualified references like "handlers.ListUsers":
            // split on '.' and use last segment as symbol name, prefix as import_path hint.
            let (to_name, import_path) = if let Some(dot_pos) = handler_name.rfind('.') {
                let prefix = &handler_name[..dot_pos];
                let symbol = &handler_name[dot_pos + 1..];
                (symbol.to_string(), Some(prefix.to_string()))
            } else {
                (handler_name.to_string(), None)
            };

            edges.push(RawEdge {
                from: from_qualified.clone(),
                to: EdgeTarget::Unresolved {
                    name: to_name,
                    import_path,
                    qualifier: None,
                },
                kind: EdgeKind::Calls,
                line,
            });
        }

        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn detect_gin_from_go_mod() {
        let mut file_set = HashSet::new();
        file_set.insert("go.mod".to_string());
        let ctx = DetectionContext {
            file_set: &file_set,
            read_file: &|_| {
                Some("module myapp\n\nrequire github.com/gin-gonic/gin v1.9.0\n".to_string())
            },
        };
        assert!(GoGinResolver.detect(&ctx));
    }

    #[test]
    fn no_gin_without_dependency() {
        let mut file_set = HashSet::new();
        file_set.insert("go.mod".to_string());
        let ctx = DetectionContext {
            file_set: &file_set,
            read_file: &|_| {
                Some("module myapp\n\nrequire github.com/gorilla/mux v1.8.0\n".to_string())
            },
        };
        assert!(!GoGinResolver.detect(&ctx));
    }

    #[test]
    fn extract_route_handler_edges() {
        let source = r#"
func SetupRoutes(r *gin.Engine) {
    r.GET("/users", GetUsers)
    r.POST("/users", CreateUser)
    r.Use(AuthMiddleware)

    admin := r.Group("/admin")
    admin.GET("/stats", GetStats)
}
"#;
        let edges = GoGinResolver.extract_edges("routes.go", source, &[]);
        let names: Vec<&str> = edges
            .iter()
            .map(|e| match &e.to {
                EdgeTarget::Unresolved { name, .. } => name.as_str(),
                _ => "",
            })
            .collect();
        assert!(names.contains(&"GetUsers"));
        assert!(names.contains(&"CreateUser"));
        assert!(names.contains(&"AuthMiddleware"));
        assert!(names.contains(&"GetStats"));
    }

    #[test]
    fn skip_non_go_files() {
        let source = r#"r.GET("/path", handler)"#;
        let edges = GoGinResolver.extract_edges("routes.ts", source, &[]);
        assert!(edges.is_empty());
    }

    #[test]
    fn qualified_handler_splits_on_dot() {
        let source = r#"
func SetupRoutes(r *gin.Engine) {
    r.GET("/users", handlers.ListUsers)
    r.POST("/users", handlers.CreateUser)
}
"#;
        let edges = GoGinResolver.extract_edges("routes.go", source, &[]);
        assert_eq!(edges.len(), 2);
        match &edges[0].to {
            EdgeTarget::Unresolved {
                name, import_path, ..
            } => {
                assert_eq!(name, "ListUsers");
                assert_eq!(import_path.as_deref(), Some("handlers"));
            }
            _ => panic!("expected Unresolved"),
        }
        match &edges[1].to {
            EdgeTarget::Unresolved {
                name, import_path, ..
            } => {
                assert_eq!(name, "CreateUser");
                assert_eq!(import_path.as_deref(), Some("handlers"));
            }
            _ => panic!("expected Unresolved"),
        }
    }
}
