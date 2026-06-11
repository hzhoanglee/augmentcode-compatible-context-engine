//! Spring framework resolver: detects Spring Boot and extracts annotation-based edges.
//!
//! Detection: Java files importing from `org.springframework`.
//! Edge extraction: `@GetMapping`/`@PostMapping`/`@Autowired` annotations → routing
//! and dependency injection edges.

use regex::Regex;
use std::sync::LazyLock;

use crate::indexing::frameworks::{DetectionContext, FrameworkResolver};
use crate::parsing::relations::{EdgeKind, EdgeTarget, RawEdge};
use crate::parsing::symbols::{QualifiedSymbol, Symbol, SymbolKind};

pub struct SpringResolver;

/// Matches `@Autowired` followed by a type name (field or constructor injection).
static AUTOWIRED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"@Autowired\s*(?:\n\s*)?(?:private\s+|protected\s+|public\s+)?([A-Z]\w*)\s+\w+")
        .unwrap()
});

/// Matches Spring mapping annotations with optional path.
static MAPPING_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"@(?:Get|Post|Put|Delete|Patch|Request)Mapping\s*(?:\(\s*(?:value\s*=\s*)?(?:"[^"]*"|'[^']*')?\s*\))?"#
    ).unwrap()
});

impl FrameworkResolver for SpringResolver {
    fn name(&self) -> &str {
        "spring"
    }

    fn detect(&self, ctx: &DetectionContext) -> bool {
        for file in ctx.file_set.iter() {
            if file.ends_with(".java")
                && let Some(content) = (ctx.read_file)(file)
                && content.contains("org.springframework")
            {
                return true;
            }
            if (file.ends_with("pom.xml") || file.ends_with("build.gradle"))
                && let Some(content) = (ctx.read_file)(file)
                && (content.contains("spring-boot") || content.contains("springframework"))
            {
                return true;
            }
        }
        false
    }

    fn extract_edges(&self, file_path: &str, source: &str, symbols: &[Symbol]) -> Vec<RawEdge> {
        if !file_path.ends_with(".java") && !file_path.ends_with(".kt") {
            return vec![];
        }

        let mut edges = Vec::new();

        // Find the containing class for edge attribution
        let containing_class = symbols
            .iter()
            .find(|s| s.qualified.file == file_path && s.kind == SymbolKind::Class)
            .map(|s| s.qualified.clone())
            .unwrap_or_else(|| QualifiedSymbol {
                file: file_path.to_string(),
                scope_path: vec![],
                name: "<module>".to_string(),
            });

        // @Autowired DI edges — type name references the injected service
        for cap in AUTOWIRED_RE.captures_iter(source) {
            let type_name = &cap[1];
            let line = source[..cap.get(0).unwrap().start()]
                .chars()
                .filter(|&c| c == '\n')
                .count() as u32
                + 1;

            edges.push(RawEdge {
                from: containing_class.clone(),
                to: EdgeTarget::Unresolved {
                    name: type_name.to_string(),
                    import_path: None,
                    qualifier: None,
                },
                kind: EdgeKind::Uses,
                line,
            });
        }

        // @XxxMapping annotations — mark controller methods as route endpoints
        // We emit edges from the controller to the handler method to establish
        // the routing relationship in the call graph.
        for cap in MAPPING_RE.captures_iter(source) {
            let match_pos = cap.get(0).unwrap().end();
            // Find the method name that follows the annotation
            let remaining = &source[match_pos..];
            if let Some(method_cap) = find_next_method_name(remaining) {
                let line = source[..match_pos].chars().filter(|&c| c == '\n').count() as u32 + 1;

                edges.push(RawEdge {
                    from: containing_class.clone(),
                    to: EdgeTarget::Unresolved {
                        name: method_cap,
                        import_path: None,
                        qualifier: None,
                    },
                    kind: EdgeKind::Calls,
                    line,
                });
            }
        }

        edges
    }
}

/// Find the next method name after a mapping annotation.
fn find_next_method_name(source: &str) -> Option<String> {
    // Look for a method signature pattern: `[modifiers] ReturnType<...> methodName(`
    // The return type may include generics like `List<User>` or `ResponseEntity<List<Foo>>`
    static METHOD_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?:public|private|protected)?\s*[\w<>,\s\[\]?]+\s+([a-z]\w*)\s*\(").unwrap()
    });
    METHOD_NAME_RE
        .captures(source)
        .map(|cap| cap[1].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn detect_spring_from_imports() {
        let mut file_set = HashSet::new();
        file_set.insert("src/main/java/App.java".to_string());
        let ctx = DetectionContext {
            file_set: &file_set,
            read_file: &|_| Some("import org.springframework.boot.SpringApplication;".to_string()),
        };
        assert!(SpringResolver.detect(&ctx));
    }

    #[test]
    fn extract_autowired_edges() {
        let source = r#"
@Controller
public class UserController {
    @Autowired
    private UserService userService;

    @Autowired
    private AuthService authService;
}
"#;
        let symbols = vec![Symbol {
            qualified: QualifiedSymbol {
                file: "UserController.java".to_string(),
                scope_path: vec![],
                name: "UserController".to_string(),
            },
            kind: SymbolKind::Class,
            line_start: 2,
            line_end: 9,
            signature: None,
            parent_fqn: None,
        }];
        let edges = SpringResolver.extract_edges("UserController.java", source, &symbols);
        let names: Vec<&str> = edges
            .iter()
            .map(|e| match &e.to {
                EdgeTarget::Unresolved { name, .. } => name.as_str(),
                _ => "",
            })
            .collect();
        assert!(names.contains(&"UserService"));
        assert!(names.contains(&"AuthService"));
    }

    #[test]
    fn extract_mapping_edges() {
        let source = r#"
@RestController
public class UserController {
    @GetMapping("/users")
    public List<User> getUsers() {
        return userService.findAll();
    }
}
"#;
        let symbols = vec![Symbol {
            qualified: QualifiedSymbol {
                file: "UserController.java".to_string(),
                scope_path: vec![],
                name: "UserController".to_string(),
            },
            kind: SymbolKind::Class,
            line_start: 2,
            line_end: 8,
            signature: None,
            parent_fqn: None,
        }];
        let edges = SpringResolver.extract_edges("UserController.java", source, &symbols);
        let names: Vec<&str> = edges
            .iter()
            .map(|e| match &e.to {
                EdgeTarget::Unresolved { name, .. } => name.as_str(),
                _ => "",
            })
            .collect();
        assert!(names.contains(&"getUsers"));
    }
}
