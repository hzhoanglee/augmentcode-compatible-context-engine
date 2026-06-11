//! Django framework resolver: detects Django and extracts URL-to-view edges.
//!
//! Detection: Python files importing from `django`.
//! Edge extraction: `path()` / `re_path()` calls in urls.py → edge to view function.

use regex::Regex;
use std::sync::LazyLock;

use crate::indexing::frameworks::{DetectionContext, FrameworkResolver};
use crate::parsing::relations::{EdgeKind, EdgeTarget, RawEdge};
use crate::parsing::symbols::{QualifiedSymbol, Symbol};

pub struct DjangoResolver;

/// Matches `path('route', views.handler_name)` or `path('route', handler_name)`
static PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?:path|re_path)\s*\(\s*(?:'[^']*'|r'[^']*'|"[^"]*")\s*,\s*(?:views\.)?([a-zA-Z_]\w*)"#,
    )
    .unwrap()
});

/// Matches class-based views: `path('route', MyView.as_view())`
static CBV_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?:path|re_path)\s*\(\s*(?:'[^']*'|r'[^']*'|"[^"]*")\s*,\s*([A-Z]\w*)\.as_view\(\)"#,
    )
    .unwrap()
});

impl FrameworkResolver for DjangoResolver {
    fn name(&self) -> &str {
        "django"
    }

    fn detect(&self, ctx: &DetectionContext) -> bool {
        for file in ctx.file_set.iter() {
            // Look for Django imports in Python files
            if file.ends_with(".py")
                && let Some(content) = (ctx.read_file)(file)
                && (content.contains("from django") || content.contains("import django"))
            {
                return true;
            }
            // Also detect via settings.py or manage.py
            if (file.ends_with("manage.py") || file.ends_with("settings.py"))
                && let Some(content) = (ctx.read_file)(file)
                && content.contains("django")
            {
                return true;
            }
        }
        false
    }

    fn extract_edges(&self, file_path: &str, source: &str, symbols: &[Symbol]) -> Vec<RawEdge> {
        // Only process urls.py files
        if !file_path.ends_with("urls.py") {
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

        // Function-based views
        for cap in PATH_RE.captures_iter(source) {
            let view_name = &cap[1];
            let line = source[..cap.get(0).unwrap().start()]
                .chars()
                .filter(|&c| c == '\n')
                .count() as u32
                + 1;

            edges.push(RawEdge {
                from: from_qualified.clone(),
                to: EdgeTarget::Unresolved {
                    name: view_name.to_string(),
                    import_path: None,
                    qualifier: None,
                },
                kind: EdgeKind::Calls,
                line,
            });
        }

        // Class-based views
        for cap in CBV_RE.captures_iter(source) {
            let view_class = &cap[1];
            let line = source[..cap.get(0).unwrap().start()]
                .chars()
                .filter(|&c| c == '\n')
                .count() as u32
                + 1;

            edges.push(RawEdge {
                from: from_qualified.clone(),
                to: EdgeTarget::Unresolved {
                    name: view_class.to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn detect_django_from_imports() {
        let mut file_set = HashSet::new();
        file_set.insert("app/views.py".to_string());
        let ctx = DetectionContext {
            file_set: &file_set,
            read_file: &|_| Some("from django.http import HttpResponse".to_string()),
        };
        assert!(DjangoResolver.detect(&ctx));
    }

    #[test]
    fn extract_url_view_edges() {
        let source = r#"
from django.urls import path
from . import views

urlpatterns = [
    path('users/', views.user_list),
    path('users/<int:pk>/', views.user_detail),
    path('admin/', AdminView.as_view()),
]
"#;
        let edges = DjangoResolver.extract_edges("app/urls.py", source, &[]);
        let names: Vec<&str> = edges
            .iter()
            .map(|e| match &e.to {
                EdgeTarget::Unresolved { name, .. } => name.as_str(),
                _ => "",
            })
            .collect();
        assert!(names.contains(&"user_list"));
        assert!(names.contains(&"user_detail"));
        assert!(names.contains(&"AdminView"));
    }

    #[test]
    fn skip_non_urls_files() {
        let source = "path('foo/', handler)";
        let edges = DjangoResolver.extract_edges("app/views.py", source, &[]);
        assert!(edges.is_empty());
    }
}
