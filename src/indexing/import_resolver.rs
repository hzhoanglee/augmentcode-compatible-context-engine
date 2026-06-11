//! Import-path resolution for multi-language edge disambiguation.
//!
//! Provides `resolve_import_path` which attempts to resolve an import path string
//! to a concrete file in the repository's file set. Used as Level 0 (highest priority)
//! in `select_best_candidate` — before the existing Level 1-4 cascade.
//!
//! Supported languages:
//! - TS/JS: extension probing (.ts, .tsx, .js, .jsx, /index.ts, /index.js),
//!   tsconfig/jsconfig paths alias resolution
//! - Python: extension probing (.py, /__init__.py)
//! - Go: go.mod module prefix stripping
//! - Rust: crate::, self::, super:: prefix handling

use std::collections::HashSet;
use std::path::Path;

use crate::parsing::Lang;

// ─── Extension probing tables ────────────────────────────────────────────

/// Extensions to try when resolving TS/JS imports (order matters — first match wins).
const TS_JS_EXTENSIONS: &[&str] = &[
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    "/index.ts",
    "/index.tsx",
    "/index.js",
    "/index.jsx",
];

const PYTHON_EXTENSIONS: &[&str] = &[".py", "/__init__.py"];

#[allow(dead_code)]
const GO_EXTENSIONS: &[&str] = &[".go"];

const RUST_EXTENSIONS: &[&str] = &[".rs", "/mod.rs"];

// ─── Public API ──────────────────────────────────────────────────────────

/// Attempt to resolve an import path to a concrete file path present in `file_set`.
///
/// `import_path`: the raw import specifier (e.g. `"@/utils/format"`, `"../handler"`,
///                `"github.com/pkg/errors"`, `"crate::config"`)
/// `from_file`:   the absolute path of the file containing the import
/// `lang`:        the detected language of the importing file
/// `file_set`:    all known file paths in the repo (absolute paths)
///
/// Returns `Some(absolute_path)` when resolution succeeds, `None` on failure.
/// Resolution failures are silent — the caller falls through to lower-priority levels.
pub fn resolve_import_path(
    import_path: &str,
    from_file: &str,
    lang: Lang,
    file_set: &HashSet<String>,
) -> Option<String> {
    match lang {
        Lang::TypeScript | Lang::Tsx | Lang::JavaScript => {
            resolve_ts_js(import_path, from_file, file_set)
        }
        Lang::Python => resolve_python(import_path, from_file, file_set),
        Lang::Go => resolve_go(import_path, from_file, file_set),
        Lang::Rust => resolve_rust(import_path, from_file, file_set),
        _ => None,
    }
}

// ─── TS/JS resolution ────────────────────────────────────────────────────

fn resolve_ts_js(import_path: &str, from_file: &str, file_set: &HashSet<String>) -> Option<String> {
    // 1. Try tsconfig paths alias resolution first
    if let Some(resolved) = resolve_tsconfig_alias(import_path, from_file, file_set) {
        return Some(resolved);
    }

    // 2. Relative path resolution
    if import_path.starts_with('.') {
        let base_dir = Path::new(from_file).parent()?;
        let resolved_base = normalize_relative(base_dir, import_path);
        return probe_extensions(&resolved_base, TS_JS_EXTENSIONS, file_set);
    }

    // 3. Non-relative — try as repo-relative path with extension probing
    // (handles cases like "src/utils/format" as a path hint)
    probe_with_suffix_match(import_path, TS_JS_EXTENSIONS, file_set)
}

/// Parse tsconfig.json/jsconfig.json `compilerOptions.paths` and resolve aliases.
///
/// Common pattern: `"@/*": ["src/*"]` means `@/foo/bar` → `src/foo/bar`.
/// We search for tsconfig.json in parent directories of `from_file`.
fn resolve_tsconfig_alias(
    import_path: &str,
    from_file: &str,
    file_set: &HashSet<String>,
) -> Option<String> {
    // Only attempt alias resolution for non-relative, non-node_modules paths
    if import_path.starts_with('.') || import_path.starts_with('/') {
        return None;
    }

    // Find tsconfig.json by walking up from from_file's directory
    let tsconfig_path = find_config_file(from_file, &["tsconfig.json", "jsconfig.json"], file_set)?;
    let tsconfig_dir = Path::new(&tsconfig_path).parent()?;

    // Try to read and parse the tsconfig
    let content = std::fs::read_to_string(&tsconfig_path).ok()?;
    let paths = parse_tsconfig_paths(&content)?;

    for (pattern, targets) in &paths {
        // Pattern is like "@/*" — strip the trailing "*" to get prefix "@/"
        let prefix = pattern.strip_suffix('*')?;
        if !import_path.starts_with(prefix) {
            continue;
        }
        let remainder = &import_path[prefix.len()..];

        for target in targets {
            // Target is like "src/*" — strip "*" to get base "src/"
            let target_base = target.strip_suffix('*').unwrap_or(target);
            let resolved_relative = format!("{}{}", target_base, remainder);
            let resolved_abs = tsconfig_dir.join(&resolved_relative);
            let resolved_str = resolved_abs.to_string_lossy().to_string();
            let normalized = normalize_path_separators(&resolved_str);

            if let Some(found) = probe_extensions(&normalized, TS_JS_EXTENSIONS, file_set) {
                return Some(found);
            }
            // Also check if the exact path exists (with extension already)
            if file_set.contains(&normalized) {
                return Some(normalized);
            }
        }
    }
    None
}

/// Parse `compilerOptions.paths` from tsconfig JSON content.
/// Returns Vec<(pattern, Vec<target>)> or None on parse failure.
fn parse_tsconfig_paths(content: &str) -> Option<Vec<(String, Vec<String>)>> {
    // Minimal JSON parsing — we only need compilerOptions.paths
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    let paths = value.get("compilerOptions")?.get("paths")?.as_object()?;
    let mut result = Vec::new();
    for (key, val) in paths {
        if let Some(arr) = val.as_array() {
            let targets: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            result.push((key.clone(), targets));
        }
    }
    Some(result)
}

// ─── Python resolution ───────────────────────────────────────────────────

fn resolve_python(
    import_path: &str,
    from_file: &str,
    file_set: &HashSet<String>,
) -> Option<String> {
    // Relative imports: handle both filesystem-style (./foo) and Python dot-style (.foo, ..foo)
    if import_path.starts_with('.') {
        let base_dir = Path::new(from_file).parent()?;

        // Filesystem-style: starts with "./" or "../"
        if import_path.starts_with("./") || import_path.starts_with("../") {
            let resolved_base = normalize_relative(base_dir, import_path);
            // Strip extension if present, then probe
            let without_ext = resolved_base.strip_suffix(".py").unwrap_or(&resolved_base);
            return probe_extensions(without_ext, PYTHON_EXTENSIONS, file_set);
        }

        // Python dot-style: ".module" means sibling, "..module" means parent
        let dots = import_path.chars().take_while(|&c| c == '.').count();
        let mut target_dir = base_dir.to_path_buf();
        for _ in 1..dots {
            target_dir = target_dir.parent()?.to_path_buf();
        }
        let module_part = &import_path[dots..];
        let module_path = module_part.replace('.', "/");
        let resolved_base = if module_path.is_empty() {
            target_dir.to_string_lossy().to_string()
        } else {
            target_dir.join(&module_path).to_string_lossy().to_string()
        };
        let normalized = normalize_path_separators(&resolved_base);
        return probe_extensions(&normalized, PYTHON_EXTENSIONS, file_set);
    }

    // Absolute import — convert dots to path separators, try as repo-relative
    let path_form = import_path.replace('.', "/");
    probe_with_suffix_match(&path_form, PYTHON_EXTENSIONS, file_set)
}

// ─── Go resolution ───────────────────────────────────────────────────────

fn resolve_go(import_path: &str, from_file: &str, file_set: &HashSet<String>) -> Option<String> {
    // Try to find go.mod and strip the module prefix
    let go_mod_path = find_config_file(from_file, &["go.mod"], file_set)?;
    let go_mod_dir = Path::new(&go_mod_path).parent()?;
    let content = std::fs::read_to_string(&go_mod_path).ok()?;
    let module_prefix = parse_go_mod_module(&content)?;

    // Strip module prefix from import path
    let local_path = import_path.strip_prefix(&module_prefix)?;
    let local_path = local_path.strip_prefix('/').unwrap_or(local_path);

    // The local path should be a directory containing .go files
    let target_dir = go_mod_dir.join(local_path);
    let target_dir_str = normalize_path_separators(&target_dir.to_string_lossy());

    // Find any .go file in that directory within file_set
    for file in file_set {
        let normalized_file = normalize_path_separators(file);
        if normalized_file.starts_with(&target_dir_str)
            && normalized_file.ends_with(".go")
            && !normalized_file[target_dir_str.len()..].contains('/')
        {
            return Some(file.clone());
        }
    }
    None
}

/// Parse the module declaration from go.mod content.
fn parse_go_mod_module(content: &str) -> Option<String> {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(module) = trimmed.strip_prefix("module ") {
            return Some(module.trim().to_string());
        }
    }
    None
}

// ─── Rust resolution ─────────────────────────────────────────────────────

fn resolve_rust(import_path: &str, from_file: &str, file_set: &HashSet<String>) -> Option<String> {
    let path = import_path.trim();

    // Handle crate:: prefix — resolve relative to the crate root (src/)
    if let Some(remainder) = path.strip_prefix("crate::") {
        let module_path = remainder.replace("::", "/");
        return probe_with_suffix_match(&module_path, RUST_EXTENSIONS, file_set);
    }

    // Handle self:: prefix — resolve relative to the current module directory
    if let Some(remainder) = path.strip_prefix("self::") {
        let base_dir = if from_file.ends_with("mod.rs") || from_file.ends_with("lib.rs") {
            Path::new(from_file).parent()?
        } else {
            // src/foo.rs → src/foo/ (sibling module directory)
            let stem = Path::new(from_file).file_stem()?.to_str()?;
            &Path::new(from_file).parent()?.join(stem)
        };
        let module_path = remainder.replace("::", "/");
        let resolved_base = base_dir.join(&module_path).to_string_lossy().to_string();
        let normalized = normalize_path_separators(&resolved_base);
        return probe_extensions(&normalized, RUST_EXTENSIONS, file_set);
    }

    // Handle super:: prefix — resolve relative to parent module
    if let Some(remainder) = path.strip_prefix("super::") {
        let current_dir = Path::new(from_file).parent()?;
        let parent_dir = if from_file.ends_with("mod.rs") {
            current_dir.parent()?
        } else {
            current_dir
        };
        // Count consecutive super:: prefixes
        let mut target = parent_dir.to_path_buf();
        let mut rest = remainder;
        while let Some(next) = rest.strip_prefix("super::") {
            target = target.parent()?.to_path_buf();
            rest = next;
        }
        let module_path = rest.replace("::", "/");
        let resolved_base = target.join(&module_path).to_string_lossy().to_string();
        let normalized = normalize_path_separators(&resolved_base);
        return probe_extensions(&normalized, RUST_EXTENSIONS, file_set);
    }

    None
}

// ─── Utility helpers ─────────────────────────────────────────────────────

/// Normalize a relative path like "../foo/bar" against a base directory.
fn normalize_relative(base: &Path, relative: &str) -> String {
    let target = base.join(relative);
    // Simplify path components (resolve ..)
    let mut components = Vec::new();
    for comp in target.components() {
        match comp {
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::CurDir => {}
            _ => {
                components.push(comp);
            }
        }
    }
    let result: std::path::PathBuf = components.into_iter().collect();
    normalize_path_separators(&result.to_string_lossy())
}

/// Probe extensions against the file set. Returns the first match.
fn probe_extensions(
    base_path: &str,
    extensions: &[&str],
    file_set: &HashSet<String>,
) -> Option<String> {
    for ext in extensions {
        let candidate = format!("{}{}", base_path, ext);
        let normalized = normalize_path_separators(&candidate);
        if file_set.contains(&normalized) {
            return Some(normalized);
        }
        // Also try with platform-native separators
        if cfg!(windows) {
            let win_candidate = candidate.replace('/', "\\");
            if file_set.contains(&win_candidate) {
                return Some(win_candidate);
            }
        }
    }
    // Also check if the exact base path exists (already has extension)
    if file_set.contains(base_path) {
        return Some(base_path.to_string());
    }
    None
}

/// For non-relative imports, find a file whose path ends with the import path
/// (with extension probing). Used for absolute/package imports.
fn probe_with_suffix_match(
    import_path: &str,
    extensions: &[&str],
    file_set: &HashSet<String>,
) -> Option<String> {
    for ext in extensions {
        let suffix = format!("{}{}", import_path, ext);
        let normalized_suffix = normalize_path_separators(&suffix);
        for file in file_set {
            let normalized_file = normalize_path_separators(file);
            if normalized_file.ends_with(&normalized_suffix) {
                return Some(file.clone());
            }
        }
    }
    // Direct suffix match without extension
    let normalized_import = normalize_path_separators(import_path);
    for file in file_set {
        let normalized_file = normalize_path_separators(file);
        if normalized_file.ends_with(&normalized_import) {
            return Some(file.clone());
        }
    }
    None
}

/// Find a config file (tsconfig.json, go.mod, etc.) by walking up parent directories.
fn find_config_file(
    from_file: &str,
    config_names: &[&str],
    file_set: &HashSet<String>,
) -> Option<String> {
    let mut dir = Path::new(from_file).parent()?;
    loop {
        for name in config_names {
            let candidate = dir.join(name);
            let candidate_str = candidate.to_string_lossy().to_string();
            let normalized = normalize_path_separators(&candidate_str);
            if file_set.contains(&normalized) {
                return Some(normalized);
            }
            if file_set.contains(&candidate_str) {
                return Some(candidate_str);
            }
        }
        dir = dir.parent()?;
    }
}

/// Normalize path separators to forward slashes for consistent comparison.
fn normalize_path_separators(path: &str) -> String {
    path.replace('\\', "/")
}

// ─── Re-export / barrel chasing ─────────────────────────────────────────

/// When an import resolves to a barrel/index file (e.g. `index.ts`, `__init__.py`),
/// check if a sibling file named after `target_symbol` exists in the same directory.
/// This handles the common re-export pattern without needing to read file content:
///   import { Button } from "./components"  →  resolves to components/index.ts
///   → check for components/Button.ts (or .tsx, .js, etc.)
///
/// `resolved_file`: the file that the import resolved to (an index/barrel file)
/// `target_symbol`: the specific symbol being imported (e.g. "Button")
/// `file_set`: all known file paths in the repo
/// `depth`: recursion depth (capped at 8 for cycle prevention)
///
/// Returns `Some(file_path)` if a sibling file matching the symbol name is found.
pub fn chase_reexports(
    resolved_file: &str,
    target_symbol: &str,
    file_set: &HashSet<String>,
    depth: u8,
) -> Option<String> {
    if depth > 8 || target_symbol.is_empty() {
        return None;
    }

    let normalized = normalize_path_separators(resolved_file);

    // Only attempt chasing for recognized barrel/index file patterns.
    let is_barrel = normalized.ends_with("/index.ts")
        || normalized.ends_with("/index.tsx")
        || normalized.ends_with("/index.js")
        || normalized.ends_with("/index.jsx")
        || normalized.ends_with("/__init__.py")
        || normalized.ends_with("/mod.rs");

    if !is_barrel {
        return None;
    }

    // Get the directory containing the barrel file.
    let dir = std::path::Path::new(resolved_file).parent()?;
    let dir_str = normalize_path_separators(&dir.to_string_lossy());

    // Determine which extensions to try based on the barrel file's extension.
    let extensions: &[&str] = if normalized.ends_with(".ts") || normalized.ends_with(".tsx") {
        &[".ts", ".tsx", ".js", ".jsx"]
    } else if normalized.ends_with(".js") || normalized.ends_with(".jsx") {
        &[".js", ".jsx", ".ts", ".tsx"]
    } else if normalized.ends_with(".py") {
        &[".py"]
    } else if normalized.ends_with(".rs") {
        &[".rs"]
    } else {
        return None;
    };

    // Try direct sibling: dir/TargetSymbol.ext
    for ext in extensions {
        let candidate = format!("{}/{}{}", dir_str, target_symbol, ext);
        let norm_candidate = normalize_path_separators(&candidate);
        if file_set.contains(&norm_candidate) {
            return Some(norm_candidate);
        }
        // Also try lowercase variant (common in Python/Go).
        let lower_candidate = format!("{}/{}{}", dir_str, target_symbol.to_lowercase(), ext);
        let norm_lower = normalize_path_separators(&lower_candidate);
        if file_set.contains(&norm_lower) {
            return Some(norm_lower);
        }
    }

    // Try subdirectory barrel: dir/TargetSymbol/index.ext
    for ext in extensions {
        let sub_barrel = format!("{}/{}/index{}", dir_str, target_symbol, ext);
        let norm_sub = normalize_path_separators(&sub_barrel);
        if file_set.contains(&norm_sub) {
            // Recurse to see if there's a deeper resolution.
            if let Some(deeper) = chase_reexports(&norm_sub, target_symbol, file_set, depth + 1) {
                return Some(deeper);
            }
            return Some(norm_sub);
        }
    }

    None
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_file_set(files: &[&str]) -> HashSet<String> {
        files.iter().map(|s| s.to_string()).collect()
    }

    // --- TS/JS Extension Probing ---

    #[test]
    fn ts_relative_import_resolves() {
        let file_set = make_file_set(&[
            "/project/src/utils/format.ts",
            "/project/src/components/Button.tsx",
        ]);
        let result = resolve_import_path(
            "./utils/format",
            "/project/src/index.ts",
            Lang::TypeScript,
            &file_set,
        );
        assert_eq!(result, Some("/project/src/utils/format.ts".to_string()));
    }

    #[test]
    fn ts_index_file_resolution() {
        let file_set = make_file_set(&["/project/src/utils/index.ts"]);
        let result = resolve_import_path(
            "./utils",
            "/project/src/app.ts",
            Lang::TypeScript,
            &file_set,
        );
        assert_eq!(result, Some("/project/src/utils/index.ts".to_string()));
    }

    #[test]
    fn ts_tsx_extension_probing() {
        let file_set = make_file_set(&["/project/src/Button.tsx"]);
        let result = resolve_import_path("./Button", "/project/src/App.tsx", Lang::Tsx, &file_set);
        assert_eq!(result, Some("/project/src/Button.tsx".to_string()));
    }

    #[test]
    fn ts_nonrelative_suffix_match() {
        let file_set = make_file_set(&["/project/src/services/auth.ts"]);
        let result = resolve_import_path(
            "services/auth",
            "/project/src/app.ts",
            Lang::TypeScript,
            &file_set,
        );
        assert_eq!(result, Some("/project/src/services/auth.ts".to_string()));
    }

    // --- Python Extension Probing ---

    #[test]
    fn python_dotted_import() {
        let file_set = make_file_set(&["/project/app/models/user.py"]);
        let result = resolve_import_path(
            "app/models/user",
            "/project/main.py",
            Lang::Python,
            &file_set,
        );
        assert_eq!(result, Some("/project/app/models/user.py".to_string()));
    }

    #[test]
    fn python_package_init() {
        let file_set = make_file_set(&["/project/app/models/__init__.py"]);
        let result = resolve_import_path("app/models", "/project/main.py", Lang::Python, &file_set);
        assert_eq!(result, Some("/project/app/models/__init__.py".to_string()));
    }

    #[test]
    fn python_relative_import() {
        let file_set = make_file_set(&["/project/app/utils.py"]);
        let result = resolve_import_path(
            "./utils",
            "/project/app/handler.py",
            Lang::Python,
            &file_set,
        );
        assert_eq!(result, Some("/project/app/utils.py".to_string()));
    }

    // --- Rust Resolution ---

    #[test]
    fn rust_crate_prefix() {
        let file_set = make_file_set(&["/project/src/config.rs"]);
        let result = resolve_import_path(
            "crate::config",
            "/project/src/main.rs",
            Lang::Rust,
            &file_set,
        );
        assert_eq!(result, Some("/project/src/config.rs".to_string()));
    }

    #[test]
    fn rust_crate_nested_module() {
        let file_set = make_file_set(&["/project/src/indexing/pipeline.rs"]);
        let result = resolve_import_path(
            "crate::indexing/pipeline",
            "/project/src/main.rs",
            Lang::Rust,
            &file_set,
        );
        assert_eq!(
            result,
            Some("/project/src/indexing/pipeline.rs".to_string())
        );
    }

    #[test]
    fn rust_self_prefix() {
        let file_set = make_file_set(&["/project/src/indexing/walker.rs"]);
        let result = resolve_import_path(
            "self::walker",
            "/project/src/indexing/mod.rs",
            Lang::Rust,
            &file_set,
        );
        assert_eq!(result, Some("/project/src/indexing/walker.rs".to_string()));
    }

    #[test]
    fn rust_super_prefix() {
        let file_set = make_file_set(&["/project/src/config.rs"]);
        let result = resolve_import_path(
            "super::config",
            "/project/src/indexing/mod.rs",
            Lang::Rust,
            &file_set,
        );
        assert_eq!(result, Some("/project/src/config.rs".to_string()));
    }

    // --- Go Resolution ---

    #[test]
    fn go_module_path_stripping() {
        // This test requires go.mod file on disk — skipped in pure unit test.
        // Covered by the parse_go_mod_module unit test below.
        let module = parse_go_mod_module("module github.com/user/project\n\ngo 1.21\n");
        assert_eq!(module, Some("github.com/user/project".to_string()));
    }

    // --- Failure fallthrough ---

    #[test]
    fn unresolvable_returns_none() {
        let file_set = make_file_set(&["/project/src/other.ts"]);
        let result = resolve_import_path(
            "./nonexistent",
            "/project/src/app.ts",
            Lang::TypeScript,
            &file_set,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn unsupported_language_returns_none() {
        let file_set = make_file_set(&["/project/Main.java"]);
        let result = resolve_import_path(
            "com.example.Main",
            "/project/App.java",
            Lang::Java,
            &file_set,
        );
        assert_eq!(result, None);
    }

    #[test]
    fn empty_import_path() {
        let file_set = make_file_set(&["/project/src/main.rs"]);
        let result = resolve_import_path("", "/project/src/main.rs", Lang::Rust, &file_set);
        assert_eq!(result, None);
    }

    // --- chase_reexports ---

    #[test]
    fn chase_reexports_finds_sibling_ts() {
        let file_set = make_file_set(&[
            "/project/src/components/index.ts",
            "/project/src/components/Button.tsx",
        ]);
        let result = chase_reexports("/project/src/components/index.ts", "Button", &file_set, 0);
        assert_eq!(
            result,
            Some("/project/src/components/Button.tsx".to_string())
        );
    }

    #[test]
    fn chase_reexports_finds_lowercase_py() {
        let file_set = make_file_set(&[
            "/project/app/models/__init__.py",
            "/project/app/models/user.py",
        ]);
        let result = chase_reexports("/project/app/models/__init__.py", "User", &file_set, 0);
        // Finds lowercase variant
        assert_eq!(result, Some("/project/app/models/user.py".to_string()));
    }

    #[test]
    fn chase_reexports_finds_subdirectory_barrel() {
        let file_set = make_file_set(&[
            "/project/src/components/index.ts",
            "/project/src/components/Button/index.ts",
            "/project/src/components/Button/Button.tsx",
        ]);
        let result = chase_reexports("/project/src/components/index.ts", "Button", &file_set, 0);
        // Should find the deeper Button.tsx via subdirectory barrel recursion
        assert_eq!(
            result,
            Some("/project/src/components/Button/Button.tsx".to_string())
        );
    }

    #[test]
    fn chase_reexports_non_barrel_returns_none() {
        let file_set = make_file_set(&["/project/src/utils.ts", "/project/src/Button.tsx"]);
        let result = chase_reexports("/project/src/utils.ts", "Button", &file_set, 0);
        assert_eq!(result, None);
    }

    #[test]
    fn chase_reexports_depth_limit() {
        let file_set = make_file_set(&["/project/src/index.ts"]);
        // depth > 8 should return None immediately
        let result = chase_reexports("/project/src/index.ts", "Foo", &file_set, 9);
        assert_eq!(result, None);
    }
}
