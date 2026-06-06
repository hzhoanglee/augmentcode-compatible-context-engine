use context_engine_rs::indexing::pipeline::IndexPipeline;
use context_engine_rs::store::open_db;
use std::time::Instant;
use tempfile::TempDir;

#[tokio::test]
async fn repro_full_rebuild_notepad_ade_fresh_db() {
    // Surface info-level logs (incl. the `PERF SUMMARY` line) on stderr.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("context_engine_rs=info")),
        )
        .with_test_writer()
        .try_init();

    let home = TempDir::new().unwrap();
    let repo = "D:/projects/Cpp/notepad-ade".to_string();
    if !std::path::Path::new(&repo).exists() {
        eprintln!("SKIP: source repo not present");
        return;
    }
    let db = open_db(home.path(), &repo).await.expect("open fresh db");
    // voyage = None — exercises parse + all DB writes + Phase 2 with zero embedding
    // work (the all-cached / no-network floor used for every prior measurement).
    let pipeline = IndexPipeline::new(repo.clone(), None);

    let start = Instant::now();
    let res = pipeline.run(&db, None, true, None, None, None, &[]).await;
    let wall = start.elapsed();

    match res {
        Ok(s) => eprintln!(
            "REPRO OK: indexed={} total={} wall={:.1}s",
            s.indexed_files,
            s.total_files,
            wall.as_secs_f64()
        ),
        Err(e) => panic!("REPRO FAILED at: {e:#}"),
    }
}

/// Warm-cache full-rebuild benchmark.
///
/// WRITES TO THE REAL ~/.vibervn INDEX — do NOT run as part of normal CI.
/// Run explicitly with:
///   cargo test --release --test repro_notepad repro_full_rebuild_notepad_ade_warm_cache -- --ignored --nocapture
///
/// Prerequisites:
///   1. The source repo D:/projects/Cpp/notepad-ade must be present.
///   2. The embedding cache dir ~/.vibervn/context-engine/embeddings/voyage-4-lite must exist
///      (delete the surreal DB first to force a real full rebuild; the cache survives).
///
/// What it measures: the complete production path including cache-READ time for ~53K .bin files.
#[ignore]
#[tokio::test]
async fn repro_full_rebuild_notepad_ade_warm_cache() {
    // Surface info-level logs including the PERF SUMMARY line.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("context_engine_rs=info")),
        )
        .with_test_writer()
        .try_init();

    let repo = "D:/projects/Cpp/notepad-ade".to_string();
    if !std::path::Path::new(&repo).exists() {
        eprintln!("SKIP: source repo not present at {repo}");
        return;
    }

    let home_dir = dirs::home_dir().expect("dirs::home_dir() must return a value on this platform");

    // Guard: confirm the embedding cache exists (otherwise the test would just benchmark
    // empty-embedding writes, not the warm-cache-read path the user cares about).
    let cache_dir = home_dir
        .join(".vibervn")
        .join("context-engine")
        .join("embeddings")
        .join("voyage-4-lite");
    if !cache_dir.exists() {
        eprintln!(
            "SKIP: embedding cache dir not found at {}",
            cache_dir.display()
        );
        return;
    }

    // Build EmbeddingCache pointed at the real on-disk cache.
    use context_engine_rs::embedding::cache::EmbeddingCache;
    let cache = match EmbeddingCache::new(&home_dir, "voyage-4-lite") {
        Some(c) => c,
        None => {
            eprintln!("SKIP: could not open EmbeddingCache (new returned None)");
            return;
        }
    };

    // Open (or create) the real SurrealDB at ~/.vibervn/context-engine/surreal/…
    // Note: the surreal dir should be deleted before running this test so the
    // rebuild is genuinely forced from scratch, but open_db handles both cases.
    let db = open_db(&home_dir, &repo)
        .await
        .expect("open real surreal db");

    // voyage = None: on a ~100% warm cache the API is never needed.
    // The concurrency of 4 matches the default in IndexPipeline::new().
    let pipeline = IndexPipeline::new_with_concurrency(repo.clone(), None, 4, Some(cache));

    let start = Instant::now();
    let res = pipeline.run(&db, None, true, None, None, None, &[]).await;
    let wall = start.elapsed();

    match res {
        Ok(s) => {
            eprintln!(
                "REPRO WARM-CACHE OK: indexed={} wall={:.1}s \
                 cache_hit_chunks={} cache_miss_chunks={} embed_total_ms={}",
                s.indexed_files,
                wall.as_secs_f64(),
                s.cache_hit_chunks,
                s.cache_miss_chunks,
                s.embed_total_ms,
            );
        }
        Err(e) => panic!("REPRO WARM-CACHE FAILED: {e:#}"),
    }
}

