use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Events streamed to the frontend during indexing.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IndexEvent {
    /// Indexing run started for a repo.
    Started {
        repo: String,
        total_files: u64,
        is_rebuild: bool,
    },
    /// A file was parsed (stage 1 complete for this file).
    FileParsed {
        file: String,
        elapsed_ms: u64,
    },
    /// An embedding API call is starting.
    EmbedCallStart {
        file: String,
        chunks: usize,
        key_hint: String,
        active_calls: u64,
    },
    /// An embedding API call finished.
    EmbedCallDone {
        file: String,
        chunks: usize,
        elapsed_ms: u64,
        active_calls: u64,
        cached: bool,
    },
    /// A file completed the full embed+write cycle.
    FileIndexed {
        file: String,
        indexed: u64,
        total: u64,
        elapsed_ms: u64,
    },
    /// Phase 2 edge resolution started.
    Phase2Start { repo: String },
    /// Phase 2 edge resolution done.
    Phase2Done { repo: String, elapsed_ms: u64 },
    /// Indexing completed successfully.
    Completed {
        repo: String,
        indexed_files: u64,
        total_files: u64,
        elapsed_ms: u64,
    },
    /// Indexing failed.
    Failed { repo: String, error: String },
}

/// Shared event broadcaster for indexing progress.
#[derive(Clone)]
pub struct IndexEventBus {
    tx: broadcast::Sender<IndexEvent>,
    pub active_api_calls: Arc<AtomicU64>,
}

impl Default for IndexEventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexEventBus {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1024);
        Self {
            tx,
            active_api_calls: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn emit(&self, event: IndexEvent) {
        let _ = self.tx.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<IndexEvent> {
        self.tx.subscribe()
    }

    pub fn inc_api_calls(&self) -> u64 {
        self.active_api_calls.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn dec_api_calls(&self) -> u64 {
        let prev = self.active_api_calls.fetch_sub(1, Ordering::Relaxed);
        prev.saturating_sub(1)
    }

    pub fn active_calls(&self) -> u64 {
        self.active_api_calls.load(Ordering::Relaxed)
    }
}
