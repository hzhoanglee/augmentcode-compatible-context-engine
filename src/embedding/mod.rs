pub mod cache;
pub mod voyage;

/// Input type hint for the embedding API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    Document,
    Query,
}

impl InputType {
    pub fn as_str(&self) -> &'static str {
        match self {
            InputType::Document => "document",
            InputType::Query => "query",
        }
    }
}
