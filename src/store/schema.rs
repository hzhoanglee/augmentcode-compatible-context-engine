/// SurrealDB DDL — applied at every startup to bootstrap or migrate tables, fields, and indexes.
///
/// Field definitions use `DEFINE FIELD OVERWRITE` so that re-running this DDL on an
/// EXISTING database actively updates the persisted field type. Without OVERWRITE, a
/// plain `DEFINE FIELD` is a no-op against a field that already exists — the old type
/// stays in the datastore and new writes that use the corrected type are rejected with
/// a FieldCheck type-violation error, rolling back the entire transaction. OVERWRITE
/// ensures the on-disk definition is always replaced with the current declaration.
///
/// Table definitions use `IF NOT EXISTS` to avoid recreating (and potentially
/// misconfiguring) tables that already hold data. Recreating a table is safe for
/// the schema but IF NOT EXISTS makes the intent explicit.
///
/// Index definitions use `IF NOT EXISTS` so that re-running the DDL on a live database
/// never triggers a rebuild of an existing index unnecessarily; a new index name will
/// still be created.
pub const SCHEMA_DDL: &str = r#"
DEFINE TABLE IF NOT EXISTS symbol SCHEMAFULL;
DEFINE FIELD OVERWRITE name       ON symbol TYPE string;
DEFINE FIELD OVERWRITE kind       ON symbol TYPE string;
DEFINE FIELD OVERWRITE file       ON symbol TYPE string;
DEFINE FIELD OVERWRITE line_start ON symbol TYPE int;
DEFINE FIELD OVERWRITE line_end   ON symbol TYPE int;
DEFINE FIELD OVERWRITE signature  ON symbol TYPE option<string>;
DEFINE FIELD OVERWRITE parent     ON symbol TYPE option<string>;
DEFINE INDEX IF NOT EXISTS idx_symbol_file ON symbol FIELDS file;
DEFINE INDEX IF NOT EXISTS idx_symbol_name ON symbol FIELDS name;

DEFINE TABLE IF NOT EXISTS chunk SCHEMAFULL;
DEFINE FIELD OVERWRITE file       ON chunk TYPE string;
DEFINE FIELD OVERWRITE line_start ON chunk TYPE int;
DEFINE FIELD OVERWRITE line_end   ON chunk TYPE int;
DEFINE FIELD OVERWRITE content    ON chunk TYPE string;
DEFINE FIELD OVERWRITE embedding  ON chunk TYPE array<float>;
DEFINE FIELD OVERWRITE symbol_ref ON chunk TYPE option<string>;
DEFINE INDEX IF NOT EXISTS idx_chunk_file ON chunk FIELDS file;

DEFINE TABLE IF NOT EXISTS calls TYPE RELATION IN symbol OUT symbol;
DEFINE FIELD OVERWRITE line     ON calls TYPE int;
DEFINE FIELD OVERWRITE in_file  ON calls TYPE string;
DEFINE FIELD OVERWRITE out_file ON calls TYPE string;
DEFINE INDEX IF NOT EXISTS idx_calls_in_file ON calls FIELDS in_file;
DEFINE INDEX IF NOT EXISTS idx_calls_out_file ON calls FIELDS out_file;

DEFINE TABLE IF NOT EXISTS uses TYPE RELATION IN symbol OUT symbol;
DEFINE FIELD OVERWRITE in_file  ON uses TYPE string;
DEFINE FIELD OVERWRITE out_file ON uses TYPE string;
DEFINE INDEX IF NOT EXISTS idx_uses_in_file ON uses FIELDS in_file;
DEFINE INDEX IF NOT EXISTS idx_uses_out_file ON uses FIELDS out_file;

DEFINE TABLE IF NOT EXISTS imports TYPE RELATION IN symbol OUT symbol;
DEFINE FIELD OVERWRITE in_file  ON imports TYPE string;
DEFINE FIELD OVERWRITE out_file ON imports TYPE string;
DEFINE INDEX IF NOT EXISTS idx_imports_in_file ON imports FIELDS in_file;
DEFINE INDEX IF NOT EXISTS idx_imports_out_file ON imports FIELDS out_file;

DEFINE TABLE IF NOT EXISTS contains TYPE RELATION IN symbol OUT symbol;
DEFINE FIELD OVERWRITE in_file  ON contains TYPE string;
DEFINE FIELD OVERWRITE out_file ON contains TYPE string;
DEFINE INDEX IF NOT EXISTS idx_contains_in_file ON contains FIELDS in_file;
DEFINE INDEX IF NOT EXISTS idx_contains_out_file ON contains FIELDS out_file;

DEFINE TABLE IF NOT EXISTS implements TYPE RELATION IN symbol OUT symbol;
DEFINE FIELD OVERWRITE in_file  ON implements TYPE string;
DEFINE FIELD OVERWRITE out_file ON implements TYPE string;
DEFINE INDEX IF NOT EXISTS idx_implements_in_file ON implements FIELDS in_file;
DEFINE INDEX IF NOT EXISTS idx_implements_out_file ON implements FIELDS out_file;

DEFINE TABLE IF NOT EXISTS file_meta SCHEMAFULL;
DEFINE FIELD OVERWRITE path  ON file_meta TYPE string;
DEFINE FIELD OVERWRITE mtime ON file_meta TYPE int;
DEFINE FIELD OVERWRITE size  ON file_meta TYPE int;
DEFINE FIELD OVERWRITE repo  ON file_meta TYPE string;
DEFINE INDEX IF NOT EXISTS idx_filemeta_path ON file_meta FIELDS path UNIQUE;

DEFINE TABLE IF NOT EXISTS index_meta SCHEMAFULL;
DEFINE FIELD OVERWRITE key   ON index_meta TYPE string;
DEFINE FIELD OVERWRITE value ON index_meta TYPE string;
DEFINE INDEX IF NOT EXISTS idx_meta_key ON index_meta FIELDS key UNIQUE;
"#;
