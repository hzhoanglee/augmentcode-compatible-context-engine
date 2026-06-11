//! Windows Defender real-time protection exclusion management.
//!
//! On Windows, checks whether the context-engine data directory and process are
//! excluded from real-time scanning (which severely impacts indexing I/O). Provides
//! a function to add exclusions via an elevated PowerShell process (triggers UAC).
//! On non-Windows platforms, all functions report "not applicable."
//!
//! ## The non-admin read problem (incident: "Failed — retry" loop)
//!
//! `Get-MpPreference` succeeds for a non-elevated process, but its
//! `ExclusionPath`/`ExclusionProcess` properties come back as a single-element
//! array whose only entry is the literal string
//! `"N/A: Must be an administrator to view exclusions"` — it does NOT throw and
//! does NOT return an empty list. So a non-admin process can never read the real
//! exclusion list, which means it can never *verify* an exclusion exists.
//!
//! The server runs non-elevated, so:
//!   1. verification must happen INSIDE the elevated `add` process (which can read
//!      prefs), reported back via the child exit code; and
//!   2. once verified, we drop a durable marker file in the data dir so the
//!      non-admin `check_status` can report "excluded" on later page loads without
//!      needing to read Defender prefs again.

use serde::Serialize;

#[derive(Debug, Serialize, Clone)]
pub struct DefenderStatus {
    pub platform: &'static str,
    pub applicable: bool,
    pub data_dir: String,
    pub process_name: String,
    pub data_dir_excluded: bool,
    pub process_excluded: bool,
}

#[derive(Debug, Serialize)]
pub struct DefenderExcludeResult {
    pub success: bool,
    pub message: String,
}

/// Sentinel string Defender returns to non-admin readers of the exclusion list.
#[cfg(windows)]
const NEEDS_ADMIN_SENTINEL: &str = "Must be an administrator";

/// Name of the durable marker written after exclusions are verified by an
/// elevated process. Lives in the data dir so it survives restarts.
#[cfg(windows)]
const MARKER_FILE: &str = ".defender-excluded";

#[cfg(not(windows))]
pub fn check_status(_data_dir: &str) -> DefenderStatus {
    DefenderStatus {
        platform: std::env::consts::OS,
        applicable: false,
        data_dir: String::new(),
        process_name: String::new(),
        data_dir_excluded: false,
        process_excluded: false,
    }
}

#[cfg(not(windows))]
pub fn add_exclusions(_data_dir: &str) -> DefenderExcludeResult {
    DefenderExcludeResult {
        success: false,
        message: "Not applicable on this platform".into(),
    }
}

#[cfg(windows)]
pub fn check_status(data_dir: &str) -> DefenderStatus {
    let process_name = "context-engine.exe";
    let normalized_dir = data_dir
        .replace('/', "\\")
        .trim_end_matches('\\')
        .to_string();

    // The marker is authoritative once written: it means an elevated process
    // already verified the exclusions exist. A non-admin server cannot re-read
    // the exclusion list, so without this it would loop forever showing the banner.
    let marker = std::path::Path::new(&normalized_dir).join(MARKER_FILE);
    if marker.exists() {
        return DefenderStatus {
            platform: "windows",
            applicable: true,
            data_dir: normalized_dir,
            process_name: process_name.to_string(),
            data_dir_excluded: true,
            process_excluded: true,
        };
    }

    // Try to read the exclusion list. Detect the "needs admin" sentinel so we
    // don't misinterpret it as a real (non-matching) exclusion entry.
    let script = format!(
        r#"
$ErrorActionPreference = 'Stop'
try {{
    $prefs = Get-MpPreference
    $paths = @($prefs.ExclusionPath)
    $procs = @($prefs.ExclusionProcess)
    if (($paths -join ' ') -like '*{sentinel}*') {{
        Write-Output 'NEEDADMIN'
    }} else {{
        $pathExcluded = $false
        $procExcluded = $false
        $target = "{normalized_dir}".TrimEnd('\')
        foreach ($p in $paths) {{
            if (-not $p) {{ continue }}
            $norm = $p.TrimEnd('\')
            if ($target -ieq $norm -or $target.StartsWith($norm + '\', [System.StringComparison]::OrdinalIgnoreCase)) {{
                $pathExcluded = $true
                break
            }}
        }}
        foreach ($p in $procs) {{
            if ($p -ieq "{process_name}") {{
                $procExcluded = $true
                break
            }}
        }}
        Write-Output "OK|$pathExcluded|$procExcluded"
    }}
}} catch {{
    Write-Output "ERR|$($_.Exception.Message)"
}}
"#,
        sentinel = NEEDS_ADMIN_SENTINEL,
    );

    let output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let line = stdout.trim();
            if let Some(rest) = line.strip_prefix("OK|") {
                let parts: Vec<&str> = rest.split('|').collect();
                let data_dir_excluded = parts
                    .first()
                    .map(|s| s.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let process_excluded = parts
                    .get(1)
                    .map(|s| s.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                DefenderStatus {
                    platform: "windows",
                    applicable: true,
                    data_dir: normalized_dir,
                    process_name: process_name.to_string(),
                    data_dir_excluded,
                    process_excluded,
                }
            } else if line == "NEEDADMIN" {
                // Non-admin can't read the list and no marker exists yet → we don't
                // know the state. Show the banner (applicable + not-excluded) so the
                // user can trigger the elevated add, which is the only path that can
                // both verify and write the marker.
                DefenderStatus {
                    platform: "windows",
                    applicable: true,
                    data_dir: normalized_dir,
                    process_name: process_name.to_string(),
                    data_dir_excluded: false,
                    process_excluded: false,
                }
            } else {
                // ERR| or unexpected — Defender disabled/absent. Suppress the banner.
                DefenderStatus {
                    platform: "windows",
                    applicable: false,
                    data_dir: normalized_dir,
                    process_name: process_name.to_string(),
                    data_dir_excluded: false,
                    process_excluded: false,
                }
            }
        }
        _ => DefenderStatus {
            platform: "windows",
            applicable: false,
            data_dir: normalized_dir,
            process_name: process_name.to_string(),
            data_dir_excluded: false,
            process_excluded: false,
        },
    }
}

#[cfg(windows)]
pub fn add_exclusions(data_dir: &str) -> DefenderExcludeResult {
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    let process_name = "context-engine.exe";
    let normalized_dir = data_dir
        .replace('/', "\\")
        .trim_end_matches('\\')
        .to_string();

    // Inner script runs ELEVATED, so it can both add and *read back* the
    // exclusions to verify. It exits 0 only when both exclusions are confirmed
    // present, 2 otherwise — the exit code is how the non-admin parent learns
    // whether the add truly succeeded (the parent cannot read the list itself).
    let inner_script = format!(
        r#"
$ErrorActionPreference = 'Stop'
try {{
    Add-MpPreference -ExclusionPath '{dir}'
    Add-MpPreference -ExclusionProcess '{proc}'
    $prefs = Get-MpPreference
    $target = '{dir}'.TrimEnd('\')
    $pathOk = $false
    foreach ($p in @($prefs.ExclusionPath)) {{
        if (-not $p) {{ continue }}
        $norm = $p.TrimEnd('\')
        if ($target -ieq $norm -or $target.StartsWith($norm + '\', [System.StringComparison]::OrdinalIgnoreCase)) {{
            $pathOk = $true; break
        }}
    }}
    $procOk = $false
    foreach ($p in @($prefs.ExclusionProcess)) {{
        if ($p -ieq '{proc}') {{ $procOk = $true; break }}
    }}
    if ($pathOk -and $procOk) {{ exit 0 }} else {{ exit 2 }}
}} catch {{
    exit 3
}}
"#,
        dir = normalized_dir,
        proc = process_name,
    );

    // Encode the inner script as UTF-16LE base64 for `-EncodedCommand`. This
    // sidesteps all the nested quoting/escaping landmines of passing a script
    // through `Start-Process -ArgumentList` (the previous escape_ps_arg approach
    // was fragile and is gone).
    let utf16: Vec<u8> = inner_script
        .encode_utf16()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    let encoded = STANDARD.encode(&utf16);

    // Outer (non-elevated) launcher: RunAs triggers UAC, -Wait blocks until the
    // elevated child exits, -PassThru gives us its ExitCode to relay.
    let outer_script = format!(
        "$p = Start-Process powershell -Verb RunAs -Wait -PassThru -ArgumentList '-NoProfile','-NonInteractive','-EncodedCommand','{encoded}'; exit $p.ExitCode"
    );

    let output = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &outer_script])
        .output();

    match output {
        Ok(o) => {
            // ExitCode is relayed via the outer process exit code.
            match o.status.code() {
                Some(0) => {
                    // Verified inside the elevated process. Persist the marker so
                    // the non-admin server reports "excluded" on future loads.
                    write_marker(&normalized_dir);
                    DefenderExcludeResult {
                        success: true,
                        message: "Exclusions added and verified.".into(),
                    }
                }
                Some(2) => DefenderExcludeResult {
                    success: false,
                    message: "Exclusions were not present after adding — try again.".into(),
                },
                Some(3) => DefenderExcludeResult {
                    success: false,
                    message: "Add-MpPreference failed inside the elevated process.".into(),
                },
                // Non-zero/None typically means UAC was declined (the user clicked
                // "No" on the prompt), so Start-Process never launched the child.
                other => DefenderExcludeResult {
                    success: false,
                    message: format!(
                        "Elevation was cancelled or failed (exit {other:?}). \
                         Approve the UAC prompt and retry."
                    ),
                },
            }
        }
        Err(e) => DefenderExcludeResult {
            success: false,
            message: format!("Failed to launch PowerShell: {e}"),
        },
    }
}

/// Write the durable "exclusions verified" marker. Best-effort: a failure here
/// only means the banner may reappear once more, not a correctness problem.
#[cfg(windows)]
fn write_marker(normalized_dir: &str) {
    let dir = std::path::Path::new(normalized_dir);
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(
        dir.join(MARKER_FILE),
        b"context-engine verified Windows Defender exclusions\n",
    );
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    // The core invariant of the "Failed — retry loop" fix: once the marker file
    // exists (written only after an elevated process verified the exclusions),
    // check_status reports excluded WITHOUT shelling out to Get-MpPreference —
    // which a non-admin server can never read. Without this short-circuit the
    // banner reappears forever.
    #[test]
    fn marker_file_short_circuits_to_excluded() {
        let tmp = std::env::temp_dir().join(format!("ce-defender-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let dir = tmp.to_string_lossy().to_string();

        // No marker yet: a real (possibly NEEDADMIN) probe runs; we only assert
        // it does not spuriously claim excluded.
        let before = check_status(&dir);
        assert!(!(before.data_dir_excluded && before.process_excluded));

        // After the marker is written, status is excluded regardless of admin.
        write_marker(&dir);
        let after = check_status(&dir);
        assert!(after.applicable);
        assert!(after.data_dir_excluded);
        assert!(after.process_excluded);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn normalizes_forward_slashes_and_trailing_sep() {
        // The marker lookup must resolve the same dir regardless of slash style,
        // otherwise a forward-slash data_dir would never see its own marker.
        let tmp = std::env::temp_dir().join(format!("ce-defender-norm-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let forward = format!("{}/", tmp.to_string_lossy().replace('\\', "/"));

        write_marker(&forward.replace('/', "\\"));
        let status = check_status(&forward);
        assert!(status.data_dir_excluded && status.process_excluded);

        std::fs::remove_dir_all(&tmp).ok();
    }
}
