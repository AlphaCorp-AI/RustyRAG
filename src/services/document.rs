use std::path::Path;

use anyhow::Context;

// ── Text extraction (from bytes) ───────────────────────────────────

/// Extract text from in-memory file data based on its extension.
/// Supports `.txt` and `.pdf`.
#[allow(dead_code)]
pub fn extract_text(filename: &str, data: &[u8]) -> anyhow::Result<String> {
    let ext = filename
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "txt" => String::from_utf8(data.to_vec()).context("File is not valid UTF-8"),
        "pdf" => {
            use std::io::Write as _;
            let mut tmp = tempfile::NamedTempFile::new().context("Failed to create temp file")?;
            tmp.write_all(data).context("Failed to write PDF to temp file")?;
            extract_pdf_from_path(tmp.path())
        }
        _ => Err(anyhow::anyhow!("Unsupported file type: .{ext}")),
    }
}

// ── Text extraction (from file path) ───────────────────────────────

/// Extract text from a file on disk. Extension is inferred from `filename`.
pub fn extract_text_from_path(path: &Path, filename: &str) -> anyhow::Result<String> {
    let ext = filename
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "txt" => std::fs::read_to_string(path).context("Failed to read text file"),
        "pdf" => extract_pdf_from_path(path),
        _ => Err(anyhow::anyhow!("Unsupported file type: .{ext}")),
    }
}

fn extract_pdf_from_path(path: &Path) -> anyhow::Result<String> {
    // stdout is globally redirected to /dev/null at startup (see main.rs),
    // so pdf-extract's noisy debug prints are silenced process-wide.
    let path_buf = path.to_path_buf();
    let result = std::panic::catch_unwind(move || pdf_extract::extract_text(path_buf));
    match result {
        Ok(Ok(text)) => Ok(text),
        Ok(Err(e)) => Err(anyhow::anyhow!("PDF extraction failed: {e}")),
        Err(panic) => {
            let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            Err(anyhow::anyhow!(
                "PDF extraction panicked (malformed PDF): {msg}"
            ))
        }
    }
}

// ── ZIP unpacking (to temp files for concurrent processing) ─────────

/// Open a `.zip` archive from disk and unpack each supported entry
/// (`.txt`, `.pdf`) into its own temporary file.
///
/// Returns `(filename, temp_file)` pairs.  The caller is responsible for
/// processing each temp file concurrently and dropping it when done.
pub fn unpack_zip_entries(
    zip_path: &Path,
) -> anyhow::Result<Vec<(String, tempfile::NamedTempFile)>> {
    use std::io::Write as _;

    let file = std::fs::File::open(zip_path).context("Failed to open ZIP temp file")?;
    let mut archive = zip::ZipArchive::new(file).context("Failed to read ZIP archive")?;

    let mut results = Vec::new();
    let entry_count = archive.len();

    for i in 0..entry_count {
        let mut entry = archive
            .by_index(i)
            .context(format!("Failed to read ZIP entry {i}"))?;

        let name = entry.name().to_string();

        // Skip directories
        if name.ends_with('/') {
            continue;
        }

        // Skip macOS resource fork metadata (.__* files inside __MACOSX/)
        let basename = name.rsplit('/').next().unwrap_or(&name);
        if name.contains("__MACOSX") || basename.starts_with("._") {
            continue;
        }

        let ext = name.rsplit('.').next().unwrap_or("").to_lowercase();
        if ext != "txt" && ext != "pdf" {
            tracing::debug!("Skipping unsupported file in ZIP: {name}");
            continue;
        }

        let mut tmp =
            tempfile::NamedTempFile::new().context("Failed to create temp file")?;
        std::io::copy(&mut entry, &mut tmp)
            .context(format!("Failed to write {name} to temp file"))?;
        tmp.flush()
            .context(format!("Failed to flush temp file for {name}"))?;
        results.push((name, tmp));
    }

    Ok(results)
}

// ── Chunking ───────────────────────────────────────────────────────

/// Split `text` into chunks of approximately `chunk_size` **words** with
/// `chunk_overlap` words of overlap between consecutive chunks.
pub fn chunk_text(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return vec![];
    }
    if words.len() <= chunk_size {
        return vec![words.join(" ")];
    }

    let step = chunk_size.saturating_sub(chunk_overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk = words[start..end].join(" ");
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
        if end >= words.len() {
            break;
        }
        start += step;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_basic() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunk_text(text, 4, 1);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "one two three four");
        assert_eq!(chunks[1], "four five six seven");
        assert_eq!(chunks[2], "seven eight nine ten");
    }

    #[test]
    fn test_chunk_text_short() {
        let text = "hello world";
        let chunks = chunk_text(text, 100, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn test_chunk_text_empty() {
        let chunks = chunk_text("", 10, 2);
        assert!(chunks.is_empty());
    }
}
