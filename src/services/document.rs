use std::path::Path;

use anyhow::Context;
use text_splitter::TextSplitter;

// ── Page-level extraction result ────────────────────────────────────

/// A single page of extracted text with its 1-based page number.
#[derive(Debug, Clone)]
pub struct PageText {
    pub text: String,
    /// 1-based page number (`None` for non-paginated formats like .txt).
    pub page_number: Option<u32>,
}

// ── Text extraction (from file path) ───────────────────────────────

/// Extract text from a file on disk, returning one [`PageText`] per logical
/// page.  For `.txt` files there is a single entry with `page_number = None`.
pub fn extract_pages_from_path(path: &Path, filename: &str) -> anyhow::Result<Vec<PageText>> {
    let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();

    match ext.as_str() {
        "txt" => {
            let text = std::fs::read_to_string(path).context("Failed to read text file")?;
            Ok(vec![PageText {
                text,
                page_number: None,
            }])
        }
        "pdf" => extract_pdf_pages(path),
        _ => Err(anyhow::anyhow!("Unsupported file type: .{ext}")),
    }
}

fn extract_pdf_pages(path: &Path) -> anyhow::Result<Vec<PageText>> {
    let bytes = std::fs::read(path).context("Failed to read PDF file into memory")?;

    let result =
        std::panic::catch_unwind(move || pdf_extract::extract_text_from_mem_by_pages(&bytes));

    match result {
        Ok(Ok(pages)) => Ok(pages
            .into_iter()
            .enumerate()
            .filter(|(_, text)| !text.trim().is_empty())
            .map(|(idx, text)| PageText {
                text,
                page_number: Some((idx + 1) as u32),
            })
            .collect()),
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

        if name.ends_with('/') {
            continue;
        }

        let basename = name.rsplit('/').next().unwrap_or(&name);
        if name.contains("__MACOSX") || basename.starts_with("._") {
            continue;
        }

        let ext = name.rsplit('.').next().unwrap_or("").to_lowercase();
        if ext != "txt" && ext != "pdf" {
            tracing::debug!("Skipping unsupported file in ZIP: {name}");
            continue;
        }

        let mut tmp = tempfile::NamedTempFile::new().context("Failed to create temp file")?;
        std::io::copy(&mut entry, &mut tmp)
            .context(format!("Failed to write {name} to temp file"))?;
        tmp.flush()
            .context(format!("Failed to flush temp file for {name}"))?;
        results.push((name, tmp));
    }

    Ok(results)
}

// ── Semantic chunking ───────────────────────────────────────────────

/// A chunk of text with metadata about its origin.
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub page_number: Option<u32>,
    pub chunk_index: usize,
}

/// Split pages into semantically coherent chunks using sentence and paragraph
/// boundaries.  `max_characters` is the target ceiling per chunk.
/// `overlap_chars` prepends trailing context from the previous chunk.
pub fn chunk_pages(
    pages: &[PageText],
    max_characters: usize,
    overlap_chars: usize,
) -> Vec<TextChunk> {
    let splitter = TextSplitter::new(max_characters);
    let mut result = Vec::new();
    let mut global_idx: usize = 0;
    let mut prev_chunk_tail = String::new();

    for page in pages {
        if page.text.trim().is_empty() {
            continue;
        }

        let raw_chunks: Vec<&str> = splitter.chunks(&page.text).collect();

        for chunk_str in raw_chunks {
            let trimmed = chunk_str.trim();
            if trimmed.is_empty() {
                continue;
            }

            let text = if overlap_chars > 0 && !prev_chunk_tail.is_empty() {
                format!("{prev_chunk_tail}\n\n{trimmed}")
            } else {
                trimmed.to_string()
            };

            if overlap_chars > 0 {
                let char_count = trimmed.chars().count();
                let skip = char_count.saturating_sub(overlap_chars);
                prev_chunk_tail = trimmed.chars().skip(skip).collect();
            }

            result.push(TextChunk {
                text,
                page_number: page.page_number,
                chunk_index: global_idx,
            });
            global_idx += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_pages_basic() {
        let pages = vec![PageText {
            text: "Hello world. This is a test. Another sentence here.".to_string(),
            page_number: Some(1),
        }];
        let chunks = chunk_pages(&pages, 5000, 0);
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].page_number, Some(1));
        assert_eq!(chunks[0].chunk_index, 0);
    }

    #[test]
    fn test_chunk_pages_empty() {
        let pages = vec![PageText {
            text: "".to_string(),
            page_number: Some(1),
        }];
        let chunks = chunk_pages(&pages, 1000, 0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_pages_multi_page() {
        let pages = vec![
            PageText {
                text: "Page one content.".to_string(),
                page_number: Some(1),
            },
            PageText {
                text: "Page two content.".to_string(),
                page_number: Some(2),
            },
        ];
        let chunks = chunk_pages(&pages, 5000, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].page_number, Some(1));
        assert_eq!(chunks[1].page_number, Some(2));
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].chunk_index, 1);
    }

    #[test]
    fn test_chunk_pages_with_overlap() {
        let pages = vec![PageText {
            text: "First sentence is quite long and has many words. Second sentence also has content.".to_string(),
            page_number: Some(1),
        }];
        let chunks = chunk_pages(&pages, 50, 10);
        if chunks.len() > 1 {
            assert!(chunks[1].text.contains('\n'));
        }
    }
}
