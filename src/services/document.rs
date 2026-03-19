use std::path::Path;

use text_splitter::TextSplitter;

use super::docling::DoclingClient;

// ── Page-level extraction result ────────────────────────────────────

/// A single page of extracted text with its 1-based page number.
#[derive(Debug, Clone)]
pub struct PageText {
    pub text: String,
    /// 1-based page number (`None` for non-paginated formats like .txt).
    pub page_number: Option<u32>,
}

// ── Document extraction ─────────────────────────────────────────────

/// Supported file extensions for upload.
const TEXT_EXTENSIONS: &[&str] = &["txt"];

/// Extract text from a file, routing to Docling for rich formats and
/// reading directly for plain text.
pub async fn extract_document(
    path: &Path,
    filename: &str,
    docling: &DoclingClient,
) -> anyhow::Result<Vec<PageText>> {
    let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();

    if TEXT_EXTENSIONS.contains(&ext.as_str()) {
        let text = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read text file: {e}"))?;
        return Ok(vec![PageText {
            text,
            page_number: None,
        }]);
    }

    if DoclingClient::supports_extension(&ext) {
        if !docling.is_configured() {
            anyhow::bail!(
                "Docling is not configured (set DOCLING_URL). \
                 Required for .{ext} files."
            );
        }
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read file: {e}"))?;
        return docling.convert_document(bytes, filename).await;
    }

    Err(anyhow::anyhow!(
        "Unsupported file type: .{ext} (allowed: .txt, .pdf, .docx, .pptx, .xlsx, .html)"
    ))
}

/// Returns true if the file extension is supported for upload.
pub fn is_supported_extension(ext: &str) -> bool {
    TEXT_EXTENSIONS.contains(&ext) || DoclingClient::supports_extension(ext)
}

// ── ZIP unpacking ───────────────────────────────────────────────────

/// Open a `.zip` archive and unpack each supported entry into a temp file.
pub fn unpack_zip_entries(
    zip_path: &Path,
) -> anyhow::Result<Vec<(String, tempfile::NamedTempFile)>> {
    use std::io::Write as _;

    let file =
        std::fs::File::open(zip_path).map_err(|e| anyhow::anyhow!("Failed to open ZIP: {e}"))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| anyhow::anyhow!("Failed to read ZIP: {e}"))?;

    let mut results = Vec::new();

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| anyhow::anyhow!("Failed to read ZIP entry {i}: {e}"))?;

        let name = entry.name().to_string();

        // Skip directories, macOS metadata, hidden files
        if name.ends_with('/') {
            continue;
        }
        let basename = name.rsplit('/').next().unwrap_or(&name);
        if name.contains("__MACOSX") || basename.starts_with("._") {
            continue;
        }

        let ext = name.rsplit('.').next().unwrap_or("").to_lowercase();
        if !is_supported_extension(&ext) {
            tracing::debug!("Skipping unsupported file in ZIP: {name}");
            continue;
        }

        let mut tmp =
            tempfile::NamedTempFile::new().map_err(|e| anyhow::anyhow!("Temp file: {e}"))?;
        std::io::copy(&mut entry, &mut tmp)
            .map_err(|e| anyhow::anyhow!("Failed to write {name}: {e}"))?;
        tmp.flush()
            .map_err(|e| anyhow::anyhow!("Failed to flush {name}: {e}"))?;
        results.push((name, tmp));
    }

    Ok(results)
}

// ── Semantic chunking ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub page_number: Option<u32>,
    pub chunk_index: usize,
}

/// Split pages into semantically coherent chunks using sentence and paragraph
/// boundaries.
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

        for chunk_str in splitter.chunks(&page.text) {
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
            PageText { text: "Page one content.".into(), page_number: Some(1) },
            PageText { text: "Page two content.".into(), page_number: Some(2) },
        ];
        let chunks = chunk_pages(&pages, 5000, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].page_number, Some(1));
        assert_eq!(chunks[1].page_number, Some(2));
    }

    #[test]
    fn test_chunk_pages_with_overlap() {
        let pages = vec![PageText {
            text: "First sentence is quite long and has many words. Second sentence also has content.".into(),
            page_number: Some(1),
        }];
        let chunks = chunk_pages(&pages, 50, 10);
        if chunks.len() > 1 {
            assert!(chunks[1].text.contains('\n'));
        }
    }

    #[test]
    fn test_is_supported_extension() {
        assert!(is_supported_extension("txt"));
        assert!(is_supported_extension("pdf"));
        assert!(is_supported_extension("docx"));
        assert!(is_supported_extension("pptx"));
        assert!(!is_supported_extension("exe"));
    }
}
