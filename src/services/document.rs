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
    let pdf_path = path.to_path_buf();

    let result =
        std::panic::catch_unwind(move || pdf_extract::extract_text_from_mem_by_pages(&bytes));

    match result {
        Ok(Ok(pages)) => {
            let mut output = Vec::new();
            for (idx, text) in pages.iter().enumerate() {
                let page_num = (idx + 1) as u32;
                let trimmed = text.trim();

                if trimmed.len() >= 50 {
                    // Good digital text
                    output.push(PageText {
                        text: text.clone(),
                        page_number: Some(page_num),
                    });
                } else if let Some(ocr_text) = ocr_pdf_page(&pdf_path, page_num) {
                    // Scanned page — use OCR text
                    output.push(PageText {
                        text: ocr_text,
                        page_number: Some(page_num),
                    });
                } else if !trimmed.is_empty() {
                    // Fallback to whatever text we got
                    output.push(PageText {
                        text: text.clone(),
                        page_number: Some(page_num),
                    });
                }
            }
            Ok(output)
        }
        Ok(Err(e)) => {
            // pdf_extract failed entirely — try full OCR
            tracing::warn!("PDF extraction failed ({e}), attempting OCR fallback");
            ocr_all_pages(&pdf_path)
        }
        Err(panic) => {
            let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            // Try OCR fallback on panic too
            tracing::warn!("PDF extraction panicked ({msg}), attempting OCR fallback");
            ocr_all_pages(&pdf_path)
        }
    }
}

// ── OCR support (requires pdftoppm + tesseract on system) ───────────

/// OCR a single page of a PDF using pdftoppm + tesseract.
/// Returns None if the tools are not available or OCR fails.
fn ocr_pdf_page(pdf_path: &Path, page_number: u32) -> Option<String> {
    use std::process::Command;

    let tmp_dir = tempfile::tempdir().ok()?;
    let prefix = tmp_dir.path().join("page");
    let page_str = page_number.to_string();

    // Render page to PNG at 300 DPI
    let pdftoppm = Command::new("pdftoppm")
        .args([
            "-png",
            "-f",
            &page_str,
            "-l",
            &page_str,
            "-r",
            "300",
        ])
        .arg(pdf_path)
        .arg(&prefix)
        .output()
        .ok()?;

    if !pdftoppm.status.success() {
        return None;
    }

    // Find the generated PNG file
    let png_path = std::fs::read_dir(tmp_dir.path())
        .ok()?
        .filter_map(|e| e.ok())
        .find(|e| {
            e.path()
                .extension()
                .map(|x| x == "png")
                .unwrap_or(false)
        })?
        .path();

    // OCR with tesseract
    let tesseract = Command::new("tesseract")
        .arg(&png_path)
        .arg("stdout")
        .args(["--dpi", "300"])
        .output()
        .ok()?;

    if !tesseract.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&tesseract.stdout).to_string();
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

/// OCR all pages of a PDF when digital extraction completely fails.
fn ocr_all_pages(pdf_path: &Path) -> anyhow::Result<Vec<PageText>> {
    use std::process::Command;

    // Get page count via pdfinfo
    let pdfinfo = Command::new("pdfinfo")
        .arg(pdf_path)
        .output()
        .context("Failed to run pdfinfo — is poppler-utils installed?")?;

    let info = String::from_utf8_lossy(&pdfinfo.stdout);
    let page_count = info
        .lines()
        .find(|l| l.starts_with("Pages:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|n| n.parse::<u32>().ok())
        .unwrap_or(0);

    if page_count == 0 {
        anyhow::bail!("Could not determine page count for OCR");
    }

    let mut pages = Vec::new();
    for page_num in 1..=page_count {
        if let Some(text) = ocr_pdf_page(pdf_path, page_num) {
            pages.push(PageText {
                text,
                page_number: Some(page_num),
            });
        }
    }

    if pages.is_empty() {
        anyhow::bail!(
            "Failed to extract any text from PDF (both digital and OCR extraction failed)"
        );
    }

    Ok(pages)
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
