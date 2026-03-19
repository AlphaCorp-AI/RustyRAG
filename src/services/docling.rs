use anyhow::Context;
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::document::PageText;

/// Supported file extensions for Docling conversion.
const DOCLING_EXTENSIONS: &[&str] = &["pdf", "docx", "pptx", "xlsx", "html"];

#[derive(Clone)]
pub struct DoclingClient {
    http: Client,
    base_url: String,
    groq_api_key: String,
    vision_model: String,
}

// ── Groq vision API types ───────────────────────────────────────────

#[derive(Serialize)]
struct VisionRequest {
    model: String,
    messages: Vec<VisionMessage>,
    max_tokens: u32,
}

#[derive(Serialize)]
struct VisionMessage {
    role: String,
    content: Vec<VisionContent>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum VisionContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Deserialize)]
struct VisionResponse {
    choices: Vec<VisionChoice>,
}

#[derive(Deserialize)]
struct VisionChoice {
    message: VisionChoiceMessage,
}

#[derive(Deserialize)]
struct VisionChoiceMessage {
    content: Option<String>,
}

impl DoclingClient {
    pub fn new(base_url: &str, groq_api_key: &str, vision_model: &str) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(5))
                .timeout(std::time::Duration::from_secs(300)) // PDFs can be slow
                .pool_idle_timeout(std::time::Duration::from_secs(90))
                .pool_max_idle_per_host(8)
                .build()
                .expect("Failed to build reqwest client for Docling"),
            base_url: base_url.trim_end_matches('/').to_string(),
            groq_api_key: groq_api_key.to_string(),
            vision_model: vision_model.to_string(),
        }
    }

    pub fn is_configured(&self) -> bool {
        !self.base_url.is_empty()
    }

    /// Returns true if this extension should be routed through Docling.
    pub fn supports_extension(ext: &str) -> bool {
        DOCLING_EXTENSIONS.contains(&ext)
    }

    /// Convert a document to markdown via the Docling API, then describe any
    /// embedded images using the vision model.
    pub async fn convert_document(
        &self,
        file_bytes: Vec<u8>,
        filename: &str,
    ) -> anyhow::Result<Vec<PageText>> {
        let markdown = self.call_docling_api(file_bytes, filename).await?;

        // Describe embedded images via vision model
        let markdown = self.replace_images_with_descriptions(&markdown).await;

        // Split by page markers if present, otherwise single page
        Ok(split_markdown_into_pages(&markdown))
    }

    /// POST file to Docling /v1/convert/file and extract markdown.
    ///
    /// Retries with exponential backoff on connection errors (e.g. Docling
    /// restarting) and 5xx responses.
    async fn call_docling_api(
        &self,
        file_bytes: Vec<u8>,
        filename: &str,
    ) -> anyhow::Result<String> {
        let url = format!("{}/v1/convert/file", self.base_url);

        let max_retries = 8u32;
        let mut backoff = std::time::Duration::from_secs(2);

        for attempt in 0..=max_retries {
            let file_part = reqwest::multipart::Part::bytes(file_bytes.clone())
                .file_name(filename.to_string())
                .mime_str("application/octet-stream")
                .context("Failed to create multipart part")?;

            let form = reqwest::multipart::Form::new()
                .part("files", file_part)
                .text("image_export_mode", "embedded")
                .text("do_table_structure", "true");

            let result = self.http.post(&url).multipart(form).send().await;

            let resp = match result {
                Ok(r) => r,
                Err(e) if attempt < max_retries => {
                    tracing::warn!(
                        "Docling connection failed for '{filename}', retrying in {}s ({}/{max_retries}): {e}",
                        backoff.as_secs(),
                        attempt + 1,
                    );
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(std::time::Duration::from_secs(60));
                    continue;
                }
                Err(e) => return Err(e).context("Docling: failed to send request"),
            };

            let status = resp.status();

            if status.is_server_error() && attempt < max_retries {
                let text = resp.text().await.unwrap_or_default();
                tracing::warn!(
                    "Docling returned {status} for '{filename}', retrying in {}s ({}/{max_retries}): {text}",
                    backoff.as_secs(),
                    attempt + 1,
                );
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(std::time::Duration::from_secs(60));
                continue;
            }

            if !status.is_success() {
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("Docling returned {status}: {text}");
            }

            let body: serde_json::Value = resp
                .json()
                .await
                .context("Docling: failed to parse response")?;

            let markdown = extract_markdown_from_response(&body)
                .ok_or_else(|| {
                    tracing::debug!("Docling response structure: {}", serde_json::to_string_pretty(&body).unwrap_or_default());
                    anyhow::anyhow!("Could not find markdown content in Docling response")
                })?;

            return Ok(markdown);
        }

        anyhow::bail!("Docling still unavailable for '{filename}' after {max_retries} retries")
    }

    /// Find base64-embedded images in markdown and replace with text descriptions.
    async fn replace_images_with_descriptions(&self, markdown: &str) -> String {
        if self.groq_api_key.is_empty() {
            return markdown.to_string();
        }

        let re = Regex::new(r"!\[([^\]]*)\]\((data:image/[^;]+;base64,[^)]+)\)")
            .expect("Invalid regex");

        let mut result = markdown.to_string();
        let matches: Vec<(String, String)> = re
            .captures_iter(markdown)
            .map(|cap| (cap[0].to_string(), cap[2].to_string()))
            .collect();

        if matches.is_empty() {
            return result;
        }

        tracing::info!("Describing {} embedded images via vision model…", matches.len());

        for (full_match, data_url) in &matches {
            match self.describe_image(data_url).await {
                Ok(description) => {
                    result = result.replace(
                        full_match,
                        &format!("[Image: {description}]"),
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to describe image: {e}");
                    result = result.replace(full_match, "[Image: description unavailable]");
                }
            }
        }

        result
    }

    /// Call Groq with Llama 4 Scout to describe a base64-encoded image.
    async fn describe_image(&self, data_url: &str) -> anyhow::Result<String> {
        let body = VisionRequest {
            model: self.vision_model.clone(),
            messages: vec![VisionMessage {
                role: "user".into(),
                content: vec![
                    VisionContent::Text {
                        text: "Describe this image concisely for a document search system. \
                               Include all visible text, numbers, data, and structural information."
                            .into(),
                    },
                    VisionContent::ImageUrl {
                        image_url: ImageUrl {
                            url: data_url.to_string(),
                        },
                    },
                ],
            }],
            max_tokens: 512,
        };

        let resp: VisionResponse = self
            .http
            .post("https://api.groq.com/openai/v1/chat/completions")
            .bearer_auth(&self.groq_api_key)
            .json(&body)
            .send()
            .await
            .context("Vision API: failed to send request")?
            .error_for_status()
            .context("Vision API: server returned error")?
            .json()
            .await
            .context("Vision API: failed to parse response")?;

        resp.choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("Vision API returned no content"))
    }
}

// ── Response parsing helpers ────────────────────────────────────────

/// Try multiple known paths to extract markdown from a Docling response.
fn extract_markdown_from_response(body: &serde_json::Value) -> Option<String> {
    // Path 1: document.md_content
    if let Some(md) = body["document"]["md_content"].as_str() {
        return Some(md.to_string());
    }
    // Path 2: document.export_to_markdown (some versions)
    if let Some(md) = body["document"]["markdown"].as_str() {
        return Some(md.to_string());
    }
    // Path 3: rendered.md
    if let Some(md) = body["rendered"]["md"].as_str() {
        return Some(md.to_string());
    }
    // Path 4: md_content at top level
    if let Some(md) = body["md_content"].as_str() {
        return Some(md.to_string());
    }
    // Path 5: content at top level
    if let Some(md) = body["content"].as_str() {
        return Some(md.to_string());
    }
    // Path 6: output.md
    if let Some(md) = body["output"]["md"].as_str() {
        return Some(md.to_string());
    }
    None
}

/// Split markdown into pages using common page break markers.
/// Falls back to a single page if no markers are found.
fn split_markdown_into_pages(markdown: &str) -> Vec<PageText> {
    // Docling may insert HTML comment page markers: <!-- page N -->
    let page_re = Regex::new(r"(?m)^<!--\s*page\s+(\d+)\s*-->").expect("Invalid regex");

    let locations: Vec<(usize, u32)> = page_re
        .captures_iter(markdown)
        .filter_map(|cap| {
            let page_num: u32 = cap[1].parse().ok()?;
            Some((cap.get(0)?.start(), page_num))
        })
        .collect();

    if locations.is_empty() {
        // No page markers — return as single page
        let text = markdown.trim().to_string();
        if text.is_empty() {
            return vec![];
        }
        return vec![PageText {
            text,
            page_number: None,
        }];
    }

    let mut pages = Vec::new();
    for (i, &(start, page_num)) in locations.iter().enumerate() {
        let end = locations
            .get(i + 1)
            .map(|&(s, _)| s)
            .unwrap_or(markdown.len());

        // Skip the marker line itself
        let content_start = markdown[start..]
            .find('\n')
            .map(|pos| start + pos + 1)
            .unwrap_or(start);

        let text = markdown[content_start..end].trim().to_string();
        if !text.is_empty() {
            pages.push(PageText {
                text,
                page_number: Some(page_num),
            });
        }
    }

    // Handle content before the first marker (e.g. page 1 without explicit marker)
    if let Some(&(first_start, _)) = locations.first() {
        let preamble = markdown[..first_start].trim();
        if !preamble.is_empty() {
            pages.insert(
                0,
                PageText {
                    text: preamble.to_string(),
                    page_number: Some(1),
                },
            );
        }
    }

    if pages.is_empty() {
        let text = markdown.trim().to_string();
        if !text.is_empty() {
            return vec![PageText {
                text,
                page_number: None,
            }];
        }
    }

    pages
}
