use serde_json::Value;

/// System prompt for competition RAG answering.
/// Includes answer-type-specific instructions and null detection.
/// For free_text, targets all 5 LLM judge criteria:
///   1. Correctness  2. Completeness  3. Grounding  4. Confidence calibration  5. Clarity
///
/// The prompt asks the LLM to cite which excerpt numbers it used via a SOURCES line,
/// so we can restrict `retrieved_chunk_pages` to only the pages actually used (improving grounding).
pub fn build_competition_system_prompt(context: &str, answer_type: &str) -> String {
    let type_instruction = match answer_type {
        "number" => {
            "Return ONLY the numeric value (integer or decimal). No units, no explanation, no text.\n\
             Examples: 42, 3.14, 1000000\n\
             Try your best to find and extract the answer from the context. Use reasoning and inference from the context when needed.\n\
             Only respond with UNANSWERABLE if the context contains absolutely no relevant information."
        }
        "boolean" => {
            "Return ONLY the word 'true' or 'false'. No explanation.\n\
             Try your best to determine the answer from the context. Use reasoning and inference when the answer is implied but not stated explicitly.\n\
             For yes/no legal questions, lean toward answering based on what the context supports rather than saying UNANSWERABLE.\n\
             Only respond with UNANSWERABLE if the context contains absolutely no relevant information."
        }
        "name" => {
            "Return ONLY the exact name or entity. No explanation, no surrounding sentence.\n\
             Copy the name verbatim from the context — do not paraphrase or abbreviate.\n\
             Examples of valid responses: \"John Smith\", \"CFI 010/2024\", \"DIFC Courts\"\n\
             Try your best to find and extract the answer from the context.\n\
             Only respond with UNANSWERABLE if the context contains absolutely no relevant information."
        }
        "names" => {
            "Return ONLY a semicolon-separated list of names exactly as they appear in the documents.\n\
             No explanation, no numbering, no extra text. Copy names verbatim from the context.\n\
             Example format: Alice Smith; Bob Jones; Carol White\n\
             Try your best to find and extract all relevant names from the context.\n\
             Only respond with UNANSWERABLE if the context contains absolutely no relevant information."
        }
        "date" => {
            "Return ONLY the date in YYYY-MM-DD format. No explanation.\n\
             Example: 2024-03-15\n\
             Try your best to find and extract the date from the context.\n\
             Only respond with UNANSWERABLE if the context contains absolutely no relevant information."
        }
        "free_text" => {
            "Provide a clear, concise answer in 1-3 sentences (max 280 characters total).\n\
             Rules:\n\
             - Every claim must be directly supported by the provided context.\n\
             - Address all aspects of the question completely.\n\
             - Do not state anything not present in the context.\n\
             - Try your best to answer using the available context. Use reasoning and inference when needed.\n\
             - If the answer is uncertain or partial, provide what you can and note the uncertainty.\n\
             - Only if the context contains absolutely no relevant information, respond exactly with: \
             \"There is no information on this question in the provided documents.\""
        }
        _ => {
            "Answer based on the context. Only respond with UNANSWERABLE if the context contains absolutely no relevant information."
        }
    };

    format!(
        "You are a precise legal document analyst. Answer the question using ONLY the provided context.\n\
         Do not add information not present in the context. Do not mention or cite source labels in your answer.\n\n\
         {type_instruction}\n\n\
         After your answer, on a new line, write SOURCES: followed by the comma-separated excerpt numbers you used.\n\
         Example: SOURCES: 1, 3, 7\n\
         If unanswerable, write SOURCES: none\n\n\
         <context>\n{context}\n</context>"
    )
}

/// Extract the cited source excerpt numbers from the LLM response.
/// Returns the set of 0-based indices (excerpt numbers are 1-based in the prompt).
pub fn extract_cited_sources(raw: &str) -> Option<Vec<usize>> {
    // Look for "SOURCES:" line
    for line in raw.lines().rev() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("SOURCES:").or_else(|| line.strip_prefix("Sources:")) {
            let rest = rest.trim();
            if rest.eq_ignore_ascii_case("none") || rest.is_empty() {
                return Some(vec![]);
            }
            let indices: Vec<usize> = rest
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .filter(|&n| n >= 1)
                .map(|n| n - 1) // convert to 0-based
                .collect();
            if !indices.is_empty() {
                return Some(indices);
            }
        }
    }
    None
}

/// Strip the SOURCES: line from the answer text before parsing.
pub fn strip_sources_line(raw: &str) -> String {
    let mut lines: Vec<&str> = raw.lines().collect();
    // Remove trailing SOURCES line(s)
    while let Some(last) = lines.last() {
        let trimmed = last.trim();
        if trimmed.starts_with("SOURCES:") || trimmed.starts_with("Sources:") || trimmed.is_empty() {
            lines.pop();
        } else {
            break;
        }
    }
    lines.join("\n")
}

/// Parse the raw LLM output into the correct JSON value based on answer_type.
pub fn parse_answer(raw: &str, answer_type: &str) -> Value {
    let text = raw.trim();

    // Check for unanswerable marker
    if text.eq_ignore_ascii_case("unanswerable")
        || text.contains("UNANSWERABLE")
        || (answer_type != "free_text"
            && (text.to_lowercase().contains("cannot be found")
                || text.to_lowercase().contains("not found in")
                || text.to_lowercase().contains("no information")))
    {
        if answer_type == "free_text" {
            // For free_text, return a natural language statement
            if text.contains("no information") || text.contains("cannot be found") {
                return Value::String(text.to_string());
            }
            return Value::String(
                "There is no information on this question in the provided documents.".to_string(),
            );
        }
        return Value::Null;
    }

    match answer_type {
        "number" => parse_number(text),
        "boolean" => parse_boolean(text),
        "name" => Value::String(parse_name(text)),
        "names" => parse_names(text),
        "date" => Value::String(text.to_string()),
        "free_text" => {
            // Truncate to 280 chars
            let truncated: String = text.chars().take(280).collect();
            Value::String(truncated)
        }
        _ => Value::String(text.to_string()),
    }
}

fn parse_number(text: &str) -> Value {
    // Strip common formatting: commas, currency symbols, percent signs, whitespace
    let cleaned: String = text
        .replace(',', "")
        .replace('$', "")
        .replace('%', "")
        .replace('\u{a0}', "") // non-breaking space
        .trim()
        .to_string();

    // Try to extract the first number-like substring
    let num_str = extract_number_str(&cleaned);

    if let Ok(n) = num_str.parse::<f64>() {
        // Return as integer if it's a whole number
        if n.fract() == 0.0 && n.abs() < i64::MAX as f64 {
            Value::Number(serde_json::Number::from(n as i64))
        } else {
            serde_json::Number::from_f64(n)
                .map(Value::Number)
                .unwrap_or(Value::String(text.to_string()))
        }
    } else {
        Value::String(text.to_string())
    }
}

fn extract_number_str(text: &str) -> String {
    let mut result = String::new();
    let mut found_digit = false;

    for ch in text.chars() {
        if ch.is_ascii_digit() || ch == '.' || (ch == '-' && !found_digit) {
            result.push(ch);
            if ch.is_ascii_digit() {
                found_digit = true;
            }
        } else if found_digit {
            break;
        }
    }

    result
}

/// Parse a name answer — if the LLM returned a full sentence instead of just the name,
/// try to extract just the entity (e.g. case number, person name, etc.)
fn parse_name(text: &str) -> String {
    let trimmed = text.trim().trim_matches('"');

    // If it looks like a short, clean answer (no sentence structure), use as-is
    if !trimmed.contains(". ") && !trimmed.contains(" is ") && !trimmed.contains(" has ")
        && !trimmed.contains(" was ") && !trimmed.contains(" are ")
        && trimmed.len() < 100
    {
        return trimmed.to_string();
    }

    // Try to extract a case number pattern like "CFI 010/2024", "SCT 295/2025", "ARB 034/2025", "CA 004/2025"
    if let Some(case_num) = extract_case_number(trimmed) {
        return case_num;
    }

    // Try to extract a quoted entity
    if let Some(start) = trimmed.find('"') {
        if let Some(end) = trimmed[start + 1..].find('"') {
            let quoted = &trimmed[start + 1..start + 1 + end];
            if !quoted.is_empty() {
                return quoted.to_string();
            }
        }
    }

    // Fallback: take the first sentence/clause
    if let Some(period_pos) = trimmed.find(". ") {
        return trimmed[..period_pos].to_string();
    }

    trimmed.to_string()
}

/// Extract a case number pattern like "CFI 010/2024" or "CA 004/2025" from text.
fn extract_case_number(text: &str) -> Option<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    for i in 0..words.len().saturating_sub(1) {
        let w = words[i];
        // Check if this word is 2-4 uppercase letters
        if w.len() >= 2 && w.len() <= 4 && w.chars().all(|c| c.is_ascii_uppercase()) {
            // Check if next word matches digits/digits pattern
            let next = words[i + 1];
            // Remove trailing punctuation
            let next_clean = next.trim_end_matches(|c: char| c == '.' || c == ',' || c == ';');
            if next_clean.contains('/') {
                let parts: Vec<&str> = next_clean.split('/').collect();
                if parts.len() == 2
                    && parts[0].chars().all(|c| c.is_ascii_digit())
                    && parts[1].chars().all(|c| c.is_ascii_digit())
                {
                    return Some(format!("{} {}", w, next_clean));
                }
            }
        }
    }
    None
}

fn parse_boolean(text: &str) -> Value {
    let lower = text.to_lowercase();
    // Extract just the first word for matching
    let first_word = lower.split_whitespace().next().unwrap_or("");
    if first_word == "true" || first_word == "yes" || first_word == "1" {
        Value::Bool(true)
    } else if first_word == "false" || first_word == "no" || first_word == "0" {
        Value::Bool(false)
    } else if lower.contains("true") {
        Value::Bool(true)
    } else if lower.contains("false") {
        Value::Bool(false)
    } else {
        Value::String(text.to_string())
    }
}

fn parse_names(text: &str) -> Value {
    let names: Vec<Value> = text
        .split(';')
        .flat_map(|part| {
            if part.contains(',') && !part.contains(' ') {
                // "a,b,c" without spaces — likely comma-separated
                part.split(',')
                    .map(|s| s.trim().to_string())
                    .collect::<Vec<_>>()
            } else {
                vec![part.trim().to_string()]
            }
        })
        .filter(|s| !s.is_empty())
        .map(Value::String)
        .collect();

    if names.is_empty() {
        Value::Array(vec![Value::String(text.to_string())])
    } else {
        Value::Array(names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_number() {
        assert_eq!(parse_answer("42", "number"), Value::Number(42.into()));
        assert_eq!(parse_answer("3.14", "number"), serde_json::json!(3.14));
        assert_eq!(
            parse_answer("$1,000,000", "number"),
            Value::Number(1000000.into())
        );
        assert_eq!(parse_answer("UNANSWERABLE", "number"), Value::Null);
    }

    #[test]
    fn test_parse_boolean() {
        assert_eq!(parse_answer("true", "boolean"), Value::Bool(true));
        assert_eq!(parse_answer("False", "boolean"), Value::Bool(false));
        assert_eq!(parse_answer("yes", "boolean"), Value::Bool(true));
        assert_eq!(parse_answer("UNANSWERABLE", "boolean"), Value::Null);
    }

    #[test]
    fn test_parse_names() {
        let result = parse_answer("Alice Smith; Bob Jones", "names");
        assert_eq!(
            result,
            serde_json::json!(["Alice Smith", "Bob Jones"])
        );
    }

    #[test]
    fn test_parse_free_text_truncation() {
        let long_text = "a".repeat(500);
        if let Value::String(s) = parse_answer(&long_text, "free_text") {
            assert_eq!(s.len(), 280);
        } else {
            panic!("Expected string");
        }
    }
}
