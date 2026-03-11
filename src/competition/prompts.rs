use serde_json::Value;

/// System prompt for competition RAG answering.
/// Includes answer-type-specific instructions and null detection.
/// For free_text, targets all 5 LLM judge criteria:
///   1. Correctness  2. Completeness  3. Grounding  4. Confidence calibration  5. Clarity
pub fn build_competition_system_prompt(context: &str, answer_type: &str) -> String {
    let type_instruction = match answer_type {
        "number" => {
            "Return ONLY the numeric value (integer or decimal). No units, no explanation, no text.\n\
             Examples: 42, 3.14, 1000000\n\
             If the information cannot be found in the context, respond with exactly: UNANSWERABLE"
        }
        "boolean" => {
            "Return ONLY the word 'true' or 'false'. No explanation.\n\
             If the information cannot be found in the context, respond with exactly: UNANSWERABLE"
        }
        "name" => {
            "Return ONLY the exact name or entity as it appears in the documents. No explanation.\n\
             If the information cannot be found in the context, respond with exactly: UNANSWERABLE"
        }
        "names" => {
            "Return ONLY a semicolon-separated list of names exactly as they appear in the documents.\n\
             No explanation, no numbering, no extra text.\n\
             Example format: Alice Smith; Bob Jones; Carol White\n\
             If the information cannot be found in the context, respond with exactly: UNANSWERABLE"
        }
        "date" => {
            "Return ONLY the date in YYYY-MM-DD format. No explanation.\n\
             Example: 2024-03-15\n\
             If the information cannot be found in the context, respond with exactly: UNANSWERABLE"
        }
        "free_text" => {
            "Provide a clear, concise answer in 1-3 sentences (max 280 characters total).\n\
             Rules:\n\
             - Every claim must be directly supported by the provided context.\n\
             - Address all aspects of the question completely.\n\
             - Do not state anything not present in the context.\n\
             - If the answer is uncertain or partial, explicitly say so.\n\
             - If the information is not in the context, respond exactly with: \
             \"There is no information on this question in the provided documents.\""
        }
        _ => {
            "Answer based on the context. If the information cannot be found, respond with exactly: UNANSWERABLE"
        }
    };

    format!(
        "You are a precise legal document analyst. Answer the question using ONLY the provided context.\n\
         Do not add information not present in the context. Do not mention or cite source labels.\n\n\
         {type_instruction}\n\n\
         <context>\n{context}\n</context>"
    )
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
        "name" => Value::String(text.to_string()),
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
