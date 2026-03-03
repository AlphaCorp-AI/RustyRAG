const RAG_SYSTEM_TEMPLATE: &str = include_str!("rag_system_prompt.txt");
const CONTEXTUAL_TEMPLATE: &str = include_str!("contextual_prompt.txt");

pub fn build_rag_system_prompt(context: &str) -> String {
    RAG_SYSTEM_TEMPLATE.replace("{context}", context)
}

pub fn build_contextual_prompt(document_text: &str, chunk_text: &str) -> String {
    CONTEXTUAL_TEMPLATE
        .replace("{document_text}", document_text)
        .replace("{chunk_text}", chunk_text)
}
