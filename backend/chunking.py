import re


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")

    normalized = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        value = normalized[start:end].strip()
        if value:
            chunks.append(value)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_pages(
    pages: list[dict], chunk_size: int = 900, overlap: int = 150
) -> list[dict]:
    chunks = []
    for page in pages:
        for text in split_text(page["text"], chunk_size, overlap):
            chunks.append(
                {
                    "page_number": page["page_number"],
                    "text": text,
                    "chunk_index": len(chunks),
                }
            )
    return chunks
