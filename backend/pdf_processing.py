from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFProcessingError(ValueError):
    pass


def extract_pdf_pages(content: bytes) -> list[dict]:
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise PDFProcessingError("Encrypted PDFs are not supported.")
        pages = [
            {"page_number": page_number, "text": page.extract_text() or ""}
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    except (PdfReadError, EOFError, OSError) as error:
        raise PDFProcessingError("The PDF could not be read. Check that it is not damaged.") from error
    except ValueError as error:
        if isinstance(error, PDFProcessingError):
            raise
        raise PDFProcessingError(
            "The PDF could not be read. Check that it is not damaged."
        ) from error

    if not any(page["text"].strip() for page in pages):
        raise PDFProcessingError(
            "No searchable text was found. Scanned PDFs need OCR before upload."
        )
    return pages
