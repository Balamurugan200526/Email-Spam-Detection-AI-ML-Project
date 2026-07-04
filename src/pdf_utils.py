"""
pdf_utils.py
--------------
Utility for extracting text from an uploaded PDF (e.g. an email saved/exported
as a PDF) so it can be run through the same cleaning + prediction pipeline as
plain-text email input.
"""

from pypdf import PdfReader


def extract_text_from_pdf(file_obj) -> str:
    """
    Extract all text from a PDF file.
    """
    reader = PdfReader(file_obj)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text).strip()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            print(extract_text_from_pdf(f))
    else:
        print("Usage: python pdf_utils.py <path_to_pdf>")