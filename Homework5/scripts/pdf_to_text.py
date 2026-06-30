from pathlib import Path

from pypdf import PdfReader


PDF_PATH = Path("data/LLMAll_en-US_FINAL.pdf")
OUTPUT_PATH = Path("data/source.md")


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))

    pages_text: list[str] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        page_block = f"""
<!-- page: {page_index} -->

{text}
""".strip()

        pages_text.append(page_block)

    return "\n\n".join(pages_text)


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    text = extract_pdf_text(PDF_PATH)

    OUTPUT_PATH.write_text(text, encoding="utf-8")

    print(f"Extracted text from: {PDF_PATH}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Characters: {len(text):,}")


if __name__ == "__main__":
    main()