from pathlib import Path

import pymupdf4llm
from docling.document_converter import DocumentConverter

# Using docling
converter = DocumentConverter()
# result = converter.convert("./2408.09869v5.pdf")
result = converter.convert("https://arxiv.org/pdf/2408.09869")

document = result.document
markdown_output = document.export_to_markdown()
print(markdown_output)

# Converting to HTML
result = converter.convert("https://docling-project.github.io/docling/")
document = result.document
markdown_output = document.export_to_markdown()
print(markdown_output)

# Using PyMuPDF4llm
pdf_path = Path(__file__).parent / "2408.09869v5.pdf"
markdown_output = pymupdf4llm.to_markdown(str(pdf_path))
print(markdown_output)
