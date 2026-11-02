import hashlib
import fitz  # PyMuPDF
from docx import Document
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParsedDocument:
    source: str
    text: str
    metadata: dict


class DocumentParser:
    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        if path.suffix.lower() == '.pdf':
            return self._parse_pdf(path)
        elif path.suffix.lower() in ('.docx', '.doc'):
            return self._parse_docx(path)
        raise ValueError(f'Unsupported format: {path.suffix}')

    @staticmethod
    def _document_id_for_file(path: Path) -> str:
        """Stable id derived from the file's raw bytes (not just its name).

        Content-based so re-uploading an edited file under the same
        filename gets a new document_id, while re-uploading the exact same
        bytes is idempotent (same document_id, same downstream chunk_ids)
        instead of silently creating a duplicate document.
        """
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(65536), b''):
                h.update(block)
        return h.hexdigest()[:16]

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            text = page.get_text('text')
            if text.strip():
                pages.append(text)
        full_text = '\n\n'.join(pages)
        return ParsedDocument(
            source=str(path),
            text=full_text,
            metadata={
                'pages': len(doc),
                'filename': path.name,
                'type': 'pdf',
                'document_id': self._document_id_for_file(path),
            }
        )

    def _parse_docx(self, path: Path) -> ParsedDocument:
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return ParsedDocument(
            source=str(path),
            text='\n\n'.join(paragraphs),
            metadata={
                'paragraphs': len(paragraphs),
                'filename': path.name,
                'type': 'docx',
                'document_id': self._document_id_for_file(path),
            }
        )