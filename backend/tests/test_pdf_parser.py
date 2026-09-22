import io
import unittest
import pypdf
from fastapi import HTTPException

from app.ingestion.pdf_parser import PDFParser


def create_test_pdf_bytes(text: str = "Texto de prueba para RAGIA") -> bytes:
    """Genera en memoria un PDF válido con una página que contiene el texto especificado."""
    stream_content = f"BT /F1 14 Tf 50 700 Td ({text}) Tj ET"
    length = len(stream_content.encode("latin-1"))

    template = (
        "%PDF-1.4\n"
        "1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        "2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        "3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>> endobj\n"
        f"4 0 obj <</Length {length}>> stream\n"
        f"{stream_content}\n"
        "endstream\nendobj\n"
        "5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
        "xref\n0 6\n"
        "0000000000 65535 f \n"
        "0000000010 00000 n \n"
        "0000000060 00000 n \n"
        "0000000117 00000 n \n"
        "0000000247 00000 n \n"
        "0000000350 00000 n \n"
        "trailer <</Size 6 /Root 1 0 R>>\n"
        "startxref\n450\n%%EOF"
    )
    return template.encode("latin-1")


class TestPDFParser(unittest.TestCase):
    """Suite de pruebas unitarias para PDFParser."""

    def test_parse_valid_pdf(self):
        """Verifica la extracción exitosa de texto y metadatos de página."""
        pdf_bytes = create_test_pdf_bytes("Contenido de prueba en pagina uno")
        sections = PDFParser.parse_bytes(pdf_bytes, filename="informe.pdf")

        self.assertEqual(len(sections), 1)
        section = sections[0]
        self.assertEqual(section.title, "Página 1")
        self.assertEqual(section.level, 1)
        self.assertIn("Contenido de prueba en pagina uno", section.content)
        self.assertEqual(section.metadata["source"], "informe.pdf")
        self.assertEqual(section.metadata["page"], 1)
        self.assertEqual(section.metadata["total_pages"], 1)
        self.assertGreater(section.metadata["char_count"], 0)

    def test_clean_extracted_text(self):
        """Verifica la limpieza de texto: caracteres nulos, espacios y palabras cortadas con guion."""
        raw = "Este es un docu-\nmento con texto   espaciado.\x00\n\n\n\nSiguiente parrafo."
        cleaned = PDFParser.clean_extracted_text(raw)

        # Debe unir 'docu-\nmento' en 'documento'
        self.assertIn("documento", cleaned)
        # No debe contener caracteres nulos
        self.assertNotIn("\x00", cleaned)
        # No debe tener más de 2 saltos de línea consecutivos
        self.assertNotIn("\n\n\n", cleaned)

    def test_skip_blank_pages(self):
        """Verifica que las páginas vacías sin texto sean omitidas."""
        # Creamos un PDF con una página en blanco usando PdfWriter
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=300, height=300)
        buf = io.BytesIO()
        writer.write(buf)
        blank_pdf_bytes = buf.getvalue()

        sections = PDFParser.parse_bytes(blank_pdf_bytes, filename="vacio.pdf")
        self.assertEqual(len(sections), 0)

    def test_corrupt_pdf_raises_http_400(self):
        """Verifica que un archivo binario corrupto que dice ser PDF sea rechazado con HTTP 400."""
        corrupt_bytes = b"%PDF-1.4 corrupt content that is not a real pdf structure"
        with self.assertRaises(HTTPException) as ctx:
            PDFParser.parse_bytes(corrupt_bytes, filename="danado.pdf")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("corrupto", ctx.exception.detail)

    def test_empty_bytes_returns_empty_list(self):
        """Verifica que bytes vacíos retornen una lista vacía."""
        self.assertEqual(PDFParser.parse_bytes(b""), [])


if __name__ == "__main__":
    unittest.main()
