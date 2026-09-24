import io
import unittest
from fastapi.testclient import TestClient

from app.main import app


class TestDocumentUploadEndpoint(unittest.TestCase):
    """Pruebas de integración para el endpoint POST /api/v1/documents/upload."""

    def setUp(self):
        self.client = TestClient(app)

    def test_upload_valid_markdown_success(self):
        """Verifica que subir un archivo .md retorne HTTP 200."""
        file_content = b"# Titulo\nEste es un archivo markdown de prueba."
        files = {"file": ("notas.md", io.BytesIO(file_content), "text/markdown")}
        response = self.client.post("/api/v1/documents/upload", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "chunked")
        self.assertEqual(data["extension"], ".md")
        self.assertEqual(data["filename"], "notas.md")
        self.assertEqual(data["total_sections"], 1)
        self.assertEqual(data["total_chunks"], 1)
        self.assertEqual(data["sections_summary"][0]["title"], "Titulo")

    def test_upload_valid_pdf_success(self):
        """Verifica que subir un archivo .pdf válido retorne HTTP 200 con status chunked."""
        from tests.test_pdf_parser import create_test_pdf_bytes
        file_content = create_test_pdf_bytes("Contenido de prueba en PDF para endpoint.")
        files = {"file": ("documento.pdf", io.BytesIO(file_content), "application/pdf")}
        response = self.client.post("/api/v1/documents/upload", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "chunked")
        self.assertEqual(data["extension"], ".pdf")
        self.assertEqual(data["filename"], "documento.pdf")
        self.assertEqual(data["total_sections"], 1)
        self.assertEqual(data["total_chunks"], 1)
        self.assertEqual(data["sections_summary"][0]["title"], "Página 1")

    def test_upload_corrupt_pdf_rejected(self):
        """Verifica que subir un archivo .pdf corrupto retorne HTTP 400 Bad Request."""
        file_content = b"%PDF-1.4 corrupt invalid content"
        files = {"file": ("danado.pdf", io.BytesIO(file_content), "application/pdf")}
        response = self.client.post("/api/v1/documents/upload", files=files)

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("corrupto", data["detail"])

    def test_upload_invalid_image_rejected(self):
        """Verifica que subir una imagen (.png) retorne HTTP 400 Bad Request."""
        file_content = b"\x89PNG\r\n\x1a\n dummy image"
        files = {"file": ("captura.png", io.BytesIO(file_content), "image/png")}
        response = self.client.post("/api/v1/documents/upload", files=files)

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Formato de archivo no permitido", data["detail"])

    def test_upload_invalid_docx_rejected(self):
        """Verifica que subir un archivo Word (.docx) retorne HTTP 400 Bad Request."""
        file_content = b"PK dummy word doc"
        files = {"file": ("reporte.docx", io.BytesIO(file_content), "application/vnd.openxmlformats")}
        response = self.client.post("/api/v1/documents/upload", files=files)

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Formato de archivo no permitido", data["detail"])


if __name__ == "__main__":
    unittest.main()
