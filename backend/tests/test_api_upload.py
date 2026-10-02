import io
import unittest
from fastapi.testclient import TestClient

from unittest.mock import patch
from app.main import app


class TestDocumentUploadEndpoint(unittest.TestCase):
    """Pruebas de integración para el endpoint POST /api/v1/documents/upload."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.api.v1.endpoints_documents.GeminiEmbeddingService.embed_chunks")
    def test_upload_valid_markdown_success(self, mock_embed):
        """Verifica que subir un archivo .md retorne HTTP 200 con embeddings."""
        def fake_embed(chunks):
            for c in chunks:
                c.embedding = [0.1] * 768
            return chunks
        mock_embed.side_effect = fake_embed

        file_content = b"# Titulo\nEste es un archivo markdown de prueba."
        files = {"file": ("notas.md", io.BytesIO(file_content), "text/markdown")}
        response = self.client.post("/api/v1/documents/upload", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "embedded")
        self.assertEqual(data["extension"], ".md")
        self.assertEqual(data["filename"], "notas.md")
        self.assertEqual(data["total_sections"], 1)
        self.assertEqual(data["total_chunks"], 1)
        self.assertEqual(data["embedding_dimension"], 768)
        self.assertEqual(data["sections_summary"][0]["title"], "Titulo")

    @patch("app.api.v1.endpoints_documents.GeminiEmbeddingService.embed_chunks")
    def test_upload_valid_pdf_success(self, mock_embed):
        """Verifica que subir un archivo .pdf válido retorne HTTP 200 con status embedded."""
        def fake_embed(chunks):
            for c in chunks:
                c.embedding = [0.1] * 768
            return chunks
        mock_embed.side_effect = fake_embed

        from tests.test_pdf_parser import create_test_pdf_bytes
        file_content = create_test_pdf_bytes("Contenido de prueba en PDF para endpoint.")
        files = {"file": ("documento.pdf", io.BytesIO(file_content), "application/pdf")}
        response = self.client.post("/api/v1/documents/upload", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "embedded")
        self.assertEqual(data["extension"], ".pdf")
        self.assertEqual(data["filename"], "documento.pdf")
        self.assertEqual(data["total_sections"], 1)
        self.assertEqual(data["total_chunks"], 1)
        self.assertEqual(data["embedding_dimension"], 768)
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
