import io
import unittest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app


class TestDocumentUploadEndpoint(unittest.TestCase):
    """Pruebas de integración para los endpoints de documentos (/upload, /documents)."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.api.v1.endpoints_documents.ChromaVectorStore.add_chunks")
    @patch("app.api.v1.endpoints_documents.GeminiEmbeddingService.embed_chunks")
    def test_upload_valid_markdown_success(self, mock_embed, mock_add_chunks):
        """Verifica que subir un archivo .md retorne HTTP 200 con status indexed."""
        def fake_embed(chunks):
            for c in chunks:
                c.embedding = [0.1] * 768
            return chunks
        mock_embed.side_effect = fake_embed
        mock_add_chunks.return_value = 1

        file_content = b"# Titulo\nEste es un archivo markdown de prueba."
        files = {"file": ("notas.md", io.BytesIO(file_content), "text/markdown")}
        response = self.client.post("/api/v1/documents/upload", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "indexed")
        self.assertEqual(data["extension"], ".md")
        self.assertEqual(data["filename"], "notas.md")
        self.assertEqual(data["total_sections"], 1)
        self.assertEqual(data["total_chunks"], 1)
        self.assertEqual(data["embedding_dimension"], 768)
        self.assertEqual(data["sections_summary"][0]["title"], "Titulo")

    @patch("app.api.v1.endpoints_documents.ChromaVectorStore.add_chunks")
    @patch("app.api.v1.endpoints_documents.GeminiEmbeddingService.embed_chunks")
    def test_upload_valid_pdf_success(self, mock_embed, mock_add_chunks):
        """Verifica que subir un archivo .pdf válido retorne HTTP 200 con status indexed."""
        def fake_embed(chunks):
            for c in chunks:
                c.embedding = [0.1] * 768
            return chunks
        mock_embed.side_effect = fake_embed
        mock_add_chunks.return_value = 1

        from tests.test_pdf_parser import create_test_pdf_bytes
        file_content = create_test_pdf_bytes("Contenido de prueba en PDF para endpoint.")
        files = {"file": ("documento.pdf", io.BytesIO(file_content), "application/pdf")}
        response = self.client.post("/api/v1/documents/upload", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "indexed")
        self.assertEqual(data["extension"], ".pdf")
        self.assertEqual(data["filename"], "documento.pdf")
        self.assertEqual(data["total_sections"], 1)
        self.assertEqual(data["total_chunks"], 1)
        self.assertEqual(data["embedding_dimension"], 768)
        self.assertEqual(data["sections_summary"][0]["title"], "Página 1")

    @patch("app.api.v1.endpoints_documents.ChromaVectorStore.list_documents")
    def test_list_documents_endpoint(self, mock_list):
        """Verifica que GET /api/v1/documents devuelva la lista de documentos indexados."""
        mock_list.return_value = [
            {"filename": "guia.md", "total_chunks": 3},
            {"filename": "manual.pdf", "total_chunks": 5, "total_pages": 2},
        ]
        response = self.client.get("/api/v1/documents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["filename"], "guia.md")

    @patch("app.api.v1.endpoints_documents.ChromaVectorStore.delete_by_source")
    def test_delete_document_success(self, mock_delete):
        """Verifica que DELETE /api/v1/documents/{filename} elimine los fragmentos."""
        mock_delete.return_value = 4
        response = self.client.delete("/api/v1/documents/archivo_a_borrar.md")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["deleted_chunks"], 4)
        self.assertIn("eliminados", data["message"])

    @patch("app.api.v1.endpoints_documents.ChromaVectorStore.delete_by_source")
    def test_delete_document_not_found(self, mock_delete):
        """Verifica que eliminar un archivo inexistente devuelva HTTP 404."""
        mock_delete.return_value = 0
        response = self.client.delete("/api/v1/documents/inexistente.pdf")
        self.assertEqual(response.status_code, 404)
        self.assertIn("No se encontraron", response.json()["detail"])

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
