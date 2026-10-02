import unittest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

from app.services.embedding_service import GeminiEmbeddingService
from app.models.schemas import DocumentChunk


class TestGeminiEmbeddingService(unittest.TestCase):
    """Suite de pruebas unitarias para el servicio GeminiEmbeddingService."""

    def test_missing_api_key_raises_http_500(self):
        """Verifica que si la clave está vacía o es placeholder, se lance HTTP 500 con mensaje claro."""
        service = GeminiEmbeddingService(api_key="")
        with self.assertRaises(HTTPException) as ctx:
            _ = service.client
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("GEMINI_API_KEY no está configurada", ctx.exception.detail)

    def test_placeholder_api_key_raises_http_500(self):
        """Verifica que el placeholder de plantilla sea rechazado."""
        service = GeminiEmbeddingService(api_key="tu_api_key_de_gemini_aqui")
        with self.assertRaises(HTTPException) as ctx:
            _ = service.client
        self.assertEqual(ctx.exception.status_code, 500)

    @patch("app.services.embedding_service.genai.Client")
    def test_embed_text_success(self, mock_client_cls):
        """Verifica la generación de embedding para un texto individual."""
        mock_embedding = MagicMock()
        mock_embedding.values = [0.123] * 768
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_instance = MagicMock()
        mock_instance.models.embed_content.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        service = GeminiEmbeddingService(api_key="test_key_abc_123")
        vector = service.embed_text("Texto de prueba")

        self.assertEqual(len(vector), 768)
        self.assertEqual(vector[0], 0.123)
        mock_instance.models.embed_content.assert_called_once_with(
            model="gemini-embedding-001",
            contents="Texto de prueba"
        )

    @patch("app.services.embedding_service.genai.Client")
    def test_embed_chunks_batching(self, mock_client_cls):
        """Verifica que los chunks sean procesados y enriquecidos con su vector numérico."""
        # Simulamos respuesta para 3 chunks
        mock_embs = []
        for i in range(3):
            m = MagicMock()
            m.values = [float(i)] * 768
            mock_embs.append(m)

        mock_response = MagicMock()
        mock_response.embeddings = mock_embs

        mock_instance = MagicMock()
        mock_instance.models.embed_content.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        service = GeminiEmbeddingService(api_key="test_key_abc_123")

        chunks = [
            DocumentChunk(chunk_id="chunk_0", content="Texto cero", metadata={}),
            DocumentChunk(chunk_id="chunk_1", content="Texto uno", metadata={}),
            DocumentChunk(chunk_id="chunk_2", content="Texto dos", metadata={}),
        ]

        result = service.embed_chunks(chunks, batch_size=2)

        self.assertEqual(len(result), 3)
        self.assertIsNotNone(result[0].embedding)
        self.assertEqual(len(result[0].embedding), 768)
        self.assertEqual(result[0].embedding[0], 0.0)
        self.assertEqual(result[1].embedding[0], 1.0)

    def test_embed_empty_text(self):
        """Verifica que textos vacíos retornen lista vacía sin llamar a la API."""
        service = GeminiEmbeddingService(api_key="test_key")
        self.assertEqual(service.embed_text(""), [])
        self.assertEqual(service.embed_text("   "), [])

    def test_embed_empty_chunks(self):
        """Verifica que lista vacía de chunks retorne lista vacía."""
        service = GeminiEmbeddingService(api_key="test_key")
        self.assertEqual(service.embed_chunks([]), [])


if __name__ == "__main__":
    unittest.main()
