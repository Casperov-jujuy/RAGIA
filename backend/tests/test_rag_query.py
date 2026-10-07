import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app


class TestRAGQueryEndpoint(unittest.TestCase):
    """Suite de pruebas para el endpoint POST /api/v1/rag/query (Paso 4.1: Vectorizar pregunta)."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.api.v1.endpoints_rag.GeminiEmbeddingService.embed_text")
    def test_query_vectorization_success(self, mock_embed_text):
        """Verifica la vectorización exitosa de una pregunta válida."""
        mock_embed_text.return_value = [0.123] * 768

        payload = {"question": "¿Cómo se estructuran los documentos en RAGIA?"}
        response = self.client.post("/api/v1/rag/query", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["question"], payload["question"])
        self.assertEqual(data["embedding_dimension"], 768)
        self.assertEqual(len(data["embedding_preview"]), 5)
        self.assertEqual(data["embedding_preview"][0], 0.123)
        self.assertIn("vectorizada exitosamente", data["message"])
        mock_embed_text.assert_called_once_with(payload["question"])

    def test_query_empty_question_rejected(self):
        """Verifica que una pregunta vacía o sólo con espacios sea rechazada."""
        payload = {"question": "   "}
        response = self.client.post("/api/v1/rag/query", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("no puede estar vacía", response.json()["detail"])

    def test_query_too_short_rejected(self):
        """Verifica que preguntas con menos de 2 caracteres fallen por validación Pydantic (422)."""
        payload = {"question": "a"}
        response = self.client.post("/api/v1/rag/query", json=payload)
        self.assertEqual(response.status_code, 422)

    @patch("app.api.v1.endpoints_rag.GeminiEmbeddingService.embed_text")
    def test_query_embedding_failure_returns_500(self, mock_embed_text):
        """Verifica que un fallo en la generación del vector retorne HTTP 500."""
        mock_embed_text.return_value = []

        payload = {"question": "¿Pregunta que falla en vectorización?"}
        response = self.client.post("/api/v1/rag/query", json=payload)

        self.assertEqual(response.status_code, 500)
        self.assertIn("No se pudo generar", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
