import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app


class TestRAGQueryEndpoint(unittest.TestCase):
    """Suite de pruebas para el endpoint POST /api/v1/rag/query (Paso 4.2: Retrieval en ChromaDB)."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.api.v1.endpoints_rag.ChromaVectorStore.similarity_search")
    @patch("app.api.v1.endpoints_rag.GeminiEmbeddingService.embed_text")
    def test_query_retrieval_success(self, mock_embed_text, mock_sim_search):
        """Verifica la recuperación exitosa de fragmentos relevantes desde ChromaDB."""
        mock_embed_text.return_value = [0.123] * 768
        mock_sim_search.return_value = [
            {
                "chunk_id": "manual.pdf_chunk_1",
                "content": "Instrucciones de instalación del sistema.",
                "metadata": {"source": "manual.pdf", "page": 2, "section_title": "Instalación"},
                "similarity_score": 0.9521,
            },
            {
                "chunk_id": "guia.md_chunk_0",
                "content": "Requisitos previos de entorno y dependencias.",
                "metadata": {"source": "guia.md", "section_title": "Requisitos"},
                "similarity_score": 0.8845,
            }
        ]

        payload = {"question": "¿Cómo se instala el sistema?", "top_k": 2}
        response = self.client.post("/api/v1/rag/query", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["question"], payload["question"])
        self.assertEqual(data["total_retrieved"], 2)
        self.assertEqual(len(data["retrieved_chunks"]), 2)

        first_chunk = data["retrieved_chunks"][0]
        self.assertEqual(first_chunk["chunk_id"], "manual.pdf_chunk_1")
        self.assertEqual(first_chunk["source"], "manual.pdf")
        self.assertEqual(first_chunk["page"], 2)
        self.assertEqual(first_chunk["similarity_score"], 0.9521)

        mock_embed_text.assert_called_once_with(payload["question"])
        mock_sim_search.assert_called_once_with(query_vector=[0.123] * 768, top_k=2)

    @patch("app.api.v1.endpoints_rag.ChromaVectorStore.similarity_search")
    @patch("app.api.v1.endpoints_rag.GeminiEmbeddingService.embed_text")
    def test_query_retrieval_empty_store(self, mock_embed_text, mock_sim_search):
        """Verifica la respuesta cuando no hay documentos indexados en ChromaDB."""
        mock_embed_text.return_value = [0.123] * 768
        mock_sim_search.return_value = []

        payload = {"question": "¿Qué dice el documento?"}
        response = self.client.post("/api/v1/rag/query", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_retrieved"], 0)
        self.assertEqual(data["retrieved_chunks"], [])
        self.assertIn("No se encontraron fragmentos relevantes", data["message"])

    def test_query_empty_question_rejected(self):
        """Verifica que una pregunta vacía o sólo con espacios sea rechazada con 400."""
        payload = {"question": "   "}
        response = self.client.post("/api/v1/rag/query", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("no puede estar vacía", response.json()["detail"])

    def test_query_too_short_rejected(self):
        """Verifica que preguntas con menos de 2 caracteres fallen por validación Pydantic (422)."""
        payload = {"question": "a"}
        response = self.client.post("/api/v1/rag/query", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_query_invalid_top_k_rejected(self):
        """Verifica que top_k menor a 1 o mayor a 20 sea rechazado con 422."""
        response_low = self.client.post("/api/v1/rag/query", json={"question": "Pregunta", "top_k": 0})
        self.assertEqual(response_low.status_code, 422)

        response_high = self.client.post("/api/v1/rag/query", json={"question": "Pregunta", "top_k": 25})
        self.assertEqual(response_high.status_code, 422)

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
