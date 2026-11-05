import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.services.prompt_service import NO_INFO_FALLBACK


class TestRAGQueryEndpoint(unittest.TestCase):
    """Suite de pruebas para los endpoints RAG (/query, /retrieve, /prompt-preview)."""

    def setUp(self):
        self.client = TestClient(app)

    @patch("app.api.v1.endpoints_rag.GeminiGenerationService.generate_response")
    @patch("app.api.v1.endpoints_rag.ChromaVectorStore.similarity_search")
    @patch("app.api.v1.endpoints_rag.GeminiEmbeddingService.embed_text")
    def test_query_rag_full_success(self, mock_embed_text, mock_sim_search, mock_generate):
        """Verifica el flujo RAG completo en POST /api/v1/rag/query (Paso 4.4: Generación con Gemini Flash)."""
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
        mock_generate.return_value = (
            "Para instalar el sistema sigue las instrucciones [Fuente: manual.pdf, Página 2].",
            "gemini-2.0-flash"
        )

        payload = {"question": "¿Cómo se instala el sistema?", "top_k": 2}
        response = self.client.post("/api/v1/rag/query", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["question"], payload["question"])
        self.assertEqual(data["total_sources"], 2)
        self.assertEqual(len(data["retrieved_chunks"]), 2)
        self.assertEqual(data["model_used"], "gemini-2.0-flash")
        self.assertIn("Para instalar el sistema", data["answer"])
        self.assertIn("[Fuente: manual.pdf, Página 2]", data["answer"])

        mock_embed_text.assert_called_once_with(payload["question"])
        mock_sim_search.assert_called_once_with(query_vector=[0.123] * 768, top_k=2)
        mock_generate.assert_called_once()

    @patch("app.api.v1.endpoints_rag.ChromaVectorStore.similarity_search")
    @patch("app.api.v1.endpoints_rag.GeminiEmbeddingService.embed_text")
    def test_query_rag_empty_store_returns_direct_fallback(self, mock_embed_text, mock_sim_search):
        """Verifica que si no hay fragmentos indexados, se responda de inmediato con la directiva anti-alucinaciones sin llamar al LLM."""
        mock_embed_text.return_value = [0.123] * 768
        mock_sim_search.return_value = []

        payload = {"question": "¿Qué dice el documento?"}
        response = self.client.post("/api/v1/rag/query", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_sources"], 0)
        self.assertEqual(data["retrieved_chunks"], [])
        self.assertEqual(data["answer"], NO_INFO_FALLBACK)
        self.assertEqual(data["model_used"], "contencion-directa")

    @patch("app.api.v1.endpoints_rag.ChromaVectorStore.similarity_search")
    @patch("app.api.v1.endpoints_rag.GeminiEmbeddingService.embed_text")
    def test_retrieve_endpoint_success(self, mock_embed_text, mock_sim_search):
        """Verifica la recuperación semántica pura en POST /api/v1/rag/retrieve."""
        mock_embed_text.return_value = [0.123] * 768
        mock_sim_search.return_value = [
            {
                "chunk_id": "manual.pdf_chunk_1",
                "content": "Contenido relevante.",
                "metadata": {"source": "manual.pdf", "page": 1},
                "similarity_score": 0.91,
            }
        ]

        response = self.client.post("/api/v1/rag/retrieve", json={"question": "¿Pregunta?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_retrieved"], 1)
        self.assertEqual(data["retrieved_chunks"][0]["source"], "manual.pdf")

    @patch("app.api.v1.endpoints_rag.ChromaVectorStore.similarity_search")
    @patch("app.api.v1.endpoints_rag.GeminiEmbeddingService.embed_text")
    def test_retrieve_endpoint_empty_store(self, mock_embed_text, mock_sim_search):
        """Verifica la respuesta de POST /api/v1/rag/retrieve cuando no hay datos."""
        mock_embed_text.return_value = [0.123] * 768
        mock_sim_search.return_value = []

        response = self.client.post("/api/v1/rag/retrieve", json={"question": "¿Pregunta?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_retrieved"], 0)
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

    @patch("app.api.v1.endpoints_rag.ChromaVectorStore.similarity_search")
    @patch("app.api.v1.endpoints_rag.GeminiEmbeddingService.embed_text")
    def test_prompt_preview_success(self, mock_embed_text, mock_sim_search):
        """Verifica la generación del prompt con contención y citas en /prompt-preview."""
        mock_embed_text.return_value = [0.123] * 768
        mock_sim_search.return_value = [
            {
                "chunk_id": "guia.md_chunk_0",
                "content": "RAGIA soporta archivos Markdown y PDF únicamente.",
                "metadata": {"source": "guia.md", "section_title": "Formatos"},
                "similarity_score": 0.9412,
            }
        ]

        payload = {"question": "¿Qué formatos soporta?", "top_k": 1}
        response = self.client.post("/api/v1/rag/prompt-preview", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["question"], "¿Qué formatos soporta?")
        self.assertIn("CONTENCIÓN ESTRICTA", data["system_instruction"])
        self.assertIn("CITACIÓN OBLIGATORIA", data["system_instruction"])
        self.assertIn("RAGIA soporta archivos Markdown y PDF únicamente.", data["formatted_prompt"])
        self.assertEqual(data["total_sources"], 1)
        self.assertEqual(data["sources"][0]["source"], "guia.md")
        self.assertEqual(data["sources"][0]["citation_tag"], "[Fuente: guia.md, Sección: Formatos]")

    def test_prompt_preview_empty_question_rejected(self):
        """Verifica que una pregunta vacía en /prompt-preview sea rechazada con 400."""
        response = self.client.post("/api/v1/rag/prompt-preview", json={"question": "   "})
        self.assertEqual(response.status_code, 400)

    @patch("app.api.v1.endpoints_rag.ChromaVectorStore.similarity_search")
    @patch("app.api.v1.endpoints_rag.GeminiEmbeddingService.embed_text")
    def test_prompt_preview_empty_store(self, mock_embed_text, mock_sim_search):
        """Verifica la respuesta de /prompt-preview cuando no hay documentos indexados."""
        mock_embed_text.return_value = [0.123] * 768
        mock_sim_search.return_value = []

        response = self.client.post("/api/v1/rag/prompt-preview", json={"question": "¿Hay algo?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_sources"], 0)
        self.assertEqual(data["sources"], [])
        self.assertIn("No se encontraron fragmentos relevantes", data["formatted_prompt"])


if __name__ == "__main__":
    unittest.main()
