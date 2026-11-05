import unittest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from google.genai.errors import ClientError

from app.services.generation_service import GeminiGenerationService


class TestGeminiGenerationService(unittest.TestCase):
    """Suite de pruebas unitarias para GeminiGenerationService (Paso 4.4)."""

    def test_missing_api_key_raises_http_500(self):
        """Verifica que API key vacía lance HTTP 500 con mensaje explicativo."""
        service = GeminiGenerationService(api_key="")
        with self.assertRaises(HTTPException) as ctx:
            _ = service.client
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("GEMINI_API_KEY no está configurada", ctx.exception.detail)

    def test_placeholder_api_key_raises_http_500(self):
        """Verifica que el placeholder de plantilla sea rechazado con HTTP 500."""
        service = GeminiGenerationService(api_key="tu_api_key_de_gemini_aqui")
        with self.assertRaises(HTTPException) as ctx:
            _ = service.client
        self.assertEqual(ctx.exception.status_code, 500)

    @patch("app.services.generation_service.genai.Client")
    def test_generate_response_success(self, mock_client_cls):
        """Verifica la generación exitosa de respuesta con el modelo configurado."""
        mock_response = MagicMock()
        mock_response.text = "RAGIA utiliza ChromaDB para almacenamiento vectorial. [Fuente: manual.md, Sección: Arquitectura]"

        mock_instance = MagicMock()
        mock_instance.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_instance

        service = GeminiGenerationService(api_key="valid_key_123", model="gemini-2.0-flash")
        answer, model_used = service.generate_response(
            system_instruction="Instrucción de contención estricta.",
            user_prompt="¿Qué base vectorial usa RAGIA?"
        )

        self.assertEqual(model_used, "gemini-2.0-flash")
        self.assertIn("RAGIA utiliza ChromaDB", answer)
        self.assertIn("[Fuente: manual.md, Sección: Arquitectura]", answer)
        mock_instance.models.generate_content.assert_called_once()

    def test_generate_response_empty_prompt(self):
        """Verifica que un prompt vacío retorne string vacío sin llamar a la API."""
        service = GeminiGenerationService(api_key="valid_key_123")
        answer, model_used = service.generate_response(
            system_instruction="System",
            user_prompt="   "
        )
        self.assertEqual(answer, "")

    @patch("app.services.generation_service.genai.Client")
    def test_generate_response_fallback_on_404(self, mock_client_cls):
        """Verifica que si gemini-2.0-flash devuelve 404, active el fallback a gemini-3.1-flash-lite."""
        mock_response = MagicMock()
        mock_response.text = "Respuesta fundamentada mediante modelo de fallback."

        mock_instance = MagicMock()
        error_404 = ClientError(404, {"error": {"message": "This model is no longer available"}})

        # Primera llamada (gemini-2.0-flash) falla con 404, segunda llamada (fallback) tiene éxito
        mock_instance.models.generate_content.side_effect = [
            error_404,
            mock_response
        ]
        mock_client_cls.return_value = mock_instance

        service = GeminiGenerationService(api_key="valid_key_123", model="gemini-2.0-flash")
        answer, model_used = service.generate_response(
            system_instruction="System",
            user_prompt="Pregunta con fallback"
        )

        self.assertEqual(model_used, "gemini-3.1-flash-lite")
        self.assertEqual(answer, "Respuesta fundamentada mediante modelo de fallback.")
        self.assertEqual(mock_instance.models.generate_content.call_count, 2)

    @patch("app.services.generation_service.genai.Client")
    def test_generate_response_all_fail_raises_502(self, mock_client_cls):
        """Verifica que si todos los modelos candidatos fallan, se lance HTTP 502."""
        mock_instance = MagicMock()
        mock_instance.models.generate_content.side_effect = Exception("Conexión fallida")
        mock_client_cls.return_value = mock_instance

        service = GeminiGenerationService(api_key="valid_key_123", model="gemini-2.0-flash")
        with self.assertRaises(HTTPException) as ctx:
            service.generate_response(
                system_instruction="System",
                user_prompt="Pregunta que fallará"
            )
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("No se pudo generar la respuesta", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
