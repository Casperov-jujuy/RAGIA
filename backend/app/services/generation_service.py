import logging
from typing import List, Optional, Tuple
from google import genai
from google.genai import types
from google.genai.errors import ClientError, APIError
from fastapi import HTTPException, status

from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiGenerationService:
    """
    Servicio para generar respuestas fundamentadas utilizando los modelos
    Google Gemini (por defecto gemini-2.0-flash con fallback de resiliencia).
    """

    FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-flash-latest"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Inicializa el servicio de generación de texto.

        Args:
            api_key (Optional[str]): Clave de API de Gemini (por defecto de settings).
            model (Optional[str]): Nombre del modelo de generación (por defecto de settings).
        """
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_LLM_MODEL
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        """Obtiene o inicializa el cliente oficial de Google GenAI."""
        if self._client is None:
            clean_key = self.api_key.strip() if self.api_key else ""
            if not clean_key or clean_key.startswith("tu_api_key"):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(
                        "GEMINI_API_KEY no está configurada o contiene el placeholder por defecto. "
                        "Configura tu clave en el archivo backend/.env para generar respuestas."
                    )
                )
            self._client = genai.Client(api_key=clean_key)
        return self._client

    def generate_response(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: float = 0.2
    ) -> Tuple[str, str]:
        """
        Envía las instrucciones del sistema y el prompt del usuario a Gemini.
        Implementa resiliencia y fallback automático si el modelo configurado
        reporta descontinuación (404 NOT_FOUND) en la API de Google.

        Args:
            system_instruction (str): Directivas de contención y formato de citas.
            user_prompt (str): Contexto inyectado y pregunta del usuario.
            temperature (float): Parámetro de temperatura (baja por defecto para fidelidad fáctica).

        Returns:
            Tuple[str, str]: (Texto de la respuesta generada, Nombre del modelo utilizado)
        """
        if not user_prompt or not user_prompt.strip():
            return "", self.model

        # Construir la lista de candidatos iniciando con el modelo seleccionado
        candidate_models: List[str] = [self.model]
        for fallback in self.FALLBACK_MODELS:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature
        )

        last_error: Optional[Exception] = None

        for model_candidate in candidate_models:
            try:
                response = self.client.models.generate_content(
                    model=model_candidate,
                    contents=user_prompt,
                    config=config
                )
                answer = response.text or ""
                return answer.strip(), model_candidate
            except ClientError as e:
                # Si el modelo fue retirado o no está disponible en la cuenta (404), probar siguiente fallback
                if getattr(e, "code", None) == 404 or "404" in str(e) or "NOT_FOUND" in str(e):
                    logger.warning(
                        "El modelo '%s' retornó 404 en Google GenAI API. Intentando fallback...",
                        model_candidate
                    )
                    last_error = e
                    continue
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Error en la API de Google Gemini ({model_candidate}): {str(e)}"
                )
            except HTTPException:
                raise
            except Exception as e:
                last_error = e
                logger.warning("Fallo al generar con '%s': %s", model_candidate, str(e))
                continue

        # Si ningún modelo pudo generar respuesta
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"No se pudo generar la respuesta con ningún modelo Gemini Flash disponible: {str(last_error)}"
        )
