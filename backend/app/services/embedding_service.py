from typing import List, Optional
from google import genai
from fastapi import HTTPException, status

from app.core.config import settings
from app.models.schemas import DocumentChunk


class GeminiEmbeddingService:
    """
    Servicio para generar representaciones vectoriales densas (embeddings)
    utilizando la API oficial de Google Gemini (modelo text-embedding-004).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Inicializa el servicio de embeddings.

        Args:
            api_key (Optional[str]): Clave de API de Gemini (por defecto tomada de settings).
            model (Optional[str]): Nombre del modelo de embeddings (por defecto text-embedding-004).
        """
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        chosen_model = model or settings.GEMINI_EMBEDDING_MODEL
        # Mapear text-embedding-004 a gemini-embedding-001 para la API v1beta
        self.model = "gemini-embedding-001" if chosen_model == "text-embedding-004" else chosen_model
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
                        "Configura tu clave en el archivo backend/.env para generar embeddings."
                    )
                )
            self._client = genai.Client(api_key=clean_key)
        return self._client

    def embed_text(self, text: str) -> List[float]:
        """
        Genera el vector de embedding para un texto individual.

        Args:
            text (str): Texto a vectorizar.

        Returns:
            List[float]: Vector numérico de punto flotante (768 dimensiones).
        """
        if not text or not text.strip():
            return []

        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            if response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0].values
            return []
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Error al conectar con la API de Gemini Embeddings: {str(e)}"
            )

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos procesándolos en lotes (batching).

        Args:
            texts (List[str]): Lista de textos a vectorizar.
            batch_size (int): Tamaño máximo de cada lote para optimizar latencia.

        Returns:
            List[List[float]]: Lista de vectores correspondientes a cada texto.
        """
        if not texts:
            return []

        all_vectors: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=batch
                )
                if response.embeddings:
                    for emb in response.embeddings:
                        all_vectors.append(emb.values)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Error en lote de embeddings de Gemini: {str(e)}"
                )

        return all_vectors

    def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 32
    ) -> List[DocumentChunk]:
        """
        Asigna el vector de embedding a cada DocumentChunk procesando sus contenidos en lotes.

        Args:
            chunks (List[DocumentChunk]): Fragmentos de texto generados por TextChunker.
            batch_size (int): Tamaño de lote para la API de Gemini.

        Returns:
            List[DocumentChunk]: Lista de fragmentos enriquecidos con su vector numérico.
        """
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        vectors = self.embed_texts(texts, batch_size=batch_size)

        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector

        return chunks
