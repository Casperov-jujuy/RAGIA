from fastapi import APIRouter, HTTPException, status
from app.core.config import settings
from app.services.embedding_service import GeminiEmbeddingService
from app.models.schemas import QueryRequest, QueryVectorResponse

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryVectorResponse,
    status_code=status.HTTP_200_OK,
    summary="Vectorizar pregunta del usuario (Paso 4.1)",
    description="Recibe una pregunta en texto y genera su vector denso utilizando Google Gemini Embeddings."
)
async def query_rag(request: QueryRequest):
    """
    Endpoint de consulta RAG: Transforma la pregunta del usuario en vector de embedding.
    """
    cleaned_question = request.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La pregunta no puede estar vacía o contener solo espacios."
        )

    embedding_service = GeminiEmbeddingService()
    query_vector = embedding_service.embed_text(cleaned_question)

    if not query_vector:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo generar el vector numérico para la pregunta enviada."
        )

    preview = [round(val, 4) for val in query_vector[:5]]

    return QueryVectorResponse(
        question=cleaned_question,
        embedding_dimension=len(query_vector),
        embedding_preview=preview,
        message="Pregunta vectorizada exitosamente con Google Gemini. Lista para búsqueda semántica en ChromaDB."
    )
