from fastapi import APIRouter, HTTPException, status
from app.services.embedding_service import GeminiEmbeddingService
from app.services.vector_store import ChromaVectorStore
from app.models.schemas import QueryRequest, QueryRetrievalResponse, RetrievedChunk

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryRetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Búsqueda semántica (Retrieval) en ChromaDB",
    description="Vectoriza la pregunta del usuario con Gemini y recupera los Top-K fragmentos más relevantes de ChromaDB ordenados por similitud coseno."
)
async def query_rag(request: QueryRequest):
    """
    Endpoint de consulta RAG (Paso 4.2): Vectorización + Búsqueda de similitud.
    """
    cleaned_question = request.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La pregunta no puede estar vacía o contener solo espacios."
        )

    # 1. Vectorizar la pregunta con Google Gemini
    embedding_service = GeminiEmbeddingService()
    query_vector = embedding_service.embed_text(cleaned_question)

    if not query_vector:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo generar el vector numérico para la pregunta enviada."
        )

    # 2. Consultar ChromaDB para recuperar los Top-K fragmentos más relevantes
    vector_store = ChromaVectorStore()
    raw_hits = vector_store.similarity_search(query_vector=query_vector, top_k=request.top_k)

    # 3. Mapear resultados a fragmentos tipados con metadatos de cita
    retrieved_chunks = []
    for hit in raw_hits:
        meta = hit.get("metadata", {})
        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=hit.get("chunk_id", ""),
                content=hit.get("content", ""),
                similarity_score=hit.get("similarity_score", 0.0),
                source=meta.get("source", "desconocido"),
                page=meta.get("page"),
                section_title=meta.get("section_title"),
                char_count=len(hit.get("content", ""))
            )
        )

    total_found = len(retrieved_chunks)
    if total_found > 0:
        message = f"Se recuperaron exitosamente {total_found} fragmentos relevantes desde ChromaDB."
    else:
        message = (
            "No se encontraron fragmentos relevantes. "
            "Asegúrate de haber subido e indexado documentos (.md o .pdf) previamente a la base vectorial."
        )

    return QueryRetrievalResponse(
        question=cleaned_question,
        total_retrieved=total_found,
        retrieved_chunks=retrieved_chunks,
        message=message
    )
