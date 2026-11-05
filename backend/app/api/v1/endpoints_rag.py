from typing import List
from fastapi import APIRouter, HTTPException, status
from app.services.embedding_service import GeminiEmbeddingService
from app.services.vector_store import ChromaVectorStore
from app.services.prompt_service import PromptBuilder, NO_INFO_FALLBACK
from app.services.generation_service import GeminiGenerationService
from app.models.schemas import (
    QueryRequest,
    QueryRetrievalResponse,
    RetrievedChunk,
    PromptPreviewResponse,
    RAGQueryResponse,
)

router = APIRouter()


def _retrieve_chunks(cleaned_question: str, top_k: int) -> List[RetrievedChunk]:
    """Recupera los fragmentos más relevantes de ChromaDB para una pregunta vectorizada."""
    embedding_service = GeminiEmbeddingService()
    query_vector = embedding_service.embed_text(cleaned_question)

    if not query_vector:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo generar el vector numérico para la pregunta enviada."
        )

    vector_store = ChromaVectorStore()
    raw_hits = vector_store.similarity_search(query_vector=query_vector, top_k=top_k)

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
    return retrieved_chunks


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Consulta RAG completa (Generación fundamentada con Gemini Flash)",
    description=(
        "Pipeline RAG completo: Vectoriza la pregunta con Gemini, recupera los fragmentos más relevantes de ChromaDB, "
        "construye un prompt con contención estricta y genera la respuesta final fundamentada con citas usando Gemini Flash."
    )
)
async def query_rag(request: QueryRequest):
    """
    Endpoint RAG completo (Paso 4.4): Retrieval + Prompt con contención + Generación fundamentada.
    """
    cleaned_question = request.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La pregunta no puede estar vacía o contener solo espacios."
        )

    retrieved_chunks = _retrieve_chunks(cleaned_question, request.top_k)

    # Si la base de datos no contiene fragmentos indexados, responder con la directiva anti-alucinaciones directa
    if not retrieved_chunks:
        return RAGQueryResponse(
            question=cleaned_question,
            answer=NO_INFO_FALLBACK,
            total_sources=0,
            retrieved_chunks=[],
            model_used="contencion-directa"
        )

    # Construir el prompt con contención estricta y formato de citas
    prompt_builder = PromptBuilder()
    system_instruction, formatted_prompt = prompt_builder.build_prompt(
        question=cleaned_question,
        retrieved_chunks=retrieved_chunks
    )

    # Generar la respuesta final utilizando Gemini Flash
    generation_service = GeminiGenerationService()
    answer, model_used = generation_service.generate_response(
        system_instruction=system_instruction,
        user_prompt=formatted_prompt
    )

    return RAGQueryResponse(
        question=cleaned_question,
        answer=answer,
        total_sources=len(retrieved_chunks),
        retrieved_chunks=retrieved_chunks,
        model_used=model_used
    )


@router.post(
    "/retrieve",
    response_model=QueryRetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Búsqueda semántica (Retrieval puro) en ChromaDB",
    description="Vectoriza la pregunta del usuario con Gemini y recupera los Top-K fragmentos más relevantes de ChromaDB ordenados por similitud coseno sin llamada al LLM generativo."
)
async def retrieve_chunks_only(request: QueryRequest):
    """
    Endpoint de recuperación semántica pura (Paso 4.2).
    """
    cleaned_question = request.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La pregunta no puede estar vacía o contener solo espacios."
        )

    retrieved_chunks = _retrieve_chunks(cleaned_question, request.top_k)
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


@router.post(
    "/prompt-preview",
    response_model=PromptPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Vista previa del prompt RAG con contención y citas",
    description="Recupera los fragmentos relevantes y formula el prompt estructurado junto a las directivas de contención estricta para Gemini."
)
async def preview_rag_prompt(request: QueryRequest):
    """
    Endpoint de vista previa del Prompt (Paso 4.3): Retrieval + PromptBuilder con contención y citas.
    """
    cleaned_question = request.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La pregunta no puede estar vacía o contener solo espacios."
        )

    retrieved_chunks = _retrieve_chunks(cleaned_question, request.top_k)

    prompt_builder = PromptBuilder()
    system_instruction, formatted_prompt = prompt_builder.build_prompt(
        question=cleaned_question,
        retrieved_chunks=retrieved_chunks
    )
    sources = prompt_builder.build_sources_summary(retrieved_chunks)

    return PromptPreviewResponse(
        question=cleaned_question,
        system_instruction=system_instruction,
        formatted_prompt=formatted_prompt,
        total_sources=len(retrieved_chunks),
        sources=sources
    )
