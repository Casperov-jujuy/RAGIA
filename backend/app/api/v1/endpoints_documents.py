from typing import Any, Dict, List
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.ingestion.validator import validate_file_extension
from app.ingestion.md_parser import MarkdownParser
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import TextChunker
from app.services.embedding_service import GeminiEmbeddingService
from app.services.vector_store import ChromaVectorStore
from app.models.schemas import DocumentUploadResponse, SectionSummary

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Subir, fragmentar, vectorizar e indexar documento (.md o .pdf)",
    description="Valida el archivo, extrae secciones, genera chunks, calcula embeddings con Gemini y los almacena de forma persistente en ChromaDB."
)
async def upload_document(file: UploadFile = File(...)):
    """
    Pipeline completo de ingesta, vectorización e indexación en ChromaDB.
    """
    # 1. Validación estricta de formato (.md o .pdf)
    validated_ext = validate_file_extension(file)

    raw_bytes = await file.read()
    sections = []

    # 2. Extracción semántica según el tipo de archivo
    if validated_ext == ".md":
        try:
            content_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content_text = raw_bytes.decode("latin-1", errors="replace")

        sections = MarkdownParser.parse_text(content_text, filename=file.filename)
        doc_type_desc = f"{len(sections)} secciones lógicas"

    elif validated_ext == ".pdf":
        sections = PDFParser.parse_bytes(raw_bytes, filename=file.filename)
        doc_type_desc = f"{len(sections)} páginas con texto"

    # 3. Fragmentación de texto (Chunking)
    chunker = TextChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.split_sections(sections, filename=file.filename)

    # 4. Generación de Embeddings con Google Gemini
    embedding_dim = None
    upload_status = "chunked"

    if settings.has_valid_gemini_key and chunks:
        embedding_service = GeminiEmbeddingService()
        chunks = embedding_service.embed_chunks(chunks)
        embedding_dim = len(chunks[0].embedding) if chunks[0].embedding else 768

        # 5. Almacenamiento persistente en ChromaDB
        vector_store = ChromaVectorStore()
        indexed_count = vector_store.add_chunks(chunks)

        upload_status = "indexed"
        status_msg = (
            f"Archivo {validated_ext.upper()[1:]} procesado e indexado exitosamente en ChromaDB: "
            f"{doc_type_desc}, {indexed_count} chunks almacenados con vectores de {embedding_dim} dimensiones."
        )
    else:
        status_msg = (
            f"Archivo {validated_ext.upper()[1:]} procesado: {doc_type_desc} y {len(chunks)} chunks generados. "
            f"(Nota: Configura tu GEMINI_API_KEY en backend/.env para generar embeddings e indexar en ChromaDB)."
        )

    sections_summary = [
        SectionSummary(
            title=sec.title,
            level=sec.level,
            char_count=len(sec.content)
        )
        for sec in sections
    ]

    return DocumentUploadResponse(
        filename=file.filename,
        extension=validated_ext,
        status=upload_status,
        message=status_msg,
        total_sections=len(sections),
        total_chunks=len(chunks),
        embedding_dimension=embedding_dim,
        sections_summary=sections_summary,
    )


@router.get(
    "",
    response_model=List[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Listar documentos indexados en ChromaDB",
    description="Retorna la lista de todos los documentos actualmente almacenados en ChromaDB con el conteo de chunks y páginas/secciones."
)
async def list_documents():
    """Obtiene el listado de documentos guardados en la base vectorial."""
    vector_store = ChromaVectorStore()
    return vector_store.list_documents()


@router.delete(
    "/{filename}",
    status_code=status.HTTP_200_OK,
    summary="Eliminar documento de ChromaDB",
    description="Elimina todos los fragmentos y vectores de un archivo específico de la base de datos persistente."
)
async def delete_document(filename: str):
    """Elimina los vectores y metadatos asociados al documento especificado."""
    vector_store = ChromaVectorStore()
    deleted_count = vector_store.delete_by_source(filename)

    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontraron fragmentos para el archivo '{filename}' en ChromaDB."
        )

    return {
        "filename": filename,
        "deleted_chunks": deleted_count,
        "message": f"Documento '{filename}' y sus {deleted_count} fragmentos eliminados exitosamente de ChromaDB."
    }
