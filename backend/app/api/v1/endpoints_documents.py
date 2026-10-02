from fastapi import APIRouter, File, UploadFile, status

from app.core.config import settings
from app.ingestion.validator import validate_file_extension
from app.ingestion.md_parser import MarkdownParser
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import TextChunker
from app.services.embedding_service import GeminiEmbeddingService
from app.models.schemas import DocumentUploadResponse, SectionSummary

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Subir, validar, fragmentar y generar embeddings (.md o .pdf)",
    description="Valida el archivo, extrae secciones lógicas, las fragmenta en chunks y genera representaciones vectoriales con Google Gemini (text-embedding-004)."
)
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint para recibir, validar, parsear, fragmentar y vectorizar documentos para RAGIA.
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
        upload_status = "embedded"
        status_msg = (
            f"Archivo {validated_ext.upper()[1:]} procesado exitosamente: "
            f"{doc_type_desc}, {len(chunks)} chunks y vectores de {embedding_dim} dimensiones generados con Gemini."
        )
    else:
        status_msg = (
            f"Archivo {validated_ext.upper()[1:]} procesado: {doc_type_desc} y {len(chunks)} chunks generados. "
            f"(Nota: Configura tu GEMINI_API_KEY en backend/.env para generar embeddings automáticamente)."
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
