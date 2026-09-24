from fastapi import APIRouter, File, UploadFile, status

from app.ingestion.validator import validate_file_extension
from app.ingestion.md_parser import MarkdownParser
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import TextChunker
from app.models.schemas import DocumentUploadResponse, SectionSummary

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Subir, validar, procesar y fragmentar documento (.md o .pdf)",
    description="Valida la extensión (.md o .pdf), extrae las secciones lógicas y las fragmenta en chunks optimizados para embeddings."
)
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint para recibir, validar, parsear y fragmentar archivos para el pipeline RAG.
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

    # 3. Fragmentación de texto (Chunking) para embeddings y vector store
    chunker = TextChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.split_sections(sections, filename=file.filename)

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
        status="chunked",
        message=(
            f"Archivo {validated_ext.upper()[1:]} procesado exitosamente: "
            f"{doc_type_desc} y fragmentado en {len(chunks)} chunks optimizados para embeddings."
        ),
        total_sections=len(sections),
        total_chunks=len(chunks),
        sections_summary=sections_summary,
    )
