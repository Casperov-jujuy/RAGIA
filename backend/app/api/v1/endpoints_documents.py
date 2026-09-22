from fastapi import APIRouter, File, UploadFile, status

from app.ingestion.validator import validate_file_extension
from app.ingestion.md_parser import MarkdownParser
from app.ingestion.pdf_parser import PDFParser
from app.models.schemas import DocumentUploadResponse, SectionSummary

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Subir, validar y procesar documento (.md o .pdf)",
    description="Valida la extensión (.md o .pdf) y fragmenta el contenido de Markdown (.md por títulos) o PDF (.pdf por páginas con texto)."
)
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint para recibir, validar y parsear archivos para el pipeline RAG.
    """
    # 1. Validación estricta de formato (.md o .pdf)
    validated_ext = validate_file_extension(file)

    raw_bytes = await file.read()
    sections = []

    # 2. Procesamiento según el tipo de archivo
    if validated_ext == ".md":
        try:
            content_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content_text = raw_bytes.decode("latin-1", errors="replace")

        sections = MarkdownParser.parse_text(content_text, filename=file.filename)
        message = f"Archivo Markdown parseado exitosamente en {len(sections)} secciones lógicas."

    elif validated_ext == ".pdf":
        sections = PDFParser.parse_bytes(raw_bytes, filename=file.filename)
        message = f"Archivo PDF parseado exitosamente en {len(sections)} páginas con texto."

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
        status="parsed",
        message=message,
        total_sections=len(sections),
        sections_summary=sections_summary,
    )
