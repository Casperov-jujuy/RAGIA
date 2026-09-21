from fastapi import APIRouter, File, UploadFile, status

from app.ingestion.validator import validate_file_extension
from app.ingestion.md_parser import MarkdownParser
from app.models.schemas import DocumentUploadResponse, SectionSummary

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Subir, validar y procesar documento (.md o .pdf)",
    description="Valida la extensión (.md o .pdf) y fragmenta el contenido de Markdown (.md) respetando la jerarquía de títulos."
)
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint para recibir, validar y parsear archivos para el pipeline RAG.
    """
    # 1. Validación estricta de formato (.md o .pdf)
    validated_ext = validate_file_extension(file)

    total_sections = None
    sections_summary = None

    # 2. Si es archivo Markdown, parsear sus secciones respetando títulos
    if validated_ext == ".md":
        raw_bytes = await file.read()
        try:
            content_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content_text = raw_bytes.decode("latin-1", errors="replace")

        sections = MarkdownParser.parse_text(content_text, filename=file.filename)
        total_sections = len(sections)
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
        status="parsed" if validated_ext == ".md" else "approved",
        message=(
            f"Archivo Markdown parseado exitosamente en {total_sections} secciones lógicas."
            if validated_ext == ".md"
            else f"Archivo PDF validado exitosamente. Listo para el parser de PDF."
        ),
        total_sections=total_sections,
        sections_summary=sections_summary,
    )
