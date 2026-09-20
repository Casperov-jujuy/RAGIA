from fastapi import APIRouter, File, UploadFile, status

from app.ingestion.validator import validate_file_extension

router = APIRouter()


@router.post(
    "/upload",
    status_code=status.HTTP_200_OK,
    summary="Subir y validar documento (.md o .pdf)",
    description="Valida estrictamente que el archivo subido sea un Markdown (.md) o PDF (.pdf). Cualquier otro formato es rechazado con error 400."
)
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint para recibir y validar archivos para el pipeline RAG.
    """
    # Validación estricta de formato (Paso 2.1)
    validated_ext = validate_file_extension(file)

    return {
        "filename": file.filename,
        "extension": validated_ext,
        "status": "approved",
        "message": f"Archivo validado exitosamente. Formato '{validated_ext}' admitido para el procesamiento RAG."
    }
