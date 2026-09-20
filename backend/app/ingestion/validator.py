import os
from fastapi import HTTPException, UploadFile, status

# Extensiones estrictamente permitidas por el sistema RAG
ALLOWED_EXTENSIONS = {".md", ".pdf"}


def validate_file_extension(file: UploadFile) -> str:
    """
    Valida estrictamente que el archivo subido tenga una extensión permitida (.md o .pdf).

    Args:
        file (UploadFile): Objeto de archivo recibido en la solicitud multipart/form-data.

    Returns:
        str: La extensión normalizada en minúsculas (ej: '.md' o '.pdf').

    Raises:
        HTTPException: Error 400 Bad Request si el archivo carece de nombre o tiene
                       una extensión no permitida (imágenes, ejecutables, word, etc.).
    """
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo enviado no tiene un nombre válido o está vacío."
        )

    _, ext = os.path.splitext(file.filename)
    normalized_ext = ext.lower().strip()

    if normalized_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Formato de archivo no permitido: '{normalized_ext if normalized_ext else 'sin extensión'}'. "
                f"El sistema solo acepta archivos {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )
        )

    return normalized_ext
