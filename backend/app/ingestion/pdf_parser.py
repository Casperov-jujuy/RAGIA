import io
import re
from typing import List
import pypdf
from fastapi import HTTPException, status

from app.models.schemas import DocumentSection


class PDFParser:
    """
    Parser especializado en documentos PDF (.pdf).
    Recorre el documento página por página, extrayendo texto limpio y
    registrando el número de página como metadato para trazabilidad RAG.
    """

    @classmethod
    def clean_extracted_text(cls, text: str) -> str:
        """
        Limpia y normaliza el texto extraído de una página PDF.

        Args:
            text (str): Texto en bruto extraído por pypdf.

        Returns:
            str: Texto limpio y normalizado.
        """
        if not text:
            return ""

        # 1. Eliminar caracteres nulos
        text = text.replace("\x00", "")

        # 2. Reconciliar palabras cortadas con guion al final de línea (ej. docu-\nmento -> documento)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

        # 3. Limpiar espacios horizontales duplicados por línea
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.splitlines()]

        # 4. Eliminar saltos de línea excesivos (más de 2 consecutivos)
        cleaned = "\n".join(lines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        return cleaned.strip()

    @classmethod
    def parse_bytes(cls, content_bytes: bytes, filename: str = "document.pdf") -> List[DocumentSection]:
        """
        Lee los bytes de un archivo PDF y extrae las secciones de texto página por página.

        Args:
            content_bytes (bytes): Contenido binario del archivo PDF.
            filename (str): Nombre del archivo para metadatos.

        Returns:
            List[DocumentSection]: Lista de secciones correspondientes a cada página con texto.

        Raises:
            HTTPException: Error 400 Bad Request si el archivo está corrupto o no es un PDF válido.
        """
        if not content_bytes:
            return []

        try:
            stream = io.BytesIO(content_bytes)
            reader = pypdf.PdfReader(stream)
            total_pages = len(reader.pages)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error al procesar el archivo PDF '{filename}'. El archivo puede estar corrupto o protegido: {str(e)}"
            )

        sections: List[DocumentSection] = []

        for idx, page in enumerate(reader.pages):
            page_number = idx + 1
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""

            cleaned_text = cls.clean_extracted_text(raw_text)

            # Si la página contiene texto válido, la incluimos como sección
            if cleaned_text:
                sections.append(
                    DocumentSection(
                        title=f"Página {page_number}",
                        level=1,
                        content=cleaned_text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                            "total_pages": total_pages,
                            "char_count": len(cleaned_text),
                        }
                    )
                )

        return sections
