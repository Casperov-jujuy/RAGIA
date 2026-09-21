import re
from typing import List
from app.models.schemas import DocumentSection

# Expresión regular para detectar títulos Markdown (# Titulo, ## Titulo, etc.)
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


class MarkdownParser:
    """
    Parser especializado en documentos Markdown (.md).
    Divide el contenido respetando la estructura lógica y jerárquica de los encabezados (#, ##, etc.),
    ignorando falsos encabezados dentro de bloques de código.
    """

    @classmethod
    def parse_text(cls, text: str, filename: str = "document.md") -> List[DocumentSection]:
        """
        Analiza el texto en formato Markdown y lo divide en secciones estructuradas.

        Args:
            text (str): Contenido completo del archivo Markdown.
            filename (str): Nombre del archivo de origen para metadatos.

        Returns:
            List[DocumentSection]: Lista de secciones lógicas con título, nivel y contenido.
        """
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        sections: List[DocumentSection] = []

        current_title = "Introducción"
        current_level = 0
        current_lines: List[str] = []
        in_code_block = False

        def flush_section():
            nonlocal current_lines, current_title, current_level
            content = "\n".join(current_lines).strip()
            # Guardar la sección si tiene contenido o si tiene un título asignado
            if content or (current_title != "Introducción" and current_level > 0):
                sections.append(
                    DocumentSection(
                        title=current_title,
                        level=current_level,
                        content=content,
                        metadata={
                            "source": filename,
                            "section": current_title,
                            "level": current_level,
                            "char_count": len(content),
                        }
                    )
                )
            current_lines = []

        for line in lines:
            stripped = line.strip()

            # Detección de delimitadores de bloques de código (``` o ~~~)
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            # Si estamos dentro de un bloque de código, no interpretamos encabezados
            if in_code_block:
                current_lines.append(line)
                continue

            # Detección de encabezados (#, ##, ###...)
            match = HEADER_PATTERN.match(line)
            if match:
                # Guardar la sección acumulada previamente
                flush_section()

                hashes, title = match.groups()
                current_level = len(hashes)
                current_title = title.strip()
            else:
                current_lines.append(line)

        # Volcar la última sección acumulada
        flush_section()

        # Si el documento no contenía títulos pero tenía texto, ajustar título a 'Documento'
        if len(sections) == 1 and sections[0].level == 0 and sections[0].title == "Introducción":
            sections[0].title = "Documento"
            sections[0].metadata["section"] = "Documento"

        return sections
