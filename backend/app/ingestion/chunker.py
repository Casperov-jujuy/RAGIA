from typing import Any, Dict, List, Optional
from app.models.schemas import DocumentChunk, DocumentSection


class TextChunker:
    """
    Estrategia de fragmentación recursiva de texto (Recursive Character Text Splitter).
    Divide textos y secciones en fragmentos de tamaño óptimo (ej. 500-1000 caracteres)
    con solapamiento (overlap) para preservar el contexto semántico en embeddings y RAG.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Inicializa el configurador de fragmentación.

        Args:
            chunk_size (int): Tamaño máximo aproximado de cada fragmento en caracteres.
            chunk_overlap (int): Cantidad de caracteres de solapamiento entre fragmentos consecutivos.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser mayor a 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap no puede ser negativo")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Divide un texto arbitrario en bloques con solapamiento respetando límites semánticos naturales
        (párrafos -> líneas -> oraciones -> palabras).

        Args:
            text (str): Texto de entrada a fragmentar.

        Returns:
            List[str]: Lista de fragmentos de texto.
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                # Buscar el corte más limpio dentro de la ventana de búsqueda
                search_start = max(start + self.chunk_overlap, start + 1)
                best_end = -1
                for sep in self.DEFAULT_SEPARATORS:
                    pos = text.rfind(sep, search_start, end)
                    if pos != -1:
                        best_end = pos + len(sep)
                        break

                if best_end != -1:
                    end = best_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break

            # Avanzar la ventana manteniendo el solapamiento (overlap)
            next_start = max(start + 1, end - self.chunk_overlap)

            # Evitar cortar una palabra a la mitad al retroceder para el solapamiento
            if next_start < text_len and text[next_start] not in (" ", "\n", "\t"):
                space_pos = text.find(" ", next_start)
                if space_pos != -1 and space_pos < end:
                    next_start = space_pos + 1

            start = next_start

        return chunks

    def split_sections(
        self,
        sections: List[DocumentSection],
        filename: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Toma una lista de secciones extraídas (de Markdown o PDF) y las fragmenta
        preservando y enriqueciendo los metadatos de trazabilidad.

        Args:
            sections (List[DocumentSection]): Secciones previas extraídas por los parsers.
            filename (Optional[str]): Nombre de archivo de respaldo para metadatos.

        Returns:
            List[DocumentChunk]: Lista de fragmentos tipados y listos para embeddings.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for sec in sections:
            source_file = sec.metadata.get("source") or filename or "documento"
            text_splits = self.split_text(sec.content)

            for sec_chunk_idx, split_text in enumerate(text_splits):
                chunk_id = f"{source_file}_chunk_{global_chunk_idx}"

                # Herencia y enriquecimiento de metadatos
                chunk_metadata: Dict[str, Any] = dict(sec.metadata)
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": global_chunk_idx,
                    "section_chunk_index": sec_chunk_idx,
                    "section_title": sec.title,
                    "section_level": sec.level,
                    "char_count": len(split_text),
                })

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        content=split_text,
                        metadata=chunk_metadata
                    )
                )
                global_chunk_idx += 1

        return all_chunks
