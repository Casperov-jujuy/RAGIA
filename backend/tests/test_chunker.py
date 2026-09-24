import unittest
from app.ingestion.chunker import TextChunker
from app.models.schemas import DocumentSection


class TestTextChunker(unittest.TestCase):
    """Suite de pruebas unitarias para la estrategia de fragmentación (TextChunker)."""

    def setUp(self):
        self.chunker = TextChunker(chunk_size=200, chunk_overlap=40)

    def test_short_text_single_chunk(self):
        """Verifica que un texto menor a chunk_size genere exactamente un chunk."""
        text = "Este es un texto corto que cabe en un solo fragmento."
        chunks = self.chunker.split_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_long_text_multiple_chunks_within_limit(self):
        """Verifica que un texto largo genere múltiples chunks y ninguno supere chunk_size."""
        paragraphs = [
            "Párrafo 1 con información importante sobre el inicio del sistema y sus objetivos generales.",
            "Párrafo 2 detallando la arquitectura modular con componentes de ingesta, almacenamiento y generación.",
            "Párrafo 3 explicando el rol de ChromaDB y los embeddings calculados con Google Gemini.",
            "Párrafo 4 con las conclusiones y recomendaciones de despliegue en entornos de producción.",
        ]
        text = "\n\n".join(paragraphs)
        chunks = self.chunker.split_text(text)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.chunker.chunk_size)

    def test_overlap_between_consecutive_chunks(self):
        """Verifica que exista solapamiento (overlap) de contenido entre fragmentos contiguos."""
        text = (
            "La inteligencia artificial generativa ha transformado la interacción con el conocimiento. "
            "Los sistemas de Recuperación Aumentada por Generación (RAG) combinan modelos de lenguaje con "
            "bases de conocimiento externas para proporcionar respuestas con alta precisión y trazabilidad."
        )
        chunker = TextChunker(chunk_size=120, chunk_overlap=35)
        chunks = chunker.split_text(text)

        self.assertGreater(len(chunks), 1)
        # Verificar que el final de un chunk tenga palabras comunes con el inicio del siguiente
        for i in range(len(chunks) - 1):
            curr_words = set(chunks[i].split()[-3:])
            next_words = set(chunks[i + 1].split()[:5])
            common = curr_words.intersection(next_words)
            self.assertGreater(len(common), 0, f"No se encontró overlap entre chunk {i} y {i+1}")

    def test_split_sections_preserves_metadata(self):
        """Verifica que split_sections genere DocumentChunk preservando y enriqueciendo metadatos."""
        sections = [
            DocumentSection(
                title="Página 1",
                level=1,
                content="Contenido de la página uno del informe técnico sobre microservicios.",
                metadata={"source": "manual.pdf", "page": 1, "total_pages": 2}
            ),
            DocumentSection(
                title="Página 2",
                level=1,
                content="Contenido de la página dos con especificaciones técnicas detalladas.",
                metadata={"source": "manual.pdf", "page": 2, "total_pages": 2}
            )
        ]

        chunks = self.chunker.split_sections(sections)
        self.assertEqual(len(chunks), 2)

        # Chunk 1
        self.assertEqual(chunks[0].chunk_id, "manual.pdf_chunk_0")
        self.assertEqual(chunks[0].metadata["source"], "manual.pdf")
        self.assertEqual(chunks[0].metadata["page"], 1)
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[0].metadata["section_title"], "Página 1")

        # Chunk 2
        self.assertEqual(chunks[1].chunk_id, "manual.pdf_chunk_1")
        self.assertEqual(chunks[1].metadata["source"], "manual.pdf")
        self.assertEqual(chunks[1].metadata["page"], 2)
        self.assertEqual(chunks[1].metadata["chunk_index"], 1)
        self.assertEqual(chunks[1].metadata["section_title"], "Página 2")

    def test_invalid_parameters(self):
        """Verifica que parámetros inválidos de tamaño o solapamiento lancen ValueError."""
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=0)
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=-10)
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=150)

    def test_empty_or_whitespace_text(self):
        """Verifica que textos vacíos o solo espacios retornen lista vacía."""
        self.assertEqual(self.chunker.split_text(""), [])
        self.assertEqual(self.chunker.split_text("   \n\n\t  "), [])


if __name__ == "__main__":
    unittest.main()
