import unittest
from app.services.prompt_service import PromptBuilder, NO_INFO_FALLBACK
from app.models.schemas import RetrievedChunk


class TestPromptService(unittest.TestCase):
    """Suite de pruebas unitarias para PromptBuilder (Paso 4.3)."""

    def setUp(self):
        self.builder = PromptBuilder()

    def test_default_system_instruction_contains_core_directives(self):
        """Verifica que las instrucciones del sistema incluyan contención, anti-alucinaciones y citas."""
        instruction = self.builder.build_system_instruction()

        # 1. Regla de contención
        self.assertIn("CONTENCIÓN ESTRICTA", instruction)
        self.assertIn("única y exclusivamente", instruction)

        # 2. Regla anti-alucinación con frase exacta
        self.assertIn("POLÍTICA ANTI-ALUCINACIONES", instruction)
        self.assertIn(NO_INFO_FALLBACK, instruction)

        # 3. Formatos de citación obligatorios
        self.assertIn("CITACIÓN OBLIGATORIA", instruction)
        self.assertIn("[Fuente: <archivo>, Página <X>]", instruction)
        self.assertIn("[Fuente: <archivo>, Sección: <Título>]", instruction)

    def test_format_source_tag(self):
        """Verifica la generación de etiquetas de citación según metadatos disponibles."""
        # PDF con página
        tag_pdf = PromptBuilder.format_source_tag("manual.pdf", page=4)
        self.assertEqual(tag_pdf, "[Fuente: manual.pdf, Página 4]")

        # Markdown con sección
        tag_md = PromptBuilder.format_source_tag("arquitectura.md", section_title="Base de Datos")
        self.assertEqual(tag_md, "[Fuente: arquitectura.md, Sección: Base de Datos]")

        # PDF con página y sección
        tag_both = PromptBuilder.format_source_tag("informe.pdf", page=12, section_title="Conclusiones")
        self.assertEqual(tag_both, "[Fuente: informe.pdf, Página 12, Sección: Conclusiones]")

        # Sin página ni sección
        tag_simple = PromptBuilder.format_source_tag("datos.txt")
        self.assertEqual(tag_simple, "[Fuente: datos.txt]")

    def test_format_context_with_mixed_chunks(self):
        """Verifica el formato del bloque de contexto para fragmentos de PDF y Markdown."""
        chunks = [
            RetrievedChunk(
                chunk_id="doc1_chunk_0",
                content="RAGIA utiliza FastAPI y ChromaDB.",
                similarity_score=0.92,
                source="manual.pdf",
                page=2,
                section_title="Arquitectura",
                char_count=33
            ),
            RetrievedChunk(
                chunk_id="doc2_chunk_1",
                content="Para ejecutar tests usa python -m unittest.",
                similarity_score=0.87,
                source="guia.md",
                page=None,
                section_title="Testing",
                char_count=43
            )
        ]

        context = self.builder.format_context(chunks)

        self.assertIn("--- FRAGMENTO 1 ---", context)
        self.assertIn("Archivo: manual.pdf (Página: 2, Sección: Arquitectura)", context)
        self.assertIn("RAGIA utiliza FastAPI y ChromaDB.", context)

        self.assertIn("--- FRAGMENTO 2 ---", context)
        self.assertIn("Archivo: guia.md (Sección: Testing)", context)
        self.assertIn("Para ejecutar tests usa python -m unittest.", context)

    def test_format_context_empty(self):
        """Verifica el comportamiento cuando la lista de fragmentos está vacía."""
        context = self.builder.format_context([])
        self.assertEqual(context, "(No se encontraron fragmentos relevantes en los documentos indexados)")

    def test_build_prompt_structure(self):
        """Verifica la construcción del prompt completo (instrucción de sistema y prompt de usuario)."""
        chunks = [
            RetrievedChunk(
                chunk_id="chunk_1",
                content="Texto relevante del manual.",
                similarity_score=0.95,
                source="manual.pdf",
                page=1,
                section_title=None,
                char_count=27
            )
        ]
        question = "¿Qué dice el manual?"

        sys_inst, user_prompt = self.builder.build_prompt(question, chunks)

        self.assertIn("CONTENCIÓN ESTRICTA", sys_inst)
        self.assertIn("CONTEXTO PROPORCIONADO:", user_prompt)
        self.assertIn("PREGUNTA DEL USUARIO:\n¿Qué dice el manual?", user_prompt)
        self.assertIn("Texto relevante del manual.", user_prompt)
        self.assertIn(NO_INFO_FALLBACK, user_prompt)

    def test_build_sources_summary(self):
        """Verifica la generación del resumen de fuentes para la respuesta de la API."""
        chunks = [
            RetrievedChunk(
                chunk_id="chunk_test",
                content="Contenido de prueba.",
                similarity_score=0.897654,
                source="especificacion.md",
                page=None,
                section_title="Endpoints",
                char_count=20
            )
        ]

        summary = self.builder.build_sources_summary(chunks)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["chunk_id"], "chunk_test")
        self.assertEqual(summary[0]["source"], "especificacion.md")
        self.assertEqual(summary[0]["section_title"], "Endpoints")
        self.assertEqual(summary[0]["similarity_score"], 0.8977)
        self.assertEqual(summary[0]["citation_tag"], "[Fuente: especificacion.md, Sección: Endpoints]")


if __name__ == "__main__":
    unittest.main()
