from typing import Any, Dict, List, Optional, Tuple
from app.models.schemas import RetrievedChunk

NO_INFO_FALLBACK = "No dispongo de suficiente información en los documentos proporcionados para responder a esta pregunta."


class PromptBuilder:
    """
    Servicio de formulación de prompts con contención estricta y citación de fuentes
    para modelos de lenguaje Google Gemini (Paso 4.3).
    """

    DEFAULT_SYSTEM_INSTRUCTION = (
        "Eres un asistente de inteligencia artificial especializado en responder preguntas "
        "basándote de forma estricta y exclusiva en la documentación proporcionada en el contexto.\n\n"
        "REGLAS OBLIGATORIAS:\n"
        "1. CONTENCIÓN ESTRICTA: Responde única y exclusivamente utilizando los hechos e información explícitamente presentes en los fragmentos de contexto proporcionados. No utilices conocimiento previo ni asumas o extrapoles información que no esté demostrada en el texto.\n"
        "2. POLÍTICA ANTI-ALUCINACIONES: Si la respuesta a la pregunta no se encuentra completa o explícitamente en el contexto proporcionado, o si el contexto está vacío, responde de manera exacta:\n"
        f'   "{NO_INFO_FALLBACK}"\n'
        "   No agregues opiniones, conjeturas ni explicaciones adicionales en este caso.\n"
        "3. CITACIÓN OBLIGATORIA DE FUENTES: Cada afirmación, dato o respuesta basada en el contexto debe incluir su referencia de origen en formato entre corchetes al final de la oración o párrafo correspondiente:\n"
        "   - Si el fragmento proviene de un PDF con número de página: [Fuente: <archivo>, Página <X>]\n"
        "   - Si el fragmento proviene de Markdown con título de sección: [Fuente: <archivo>, Sección: <Título>]\n"
        "   - Si contiene ambos: [Fuente: <archivo>, Página <X>, Sección: <Título>]\n"
        "   - Si no dispone de página ni sección: [Fuente: <archivo>]\n"
        "4. TONO Y FORMATO: Mantén un tono formal, objetivo, claro y conciso. Responde en el mismo idioma de la pregunta formulada."
    )

    def __init__(self, system_instruction: Optional[str] = None):
        self._system_instruction = system_instruction or self.DEFAULT_SYSTEM_INSTRUCTION

    def build_system_instruction(self) -> str:
        """Devuelve las directivas estrictas de contención y citación para Gemini."""
        return self._system_instruction

    @staticmethod
    def format_source_tag(source: str, page: Optional[int] = None, section_title: Optional[str] = None) -> str:
        """
        Genera la etiqueta estandarizada de referencia a la fuente:
        Ejemplo: [Fuente: manual.pdf, Página 3] o [Fuente: readme.md, Sección: Instalación]
        """
        parts = [f"Fuente: {source}"]
        if page is not None:
            parts.append(f"Página {page}")
        if section_title:
            parts.append(f"Sección: {section_title}")
        return f"[{', '.join(parts)}]"

    def format_context(self, retrieved_chunks: List[RetrievedChunk]) -> str:
        """
        Estructura los fragmentos recuperados en bloques delimitados y legibles con sus metadatos.
        """
        if not retrieved_chunks:
            return "(No se encontraron fragmentos relevantes en los documentos indexados)"

        formatted_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source_desc = f"Archivo: {chunk.source}"
            details = []
            if chunk.page is not None:
                details.append(f"Página: {chunk.page}")
            if chunk.section_title:
                details.append(f"Sección: {chunk.section_title}")
            if details:
                source_desc += f" ({', '.join(details)})"

            block = (
                f"--- FRAGMENTO {idx} ---\n"
                f"Origen: {source_desc}\n"
                f"Texto:\n{chunk.content.strip()}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        custom_system_instruction: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Construye la tupla (system_instruction, user_prompt) para la llamada generativa a Gemini.
        """
        system_instruction = custom_system_instruction or self.build_system_instruction()
        formatted_context = self.format_context(retrieved_chunks)

        user_prompt = (
            "CONTEXTO PROPORCIONADO:\n"
            f"{formatted_context}\n\n"
            "PREGUNTA DEL USUARIO:\n"
            f"{question.strip()}\n\n"
            "INSTRUCCIÓN:\n"
            "Responde a la pregunta del usuario utilizando exclusivamente el CONTEXTO PROPORCIONADO arriba. "
            "Asegúrate de citar las fuentes de forma rigurosa utilizando la etiqueta de origen correspondiente. "
            f'Si la respuesta no se encuentra en el contexto, responde exactamente: "{NO_INFO_FALLBACK}"'
        )

        return system_instruction, user_prompt

    def build_sources_summary(self, retrieved_chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """
        Genera una lista de metadatos estructurados para auditoría y visualización de las fuentes.
        """
        return [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page,
                "section_title": chunk.section_title,
                "similarity_score": round(chunk.similarity_score, 4),
                "citation_tag": self.format_source_tag(chunk.source, chunk.page, chunk.section_title),
            }
            for chunk in retrieved_chunks
        ]
