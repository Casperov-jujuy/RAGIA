from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    """Representa una sección o fragmento lógico extraído de un documento."""
    title: str = Field(..., description="Título de la sección")
    level: int = Field(default=1, description="Nivel jerárquico del encabezado (0 para intro, 1 para #, 2 para ##)")
    content: str = Field(..., description="Contenido de texto de la sección")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadatos contextuales (fuente, posición, longitud)")


class DocumentChunk(BaseModel):
    """Representa un fragmento de texto optimizado para embeddings y búsqueda vectorial."""
    chunk_id: str = Field(..., description="Identificador único del chunk")
    content: str = Field(..., description="Contenido de texto del fragmento")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadatos contextuales heredados (fuente, página, sección, posición)")
    embedding: Optional[List[float]] = Field(default=None, description="Vector de embedding generado por Gemini (768/3072 dimensiones)")


class SectionSummary(BaseModel):
    """Resumen liviano de una sección para respuestas de API."""
    title: str
    level: int
    char_count: int


class DocumentUploadResponse(BaseModel):
    """Respuesta estructurada del endpoint de subida y validación de documentos."""
    filename: str
    extension: str
    status: str
    message: str
    total_sections: Optional[int] = None
    total_chunks: Optional[int] = None
    embedding_dimension: Optional[int] = None
    sections_summary: Optional[List[SectionSummary]] = None


class QueryRequest(BaseModel):
    """Solicitud de consulta o pregunta del usuario."""
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Pregunta del usuario para consultar sobre los documentos indexados",
        examples=["¿Cómo funciona la arquitectura de RAGIA?"]
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Cantidad máxima de fragmentos relevantes a recuperar de ChromaDB"
    )


class RetrievedChunk(BaseModel):
    """Fragmento relevante recuperado de la base vectorial con metadatos de cita."""
    chunk_id: str = Field(..., description="ID único del chunk")
    content: str = Field(..., description="Texto del fragmento recuperado")
    similarity_score: float = Field(..., description="Score de similitud coseno (0 a 1)")
    source: str = Field(..., description="Archivo de origen del documento")
    page: Optional[int] = Field(default=None, description="Página del documento (si es PDF)")
    section_title: Optional[str] = Field(default=None, description="Título de la sección (si es Markdown o identificado)")
    char_count: int = Field(..., description="Cantidad de caracteres del fragmento")


class QueryRetrievalResponse(BaseModel):
    """Respuesta de recuperación semántica con los Top-K fragmentos más relevantes."""
    question: str
    total_retrieved: int
    retrieved_chunks: List[RetrievedChunk]
    message: str


class PromptPreviewResponse(BaseModel):
    """Respuesta estructurada para la vista previa del prompt con contención y citas."""
    question: str = Field(..., description="Pregunta formulada por el usuario")
    system_instruction: str = Field(..., description="Directivas del sistema de contención estricta y formato de citas")
    formatted_prompt: str = Field(..., description="Prompt dinámico completo con el contexto inyectado y la pregunta")
    total_sources: int = Field(..., description="Cantidad de fuentes o fragmentos incluidos en el contexto")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Lista de fuentes y metadatos de los fragmentos inyectados")


class RAGQueryResponse(BaseModel):
    """Respuesta final fundamentada generada por Gemini Flash con citas a fuentes."""
    question: str = Field(..., description="Pregunta formulada por el usuario")
    answer: str = Field(..., description="Respuesta final fundamentada generada por el LLM o respuesta de contención")
    total_sources: int = Field(..., description="Cantidad de fragmentos contextuales recuperados de ChromaDB")
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Lista de fragmentos recuperados con metadatos de cita")
    model_used: str = Field(..., description="Modelo de lenguaje utilizado para generar la respuesta")


class DocumentListItem(BaseModel):
    """Información agregada de un documento indexado en ChromaDB."""
    filename: str = Field(..., description="Nombre del archivo almacenado")
    total_chunks: int = Field(..., description="Cantidad total de fragmentos indexados")
    total_pages: Optional[int] = Field(default=None, description="Total de páginas procesadas (si es PDF)")
    total_sections: Optional[int] = Field(default=None, description="Total de secciones identificadas (si es Markdown)")


class DocumentDeleteResponse(BaseModel):
    """Resultado de la operación de eliminación de un documento en ChromaDB."""
    filename: str = Field(..., description="Nombre del archivo eliminado")
    deleted_chunks: int = Field(..., description="Cantidad de fragmentos y vectores eliminados")
    message: str = Field(..., description="Mensaje descriptivo del resultado")
