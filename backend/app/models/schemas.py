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
    sections_summary: Optional[List[SectionSummary]] = None
