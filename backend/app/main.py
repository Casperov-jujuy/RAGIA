from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints_documents import router as documents_router

app = FastAPI(
    title="RAGIA API",
    description="API Backend RAG con Google Gemini, ChromaDB y soporte exclusivo para .md y .pdf",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir peticiones desde el futuro frontend o clientes locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de la API
app.include_router(documents_router, prefix="/api/v1/documents", tags=["Documentos"])


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Bienvenido a RAGIA API",
        "docs": "/docs",
        "health": "/health",
        "version": "0.1.0"
    }


@app.get("/health", tags=["Salud"])
async def health_check():
    return {
        "status": "healthy",
        "service": "ragia-backend",
        "version": "0.1.0"
    }
