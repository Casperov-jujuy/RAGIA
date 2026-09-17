from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
