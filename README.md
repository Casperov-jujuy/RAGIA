# 🧠 RAGIA — Backend RAG (.md & .pdf) con Google Gemini y Docker

**RAGIA** es un sistema backend de **Generación Aumentada por Recuperación** (*Retrieval-Augmented Generation* - **RAG**) diseñado para ingestar, indexar y consultar documentos personales en formatos **Markdown (`.md`)** y **PDF (`.pdf`)**.

El proyecto utiliza **FastAPI**, **Google Gemini API** (para generación y embeddings), **ChromaDB** para el almacenamiento vectorial persistente y está completamente contenerizado con **Docker Compose**.

---

## 🎯 Características Principales

- 📄 **Enfoque Estricto en Texto (`.md` y `.pdf`):**
  - Parser de **Markdown** con preservación de estructura de títulos y encabezados.
  - Parser de **PDF** enfocado en extracción de texto limpio página por página (sin procesamiento de imágenes ni capas binarias innecesarias).
  - Filtro estricto que rechaza cualquier otro formato de archivo no permitido.
- 🤖 **Potenciado por Google Gemini:**
  - Embeddings de alta dimensionalidad con `text-embedding-004`.
  - Respuestas contextualizadas y libres de alucinaciones con modelos Gemini Flash (`gemini-1.5-flash` / `gemini-2.0-flash`).
- 💾 **Persistencia Vectorial (ChromaDB):**
  - Almacén de vectores embebido que persiste sus datos en un volumen local montado (`./data/chroma_db`), evitando la necesidad de servicios externos pesados.
- 🐳 **Contenerizado con Docker Compose:**
  - Despliegue en un solo comando con recarga en caliente (*hot-reload*) para desarrollo ágil.
- 📮 **Listo para Postman y Swagger UI:**
  - Endpoints REST para subida multipart de archivos y consultas RAG, testeables directamente vía Postman o en `http://localhost:8000/docs`.
- 📌 **Citas y Fuentes Verificables:**
  - Cada respuesta generada incluye las fuentes exactas de donde se extrajo la información (nombre de archivo, página o sección y fragmento original).

---

## 🏗️ Flujo de Trabajo (Pipeline)

```
       [ Archivo .md o .pdf ]
                 │
                 ▼
       [ 1. Validador de Formato ] ──> Rechaza cualquier otro formato
                 │
                 ▼
       [ 2. Parser Especializado ]
        ├── .md  ──> Estructura jerárquica de títulos
        └── .pdf ──> Extracción de texto plano por página
                 │
                 ▼
       [ 3. Chunking & Hashing ] ────> Fragmentación inteligente + ID único
                 │
                 ▼
       [ 4. Gemini Embeddings ] ─────> Vectorización (text-embedding-004)
                 │
                 ▼
       [ 5. ChromaDB Store ] ────────> Persistencia en ./data/chroma_db
                 │
                 ▲
 ┌───────────────┴──────────────────────────────┐
 │ [ 6. Query / Consulta Semántica ]            │
 │        Top-K fragmentos más relevantes       │
 └───────────────┬──────────────────────────────┘
                 │
                 ▼
 [ 7. Prompt Augmentation ] ────────────────────> Contexto inyectado al prompt
                 │
                 ▼
 [ 8. Google Gemini LLM ] ──────────────────────> Respuesta fundamentada con citas
```

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología | Descripción |
| :--- | :--- | :--- |
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) | API REST asíncrona, rápida y autogestionada con OpenAPI |
| **Lenguaje** | Python 3.11 | Entorno principal de desarrollo |
| **Modelos & LLM** | [Google Gemini API](https://ai.google.dev/) | `text-embedding-004` (vectores) y `gemini-1.5-flash` (generación) |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Base de datos vectorial persistida en volumen local |
| **Extracción de Texto** | `pypdf` + Parser nativo MD | Lectura limpia y eficiente sin OCR pesado |
| **Entorno & Despliegue**| Docker & Docker Compose | Contenedores portables con volúmenes montados |

---

## 📂 Estructura del Proyecto

```text
RAGIA/
├── docker-compose.yml        # Orquestación de Docker (backend + volúmenes)
├── Dockerfile                # Imagen optimizada Python 3.11-slim
├── requirements.txt          # Dependencias del proyecto
├── .env.example              # Plantilla de variables de entorno
├── .gitignore                # Reglas de exclusión para Git
├── README.md                 # Documentación del proyecto
├── data/
│   ├── chroma_db/            # Directorio persistente de ChromaDB (ignorado en git)
│   ├── uploads/              # Archivos procesados temporalmente (ignorado en git)
│   └── sample_docs/          # Documentos de prueba (.md y .pdf)
└── app/
    ├── main.py               # Punto de entrada de FastAPI y rutas
    ├── core/
    │   ├── config.py         # Configuración y lectura de variables (.env)
    │   └── logging.py        # Configuración de logs
    ├── models/
    │   └── schemas.py        # Modelos Pydantic (Request / Response)
    ├── ingestion/
    │   ├── validator.py      # Filtro estricto de extensiones (.md, .pdf)
    │   ├── md_parser.py      # Extractor de Markdown
    │   ├── pdf_parser.py     # Extractor de texto PDF
    │   └── chunker.py        # Estrategia de fragmentación con metadatos
    ├── services/
    │   ├── gemini_service.py # Interacción con la API de Google Gemini
    │   └── vector_store.py   # Operaciones sobre ChromaDB
    └── api/
        └── v1/
            ├── endpoints_documents.py  # Endpoints de subida y gestión
            └── endpoints_rag.py        # Endpoint de consulta RAG
```

---

## 🚀 Puesta en Marcha Rápida con Docker

### 1. Prerrequisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución.
- Una API Key de **Google Gemini** ([Obtener aquí](https://aistudio.google.com/)).

### 2. Configurar variables de entorno
Crea tu archivo `.env` a partir de la plantilla:

```bash
cp .env.example .env
```

Edita `.env` y coloca tu API Key:
```env
GEMINI_API_KEY=tu_api_key_aqui
CHROMA_PERSIST_DIR=/app/data/chroma_db
UPLOAD_DIR=/app/data/uploads
```

### 3. Iniciar el servicio con Docker Compose
```bash
docker compose up --build
```
La API estará lista y disponible en:
- **API Base:** `http://localhost:8000`
- **Documentación Interactiva (Swagger):** `http://localhost:8000/docs`

---

## 📮 Endpoints para Probar en Postman

| Método | Endpoint | Tipo | Descripción |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | — | Comprobación de estado del servicio |
| `POST` | `/api/v1/documents/upload` | `multipart/form-data` | Sube e indexa un archivo `.md` o `.pdf` (campo: `file`) |
| `GET` | `/api/v1/documents` | — | Lista todos los documentos actualmente indexados |
| `DELETE` | `/api/v1/documents/{filename}` | — | Elimina un documento y todos sus vectores de ChromaDB |
| `POST` | `/api/v1/rag/query` | `application/json` | Realiza una pregunta contextualizada sobre los documentos |

### Ejemplo de Payload para `/api/v1/rag/query`:
```json
{
  "question": "¿Cuáles son los puntos clave mencionados en el informe?"
}
```

### Ejemplo de Respuesta:
```json
{
  "answer": "De acuerdo con el documento...",
  "sources": [
    {
      "filename": "informe_tecnico.pdf",
      "page": 3,
      "snippet": "Los puntos clave a considerar en el despliegue son...",
      "score": 0.89
    }
  ]
}
```

---

## 🗺️ Roadmap de Desarrollo

- [x] **Fase 0:** Definición de arquitectura, pipeline y especificación técnica.
- [ ] **Fase 1:** Configuración de Docker, esqueleto de FastAPI y cliente Gemini.
- [ ] **Fase 2:** Implementación de parsers especializados para `.md` y `.pdf` con filtro estricto.
- [ ] **Fase 3:** Sistema de chunking y almacenamiento vectorial persistente con ChromaDB.
- [ ] **Fase 4:** Endpoints de subida e indexación testeables con Postman.
- [ ] **Fase 5:** Endpoint de consulta RAG con prompt de contención y citas de fuentes.
- [ ] **Fase 6:** Pruebas de integración y validación de respuestas.

---

## 👤 Autor

- **Casperov-jujuy** — [GitHub](https://github.com/Casperov-jujuy)
