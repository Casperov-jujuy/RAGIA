# 🧠 RAGIA — Backend RAG (.md & .pdf) con Google Gemini y Docker

**RAGIA** es un sistema backend de **Generación Aumentada por Recuperación** (*Retrieval-Augmented Generation* - **RAG**) diseñado para ingestar, indexar y consultar documentos personales en formatos **Markdown (`.md`)** y **PDF (`.pdf`)**.

El proyecto está diseñado bajo una arquitectura modular (preparada para monorepo con backend y frontend independientes), impulsado por **FastAPI**, **Google Gemini API** (para generación y embeddings), **ChromaDB** para almacenamiento vectorial persistente y completamente contenerizado con **Docker Compose**.

---

## 🎯 Características Principales

- 📄 **Enfoque Estricto en Texto (`.md` y `.pdf`):**
  - Validador estricto a nivel de API: rechaza inmediatamente cualquier archivo que no sea `.md` o `.pdf` (imágenes, ejecutables, formatos de Office, etc.).
  - Parser de **Markdown** con preservación de estructura de títulos y encabezados *(en desarrollo)*.
  - Parser de **PDF** enfocado en extracción de texto limpio página por página *(en desarrollo)*.
- 🤖 **Potenciado por Google Gemini:**
  - Embeddings de alta dimensionalidad con `text-embedding-004`.
  - Respuestas contextualizadas y libres de alucinaciones con modelos Gemini Flash (`gemini-1.5-flash` / `gemini-2.0-flash`).
- 💾 **Persistencia Vectorial (ChromaDB):**
  - Almacén de vectores embebido que persiste sus datos en un volumen local montado (`./backend/data/chroma_db`), evitando servicios externos pesados.
- 🐳 **Contenerizado con Docker Compose:**
  - Despliegue en un solo comando con recarga en caliente (*hot-reload*) para desarrollo ágil y volúmenes montados.
- 📮 **Listo para Postman y Swagger UI:**
  - Endpoints REST para subida multipart de archivos y consultas RAG, testeables directamente vía Postman o en `http://localhost:8000/docs`.
- 🧪 **Suite de Pruebas Automatizadas:**
  - Tests unitarios y de integración HTTP (con `TestClient`) para asegurar la solidez de cada paso implementado.

---

## 🏗️ Flujo de Trabajo (Pipeline)

```
       [ Archivo .md o .pdf ]
                 │
                 ▼
       [ 1. Validador Estricto ] ──> Rechaza imágenes u otros formatos (HTTP 400)
                 │                   [✅ IMPLEMENTADO]
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
       [ 5. ChromaDB Store ] ────────> Persistencia en ./backend/data/chroma_db
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
| **Contenerización** | Docker & Docker Compose | Contenedores portables con recarga en caliente |
| **Testing** | `unittest` / `pytest` + `TestClient` | Cobertura de validadores y endpoints HTTP |

---

## 📂 Estructura del Proyecto

```text
RAGIA/
├── docker-compose.yml            # Orquestación Docker (servicio backend + volúmenes)
├── .gitignore                    # Reglas globales de exclusión para Git
├── README.md                     # Documentación principal del proyecto
│
├── backend/                      # Módulo aislado del Backend
│   ├── Dockerfile                # Imagen optimizada Python 3.11-slim
│   ├── requirements.txt          # Dependencias de Python (FastAPI, ChromaDB, Gemini...)
│   ├── .env.example              # Plantilla de variables de entorno
│   ├── .env                      # Variables locales y API Keys (ignorado por Git)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # Punto de entrada FastAPI, CORS y rutas base
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       └── endpoints_documents.py # Endpoints de documentos (/upload)
│   │   └── ingestion/
│   │       ├── __init__.py
│   │       └── validator.py      # Validador estricto (.md y .pdf)
│   ├── data/
│   │   ├── chroma_db/            # Almacén persistente de ChromaDB (ignorado por Git)
│   │   ├── uploads/              # Almacén temporal de subidas (ignorado por Git)
│   │   └── sample_docs/          # Documentos de prueba de ejemplo
│   └── tests/
│       ├── __init__.py
│       ├── test_validator.py     # Tests unitarios del validador de archivos
│       └── test_api_upload.py    # Tests de integración del endpoint de subida
│
└── frontend/                     # (Espacio reservado para futura UI)
```

---

## 🚀 Puesta en Marcha Rápida con Docker

### 1. Prerrequisitos
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución.
- Una API Key de **Google Gemini** ([Obtener en Google AI Studio](https://aistudio.google.com/)).

### 2. Configurar variables de entorno
Crea tu archivo `.env` a partir de la plantilla dentro de `backend/`:

```bash
cp backend/.env.example backend/.env
```

Edita `backend/.env` y coloca tu API Key:
```env
GEMINI_API_KEY=tu_api_key_de_gemini_aqui
CHROMA_PERSIST_DIR=/app/data/chroma_db
UPLOAD_DIR=/app/data/uploads
```

### 3. Iniciar el servicio con Docker Compose
Desde la raíz del proyecto:

```bash
docker compose up --build
```
*(Para ejecutarlo en segundo plano agrega la bandera `-d`: `docker compose up --build -d`)*

La API estará lista y disponible en:
- **Bienvenida:** `http://localhost:8000/`
- **Salud del servicio:** `http://localhost:8000/health`
- **Documentación Interactiva (Swagger UI):** `http://localhost:8000/docs`
- **Documentación Alternativa (ReDoc):** `http://localhost:8000/redoc`

---

## 📮 Estado de Endpoints (Pruebas con Postman y Swagger)

| Método | Endpoint | Estado | Descripción |
| :--- | :--- | :---: | :--- |
| `GET` | `/` | ✅ Listo | Mensaje de bienvenida y enlaces a documentación |
| `GET` | `/health` | ✅ Listo | Comprobación de estado y salud del backend |
| `POST` | `/api/v1/documents/upload` | ✅ Listo | Valida y recibe archivos `multipart` (**solo `.md` y `.pdf`**, campo: `file`) |
| `GET` | `/api/v1/documents` | ⏳ Pendiente | Listado de documentos indexados en ChromaDB |
| `DELETE` | `/api/v1/documents/{filename}` | ⏳ Pendiente | Eliminación de documento y vectores asociados |
| `POST` | `/api/v1/rag/query` | ⏳ Pendiente | Consulta RAG con contexto y citas de fuentes |

### 🧪 Probar Validación en Postman:
1. Petición: `POST http://localhost:8000/api/v1/documents/upload`
2. Pestaña **Body** ➔ **form-data** ➔ Key: `file` (tipo File).
3. **Casos válidos (`.md`, `.pdf`):** Retorna `200 OK` con estado `"approved"`.
4. **Casos inválidos (`.png`, `.jpg`, `.txt`, `.docx`, etc.):** Retorna `400 Bad Request` con mensaje explicativo de rechazo inmediato.

---

## 🧪 Ejecución de Pruebas Automatizadas

Para correr la suite de pruebas unitarias e integradas:

```bash
cd backend
python -m unittest discover tests
```

---

## 🗺️ Roadmap de Desarrollo

- [x] **Fase 0:** Definición de arquitectura, pipeline conceptual y especificación técnica.
- [x] **Fase 1:** Dockerización, FastAPI base (`/health`, `/`), entorno de dependencias y CORS.
- [ ] **Fase 2:** La entrada de datos y Parsers especializados.
  - [x] **Paso 2.1:** Validador estricto de extensiones (`.md` y `.pdf`) y endpoint `POST /upload`.
  - [ ] **Paso 2.2:** Parser de Markdown (`.md`) preservando jerarquía de títulos.
  - [ ] **Paso 2.3:** Parser de PDF (`.pdf`) con extracción de texto por páginas.
- [ ] **Fase 3:** Fragmentación (Chunking) inteligente y hashing determinista de chunks.
- [ ] **Fase 4:** Integración con Google Gemini (`text-embedding-004`) y almacenamiento persistente en ChromaDB.
- [ ] **Fase 5:** Endpoint de consulta RAG (`POST /query`) con prompt de contención y citas de fuentes.
- [ ] **Fase 6:** Pruebas de integración completa y benchmarks de fidelidad.

---

## 👤 Autor

- **Casperov-jujuy** — [GitHub](https://github.com/Casperov-jujuy)
