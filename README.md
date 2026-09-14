# 🧠 RAGIA

**RAGIA** es un proyecto personal enfocado en el diseño, experimentación e implementación de un sistema de **Generación Aumentada por Recuperación** (*Retrieval-Augmented Generation* - **RAG**).

El objetivo principal es construir una arquitectura modular y flexible que permita conectar fuentes de información personal o conocimiento privado (documentos, notas, código, manuales) con modelos de lenguaje (LLMs), logrando respuestas precisas, contextualizadas y fundamentadas con citas de origen.

---

## 🚀 Visión del Proyecto

El desarrollo se enfoca en comprender a fondo y optimizar cada etapa del ciclo de vida de un sistema RAG antes de comprometerse con un framework o lenguaje definitivo:

- **Flexibilidad de Stack:** Diseñado para ser agnóstico al inicio (evaluando opciones como Python con LangChain / LlamaIndex, o arquitecturas ligeras en TypeScript / Go).
- **Control de Contexto:** Minimizar alucinaciones proporcionando al LLM únicamente la información relevante extraída de las fuentes de datos.
- **Soporte Híbrido:** Capacidad de trabajar tanto con modelos locales (*open-source* vía Ollama, LM Studio o HuggingFace) como con APIs en la nube (OpenAI, Google Gemini, Anthropic).

---

## 🏗️ Arquitectura y Flujo de Trabajo (Pipeline)

```
[ Documentos / Datos ]
         │
         ▼
[ 1. Ingesta y Parsing ] ──> Limpieza, extracción y normalización
         │
         ▼
[ 2. Chunking ] ───────────> Estrategias de fragmentación de texto
         │
         ▼
[ 3. Embeddings ] ─────────> Vectorización del texto
         │
         ▼
[ 4. Vector Store ] ───────> Almacenamiento e indexación de vectores
         │
         ▲
 ┌───────┴───────────────────────────────┐
 │ [ 5. Query / Búsqueda Semántica ]     │
 │        + Reranking (opcional)         │
 └───────┬───────────────────────────────┘
         │
         ▼
[ 6. Prompt Augmentation ] ──> Construcción del prompt con contexto
         │
         ▼
[ 7. LLM & Generación ] ─────> Respuesta fundamentada con citas
```

### Componentes Clave:
1. **Ingesta y Fragmentación (Chunking):** Estrategias fijas, semánticas y recursivas para preservar el significado de los textos.
2. **Representación Vectorial (Embeddings):** Generación de representaciones vectoriales densas para captura de similitud semántica.
3. **Base de Datos Vectorial (Vector Database):** Almacén optimizado para búsquedas por similitud (coseno, producto punto, euclidiana).
4. **Recuperación (Retrieval & Reranking):** Búsqueda semántica, híbrida (denso + BM25) y reordenamiento de relevancia.
5. **Generador (LLM):** Síntesis de respuestas orientadas a preguntas sobre el material ingerido.

---

## 🛠️ Tecnologías Bajo Evaluación

| Capa | Alternativas en Estudio |
| :--- | :--- |
| **Lenguaje / Backend** | Python (FastAPI) · TypeScript (Node.js/Bun) |
| **Frameworks RAG** | LangChain · LlamaIndex · Implementación *from-scratch* (sin dependencias complejas) |
| **Vector Stores** | ChromaDB · FAISS · Qdrant · LanceDB · PGVector |
| **Modelos de Lenguaje** | Ollama (Llama 3, Mistral, Gemma) · Google Gemini API · OpenAI API |
| **Modelos de Embeddings** | `bge-m3`, `nomic-embed-text`, OpenAI `text-embedding-3`, Gemini Embeddings |
| **Interfaz (Futura)** | CLI · Web UI (Streamlit, Gradio o frontend en React/Next.js) |

---

## 🗺️ Roadmap de Desarrollo

- [x] **Fase 0:** Definición del concepto, objetivos y arquitectura base.
- [ ] **Fase 1:** Prototipo funcional mínimo (CLI) con ingesta de archivos locales (Markdown/PDF/TXT).
- [ ] **Fase 2:** Pruebas de diferentes estrategias de chunking y modelos de embeddings.
- [ ] **Fase 3:** Incorporación de búsqueda híbrida y técnicas de reranking.
- [ ] **Fase 4:** Evaluación de calidad de recuperación y respuestas (métricas de fidelidad y relevancia).
- [ ] **Fase 5:** Desarrollo de interfaz gráfica o API REST para consumo externo.

---

## 📂 Estructura Tentativa del Repositorio

```text
RAGIA/
├── README.md               # Documentación general del proyecto
├── docs/                   # Notas de diseño, arquitectura y experimentos
├── data/                   # Documentos y fuentes de prueba (raw / processed)
├── src/                    # Código fuente (a definir según stack elegido)
│   ├── ingestion/          # Cargadores y procesadores de texto
│   ├── embeddings/         # Clientes y generación de embeddings
│   ├── storage/            # Conexión con Vector DB
│   └── retrieval/          # Búsqueda y reranking
│   └── generation/         # Orquestación con el LLM
└── tests/                  # Pruebas unitarias y benchmarks
```

---

## 👤 Autor

- **Casperov-jujuy** — [GitHub](https://github.com/Casperov-jujuy)
