import os
import shutil
import tempfile
import unittest

from app.models.schemas import DocumentChunk
from app.services.vector_store import ChromaVectorStore


class TestChromaVectorStore(unittest.TestCase):
    """Suite de pruebas unitarias para ChromaVectorStore con almacenamiento temporal aislado."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = ChromaVectorStore(
            persist_dir=self.temp_dir,
            collection_name="test_collection"
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Verifica que el almacén se inicialice y cree el directorio."""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertEqual(self.store.count(), 0)

    def test_add_chunks_and_count(self):
        """Verifica la inserción de fragmentos con vectores y metadatos."""
        chunks = [
            DocumentChunk(
                chunk_id="doc1.md_chunk_0",
                content="Contenido sobre inteligencia artificial y redes neuronales.",
                metadata={"source": "doc1.md", "section_title": "Intro", "page": 1},
                embedding=[0.1] * 768
            ),
            DocumentChunk(
                chunk_id="doc1.md_chunk_1",
                content="Contenido sobre bases de datos vectoriales y ChromaDB.",
                metadata={"source": "doc1.md", "section_title": "DBs", "page": 1},
                embedding=[0.2] * 768
            ),
        ]

        inserted = self.store.add_chunks(chunks)
        self.assertEqual(inserted, 2)
        self.assertEqual(self.store.count(), 2)

    def test_upsert_idempotency(self):
        """Verifica que reinsertar chunks actualice sin duplicar registros."""
        chunks = [
            DocumentChunk(
                chunk_id="doc1.md_chunk_0",
                content="Versión inicial",
                metadata={"source": "doc1.md"},
                embedding=[0.1] * 768
            )
        ]
        self.store.add_chunks(chunks)
        self.assertEqual(self.store.count(), 1)

        # Reinsertar con mismo ID pero contenido actualizado
        updated_chunks = [
            DocumentChunk(
                chunk_id="doc1.md_chunk_0",
                content="Versión actualizada",
                metadata={"source": "doc1.md"},
                embedding=[0.15] * 768
            )
        ]
        self.store.add_chunks(updated_chunks)
        self.assertEqual(self.store.count(), 1)

    def test_list_documents(self):
        """Verifica el agrupamiento de documentos por archivo de origen."""
        chunks = [
            DocumentChunk(
                chunk_id="manual.pdf_chunk_0",
                content="Texto página 1",
                metadata={"source": "manual.pdf", "page": 1},
                embedding=[0.1] * 768
            ),
            DocumentChunk(
                chunk_id="manual.pdf_chunk_1",
                content="Texto página 2",
                metadata={"source": "manual.pdf", "page": 2},
                embedding=[0.2] * 768
            ),
            DocumentChunk(
                chunk_id="notas.md_chunk_0",
                content="Notas de diseño",
                metadata={"source": "notas.md", "section_title": "Diseño"},
                embedding=[0.3] * 768
            ),
        ]
        self.store.add_chunks(chunks)

        docs = self.store.list_documents()
        self.assertEqual(len(docs), 2)

        pdf_doc = next(d for d in docs if d["filename"] == "manual.pdf")
        self.assertEqual(pdf_doc["total_chunks"], 2)
        self.assertEqual(pdf_doc["total_pages"], 2)

        md_doc = next(d for d in docs if d["filename"] == "notas.md")
        self.assertEqual(md_doc["total_chunks"], 1)
        self.assertEqual(md_doc["total_sections"], 1)

    def test_delete_by_source(self):
        """Verifica la eliminación de vectores asociados a un archivo."""
        chunks = [
            DocumentChunk(
                chunk_id="docA.md_chunk_0",
                content="Texto A",
                metadata={"source": "docA.md"},
                embedding=[0.1] * 768
            ),
            DocumentChunk(
                chunk_id="docB.md_chunk_0",
                content="Texto B",
                metadata={"source": "docB.md"},
                embedding=[0.2] * 768
            ),
        ]
        self.store.add_chunks(chunks)
        self.assertEqual(self.store.count(), 2)

        deleted = self.store.delete_by_source("docA.md")
        self.assertEqual(deleted, 1)
        self.assertEqual(self.store.count(), 1)

        docs = self.store.list_documents()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["filename"], "docB.md")

    def test_similarity_search(self):
        """Verifica la búsqueda vectorial por similitud coseno."""
        vec1 = [1.0] + [0.0] * 767
        vec2 = [0.0] + [1.0] + [0.0] * 766

        chunks = [
            DocumentChunk(
                chunk_id="hit_1",
                content="Documento sobre Python y FastAPI",
                metadata={"source": "python.md"},
                embedding=vec1
            ),
            DocumentChunk(
                chunk_id="hit_2",
                content="Documento sobre recetas de cocina",
                metadata={"source": "cocina.md"},
                embedding=vec2
            ),
        ]
        self.store.add_chunks(chunks)

        # Consulta con vector idéntico a vec1
        results = self.store.similarity_search(query_vector=vec1, top_k=2)

        self.assertEqual(len(results), 2)
        # El primer resultado debe ser el más similar (score cercano a 1.0)
        self.assertEqual(results[0]["chunk_id"], "hit_1")
        self.assertGreaterEqual(results[0]["similarity_score"], 0.99)
        self.assertEqual(results[0]["content"], "Documento sobre Python y FastAPI")


if __name__ == "__main__":
    unittest.main()
