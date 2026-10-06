import os
from typing import Any, Dict, List, Optional
import chromadb
from app.core.config import settings
from app.models.schemas import DocumentChunk


class ChromaVectorStore:
    """
    Servicio de almacenamiento e indexación vectorial persistente utilizando ChromaDB.
    Guarda fragmentos de texto, metadatos y vectores densos en disco local.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "ragia_documents"
    ):
        """
        Inicializa el cliente de almacenamiento vectorial persistente.

        Args:
            persist_dir (Optional[str]): Directorio de persistencia en disco (por defecto de settings).
            collection_name (str): Nombre de la colección en ChromaDB.
        """
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        self.collection_name = collection_name
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """Inicializa o retorna la instancia del cliente persistente de ChromaDB."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    @property
    def collection(self):
        """Inicializa o retorna la colección con espacio métrico de similitud coseno."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB solo admite primitivos (str, int, float, bool) en los metadatos."""
        sanitized = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Inserta o actualiza fragmentos con sus vectores y metadatos en ChromaDB.
        Utiliza operación idempotente upsert.

        Args:
            chunks (List[DocumentChunk]): Fragmentos con vector de embedding asignado.

        Returns:
            int: Cantidad de fragmentos indexados exitosamente.
        """
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            if not chunk.embedding:
                continue

            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            embeddings.append(chunk.embedding)
            metadatas.append(self._sanitize_metadata(chunk.metadata))

        if not ids:
            return 0

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return len(ids)

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Lista los documentos indexados en ChromaDB agrupando por archivo de origen.

        Returns:
            List[Dict[str, Any]]: Lista de documentos con conteo de chunks y páginas/secciones.
        """
        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []

        docs_map: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            source = meta.get("source", "desconocido")
            if source not in docs_map:
                docs_map[source] = {
                    "filename": source,
                    "total_chunks": 0,
                    "pages": set(),
                    "sections": set(),
                }
            docs_map[source]["total_chunks"] += 1
            if "page" in meta:
                docs_map[source]["pages"].add(meta["page"])
            if "section_title" in meta:
                docs_map[source]["sections"].add(meta["section_title"])

        doc_list = []
        for source, info in docs_map.items():
            doc_item = {
                "filename": info["filename"],
                "total_chunks": info["total_chunks"],
            }
            if info["pages"]:
                doc_item["total_pages"] = len(info["pages"])
            if info["sections"]:
                doc_item["total_sections"] = len(info["sections"])
            doc_list.append(doc_item)

        return sorted(doc_list, key=lambda d: d["filename"])

    def delete_by_source(self, filename: str) -> int:
        """
        Elimina de ChromaDB todos los fragmentos y vectores de un archivo específico.

        Args:
            filename (str): Nombre del archivo a eliminar.

        Returns:
            int: Cantidad de fragmentos eliminados.
        """
        results = self.collection.get(
            where={"source": filename},
            include=["metadatas"]
        )
        ids_to_delete = results.get("ids") or []
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def count(self) -> int:
        """Retorna el total de vectores indexados en la colección."""
        return self.collection.count()

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda semántica de los Top-K fragmentos más cercanos por distancia coseno.

        Args:
            query_vector (List[float]): Vector de la consulta de búsqueda.
            top_k (int): Cantidad máxima de resultados a retornar.

        Returns:
            List[Dict[str, Any]]: Lista de coincidencias con contenido, metadatos y score de similitud.
        """
        if not query_vector or self.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        if results and "ids" in results and results["ids"]:
            ids = results["ids"][0]
            docs = results["documents"][0] if results.get("documents") else []
            metas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for i in range(len(ids)):
                distance = distances[i] if i < len(distances) else 0.0
                # En espacio coseno, score aproximado de similitud entre 0 y 1
                similarity_score = round(max(0.0, 1.0 - distance), 4)

                hits.append({
                    "chunk_id": ids[i],
                    "content": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "similarity_score": similarity_score,
                })

        return hits
