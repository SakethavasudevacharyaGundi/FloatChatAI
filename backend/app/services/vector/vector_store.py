import chromadb

from pathlib import Path


class VectorStore:

    def __init__(self):

        base_dir = (
            Path(__file__)
            .resolve()
            .parents[4]
        )

        chroma_path = (
            base_dir
            / "chroma_db"
        )

        print(
            f"CHROMA PATH: {chroma_path}"
        )

        self.client = (
            chromadb.PersistentClient(
                path=str(chroma_path)
            )
        )

        self.collection = (
            self.client.get_or_create_collection(
                "ocean_docs"
            )
        )

        print(
            "VECTOR COUNT:",
            self.collection.count()
        )

    def add_documents(
        self,
        ids,
        texts,
        embeddings,
        metadatas
    ):

        self.collection.add(

            ids=ids,

            documents=texts,

            embeddings=embeddings,

            metadatas=metadatas
        )

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        return (
        self.collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=top_k,

        include=[
            "documents",
            "metadatas",
            "distances"
                ]
            )
        )