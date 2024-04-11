from pathlib import Path

import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

from typing import cast

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Qdrant
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from langchain.vectorstores import VectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document

from ..utils import copy_files_to_dest, delete_files_from_dest


class ModelPersistentContext:
    """
    Object to manage the non-parametric parameters of the model (i.e, the embeddings of stored context).
    """
    db: VectorStore | None = None
    
    def __init__(
        self,
        collection="model_persistent_context",
        embedder="sentence-transformers/all-mpnet-base-v2",
        address="http://habitllm-persistent-context-store-1:6333",
        port=6333,
        file_dir="/persistent/model_context/files",
        logger=logging.getLogger(__name__),
    ) -> None:
        self.logger = logger
        self.collection = collection
        self.address = address
        self.port = port
        self.file_dir = file_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=embedder)
        Path(self.file_dir).mkdir(exist_ok=True)
        logger.info(f"Initializing connection to qdrant server @ {address}:{port} ...")
        self.client = QdrantClient(self.address, port=self.port)
        logger.info("Connection established.")
        logger.info(f"Checking if collection {self.collection} exists...")
        if self.client.collection_exists(collection_name=collection):
            self.db = Qdrant(
                self.client, self.collection, embeddings=self.embeddings
            )
            vectors_count = self.client.get_collection(self.collection).vectors_count
            logger.info(f"Collection exists with {vectors_count} vectors")
        else:
            logger.info("No collection found. Will be created during initial document ingestion in downtime.")
        # self.db: VectorStore = Qdrant.from_documents(
        #     [], self.embeddings, url=self.address, port=self.port
        # )
        logger.info("Connection established.")

    def add_files_to_vector_store(self, files: list[str]) -> None:
        """Adds files to the model's persistent context.

        Stores files in persistent context folder and ingests their embeddings into the db.

        Args:
            files: list of filepaths to add to persisted model context.
        """
        self.logger.info("Copying new documents to persistent context dir.")
        updated_files = copy_files_to_dest(self.file_dir, files)

        self.logger.info("Ingesting new documents into model persistent context.")
        for file in updated_files:
            self.logger.info(f"Storing embeddings for {file}")
            loader: TextLoader = TextLoader(file_path=file)
            documents: list[Document] = loader.load()
            text_splitter: CharacterTextSplitter = CharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0
            )
            docs = text_splitter.split_documents(documents)
            db_just_created = False
            if self.db is None:
                if self.client.collection_exists(self.collection):
                    self.logger.info("Collection exists, but langchain did not create VectorStore object. Retrying...")
                    try:
                        self.db: VectorStore = Qdrant(
                            self.client, self.collection, embeddings=self.embeddings
                        )
                        self.logger.info(f"VectorStore, now connected to {self.collection}.")
                    except Exception as e:
                        raise Exception("Error [habitllm.model_persistant_context_system]: something went wrong externally.") from e
                else:
                    self.logger.info(f"Creating collection {self.collection}")
                    self.db: VectorStore = Qdrant.from_documents(docs, self.embeddings, url=self.address, port=self.port, collection_name=self.collection)
                    db_just_created = True
            if not db_just_created:
                ids = cast(VectorStore, self.db).add_documents(docs)
                self.logger.debug(f"{file} was ingested with ids: {ids}.")
        self.logger.info("Document ingestion complete!")

    def delete_files_from_vector_store(self, files: list[str]) -> None:
        """Deltes files and their embeddings from the persistent context.

        Args:
            file: List of filepaths to delete.
        """
        # deleted_files = delete_files_from_dest(self.file_dir, files)
        # Qdrant.delete
        pass

    def delete_concept_from_vector_store(self, concept_query: list[str]) -> None:
        """Deletes related embeddings from db.

        Uses similarity with concept_query to retrieve related vector embedding ids,
        then deletes them from the db.
        TODO: If no references to a file remains deletes that as well.

        Args:
            concept_query: A text query relating to a concept wished to be deleted.
        """
        pass

    def reindex_vector_store(self) -> None:
        """Reindex the db to keep access performant."""
        pass

    def perform_similarity_search(
        self, query: str, retrieve_num: int = 4
    ) -> list[tuple[Document, float]]:
        if self.db is not None:
            return self.db.similarity_search_with_relevance_scores(query, k=retrieve_num)
        else:
            return []

    def lookup_file_ids(self, file: str) -> list[int]:
        """Get list of embedding ids in db sourced from file.

        Args:
            file: Filepath to lookup ids for.

        Returns:
            The ids of embeddings that come from file.
        """
        return []

    def get_ids_from_concept(self, concept_query: str) -> list[int]:
        """Get ids related to concept

        Args:
            concept_query: _description_

        Returns:
            _description_
        """
        return []
