import os
from pathlib import Path
import shutil

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.text import TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS

INF_SPECIFIC_DST = str(Path("cache/inference_context_upload_dir/").resolve())

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db: VectorStore = None


def copy_files_to_dst(files: list[str]) -> list[str]:
    if not os.path.exists(INF_SPECIFIC_DST):
        print(f"Making {INF_SPECIFIC_DST}")
        os.makedirs(INF_SPECIFIC_DST)

    updated_files = []
    for file in files:
        if os.path.isfile(file):
            print(f"Copy {file} to destination {INF_SPECIFIC_DST}")
            updated_file = shutil.copy(file, INF_SPECIFIC_DST)
            updated_files.append(updated_file)
        else:
            print(f"Warning: {file} does not exist or is not a file.")

    return updated_files


def add_files_to_vector_store(files: list[str]):
    global db

    files = copy_files_to_dst(files)
    print(f"Adding files to vector store: {files}")

    for file in files:
        print(f"Loading file: {file}")
        loader: TextLoader = TextLoader(file_path=file)
        documents: list[Document] = loader.load()

        text_splitter: CharacterTextSplitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0
        )

        docs = text_splitter.split_documents(documents)
        if db is not None:
            db.add_documents(docs)
        else:
            db = FAISS.from_documents(docs, embeddings)


def perform_similarity_search(query, k: int) -> list[tuple[Document, float]]:
    global db
    if db is not None:
        results: list[tuple[Document, float]] = (
            db.similarity_search_with_relevance_scores(query, k=4)
        )
    else:
        results = []

    return results
