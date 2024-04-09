import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer
from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)

class Encoder():
    def __init__(self, model_name, model_kwargs) -> None:
        self.embedding_function = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs = model_kwargs,
            cache_folder=CACHE_DIR
        )


class FaissDB:
    def __init__(self, docs, embedding_function) -> None:
        self.db = FAISS.from_documents(
            docs, embedding_function, distance_strategy=DistanceStrategy.COSINE
        )
    
    def similarity_search(self, question: str, k: int):
        retrieved = self.db.similarity_search(question, k=k)
        return ''.join(doc.page_content + '\n' for doc in retrieved)


def loader_splitter(file_paths: list, chunk_size: int=256):
    loaders = [PyPDFLoader(file_path) for file_path in file_paths]

    pages = []
    for loader in loaders:
        pages.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            'sentence-transformers/all-MiniLM-L12-v2'
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 10,
        strip_whitespace=True,
    )

    return text_splitter.split_documents(pages)
