import os
import shutil
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "chroma"

class StellaEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, model_name="infgrad/stella-base-en-v2", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.client = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.client.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.client.encode(text, normalize_embeddings=True).tolist()

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding_function = StellaEmbeddings()
    Chroma.from_documents(
        documents=chunks, embedding=embedding_function, persist_directory=CHROMA_PATH
    )
def load_chroma(persist_directory, embedding_function):
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)