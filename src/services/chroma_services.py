import os
import shutil
from langchain_chroma import Chroma

def save_to_chroma(persist_directory, chunks, embedding_function):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    Chroma.from_documents(
        documents=chunks, embedding=embedding_function, persist_directory=persist_directory
    )

def load_chroma(persist_directory, embedding_function):
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
