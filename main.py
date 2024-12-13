import os
import shutil
import argparse
import numpy as np
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

BOOK_PATH = "./docs/alice_in_wonderland.txt"
CHROMA_PATH = "chroma"


sentense_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        # Batch embedding of document texts
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text):
        # Embedding a single query
        return self.model.encode(text, show_progress_bar=False)

def load_book():       
    loader = TextLoader(BOOK_PATH)
    documents = loader.load()

    # for doc in documents:
    #     print(doc.page_content)
    #     print(doc.metadata)


    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=55,
        separators=["\n\n","\n"," ",""]
    )
    text_chunks = text_splitter.split_documents(documents)
    print(f'split the book into {len(text_chunks)} chunks')
    # for i, chunk in enumerate(text_chunks[:10]):
    #     print(f'{i} - {chunk}' )

    return text_chunks

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, SentenceTransformerEmbeddings(sentense_transformer_model), persist_directory=CHROMA_PATH
    )

    db.persist()

documents = load_book()
chunks = split_text(documents)
save_to_chroma(chunks)

def vector_conversion():
    embedding_function = SentenceTransformerEmbeddings(sentense_transformer_model)
    vector = embedding_function.embed_query("apple")
    print(vector)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type = str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = SentenceTransformerEmbeddings(sentense_transformer_model)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0 :
        print(f"\nUnable to find matching results for '{query_text}'")
        return
    
    first_score = results[0][1]
    if isinstance(first_score, (list, np.ndarray)):
        first_score = first_score.item()

    if first_score < 0.2:
        print(f"\nUnable to find matching results for '{query_text}'")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)

if __name__ == "__main__":
    main()


