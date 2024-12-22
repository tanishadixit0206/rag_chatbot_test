import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data.loader import load_book
from src.data.splitter import split_text
from src.services.chroma_services import save_to_chroma, load_chroma
from src.models.summarizer import generate_result

BOOK_PATH = "./docs/alice_in_wonderland.txt"
CHROMA_PATH = "chroma"
model_name = "facebook/bart-large-cnn"
sentense_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text):
        return self.model.encode(text, show_progress_bar=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q","--query_text", type = str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    if not query_text:
        query_text = input("Enter your prompt: ")

    print("Recieved your prompt ! Loading the db and finding matching results....")
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
        documents = load_book()
        chunks = split_text(documents)
        save_to_chroma(chunks)
    embedding_function = SentenceTransformerEmbeddings(sentense_transformer_model)
    db = load_chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

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

    print("Fetched results ! Framing the answer....\n")
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = "".join([doc.page_content for doc, _score in results])
    reply = generate_result(prompt)
    print(reply)

if __name__ == "__main__":
    main()