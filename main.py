import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data.loader import load_book
from src.data.splitter import split_text
from src.services.chroma_services import save_to_chroma, load_chroma
from src.models.summarizer import generate_result
from langchain.embeddings import HuggingFaceEmbeddings

BOOK_PATH = "./docs/alice_in_wonderland.txt"
CHROMA_PATH = "chroma"

class StellaEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, model_name="infgrad/stella-base-en-v2", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.client = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.client.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.client.encode(text, normalize_embeddings=True).tolist()

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

    embedding_function = StellaEmbeddings()
    db = load_chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0 :
        print(f"\nUnable to find matching results for '{query_text}'")
        return
    
    first_score = results[0][1]
    if isinstance(first_score, (list, np.ndarray)):
        first_score = first_score.item()

    if first_score < 0.4:
        print(f"\nUnable to find matching results for '{query_text}'")
        return

    print("Fetched results ! Framing the answer....\n")

    prompt = 'On the basis of the given context and relevance score data: \n\n\n'

    for doc, _score in results:
        prompt = f'{prompt} context: {doc.page_content}\nrelevance score: {_score}\n\n'
    prompt = f'{prompt}\nAnswer the following question:\n\n{query_text}'
    context = "".join([doc.page_content for doc, _score in results])
    prompt = f"Context: {context} \n\n Question: {query_text}\nPlease answer in at least one complete sentense."

    print("The prompt being passed to the Model is: ")
    print(prompt)
    print("\n\n\n")
    reply = generate_result(prompt)
    print(reply)

if __name__ == "__main__":
    main()