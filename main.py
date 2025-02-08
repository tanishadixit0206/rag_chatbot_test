import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from src.services.faiss_services import save_to_faiss, search_faiss
from src.data.loader import load_book
from src.data.splitter import split_text
from src.services.chroma_services import load_chroma, save_to_chroma
from src.models.summarizer import generate_result
from langchain.embeddings import HuggingFaceEmbeddings

BOOK_PATH = "./docs/alice_in_wonderland.txt"
CHROMA_PATH = "chroma"
FAISS_INDEX_PATH = "faiss.index"

class StellaEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, model_name="infgrad/stella-base-en-v2", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.client = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.client.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.client.encode(text, normalize_embeddings=True).tolist()

embeddingFunction = StellaEmbeddings()
def main():
    global embeddingFunction
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
        save_to_chroma(persist_directory=CHROMA_PATH, chunks=chunks, embedding_function=embeddingFunction)
        save_to_faiss(faiss_index_path=FAISS_INDEX_PATH, chunks=chunks, embedding_function=embeddingFunction)

    db = load_chroma(persist_directory=CHROMA_PATH, embedding_function=embeddingFunction)

    faissResults = search_faiss(faiss_index_path=FAISS_INDEX_PATH, query_text=query_text, embedding_function=embeddingFunction, k=3)
    indices, scores = zip(*faissResults) if faissResults else ([], [])
    docs_metadata = db.get(limit=len(indices), offset=min(indices))

    doc_ids = docs_metadata.get("ids", [])

    results = db.get_by_ids(doc_ids)
    
    if len(results) == 0 :
        print(f"\nUnable to find matching results for '{query_text}'")
        return
    
    first_score = scores[0]
    if isinstance(first_score, (list, np.ndarray)):
        first_score = first_score.item()

    if first_score < 0.4:
        print(f"\nUnable to find matching results for '{query_text}'")
        return

    print("Fetched results ! Framing the answer....\n")
    prompt = 'On the basis of the given context and relevance score data: \n\n\n'
    for doc in results:
        prompt = f'{prompt} context: {doc.page_content}\nrelevance score: {scores}\n\n'

    page_contents = [doc.page_content for doc in results]
    prompt = f'{prompt}\nAnswer the following question:\n\n{query_text}'
    context = "".join(page_contents)
    prompt = f"Context: {context} \n\n Question: {query_text}\nPlease answer in at least one complete sentense."

    print("The prompt being passed to the Model is: ")
    print(prompt)
    print("\n\n\n")
    reply = generate_result(prompt)
    print(reply)

if __name__ == "__main__":
    main()