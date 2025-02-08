import os
import faiss
import numpy as np

def save_to_faiss(faiss_index_path, chunks, embedding_function):
    embeddings = embedding_function.embed_documents([chunk.page_content for chunk in chunks])
    embeddings_np = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    faiss.write_index(index, faiss_index_path)

def load_faiss_index(faiss_index_path):
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError("FAISS index not found. Run save_to_faiss first.")
    return faiss.read_index(faiss_index_path)

def search_faiss(faiss_index_path, query_text, embedding_function, k=3):
    index = load_faiss_index(faiss_index_path=faiss_index_path)
    
    query_vector = np.array([embedding_function.embed_query(query_text)], dtype=np.float32)

    distances, indices = index.search(query_vector, k)

    return list(zip(indices[0], distances[0]))
