from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np
from pymilvus import Collection

def bm25_retrieval(query, top_k=10):
    # Placeholder for BM25 implementation. You need to replace this with actual BM25 retrieval logic.
    return [{"text": "Sample text from BM25 result."}]

def hybrid_retrieval(query, top_k=10):
    bm25_results = bm25_retrieval(query, top_k)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = cross_encoder.predict([(query, result['text']) for result in bm25_results])
    sorted_results = sorted(zip(bm25_results, scores), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

def retrieve_from_milvus(query_embedding, top_k=10):
    collection = Collection("raptor_index")
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=None
    )
    return results

if __name__ == "__main__":
    query = "What is the main idea of chapter 5?"
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    results = retrieve_from_milvus(query_embedding)
    for result in results[0]:
        print(result)
