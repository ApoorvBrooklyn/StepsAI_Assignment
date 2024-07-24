from sentence_transformers import CrossEncoder
import numpy as np
from milvus import Milvus

def bm25_retrieval(query, top_k=10):
    # Placeholder for BM25 implementation
    pass

def hybrid_retrieval(query, top_k=10):
    bm25_results = bm25_retrieval(query, top_k)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = cross_encoder.predict([(query, result['text']) for result in bm25_results])
    sorted_results = sorted(zip(bm25_results, scores), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

def retrieve_from_milvus(query_embedding, top_k=10):
    milvus = Milvus(host='localhost', port='19530')
    status, results = milvus.search(collection_name='raptor_index', query_records=[query_embedding.tolist()], top_k=top_k, params={})
    return results

if __name__ == "__main__":
    query = "What is the main idea of chapter 5?"
    query_embedding = embed_chunks([query])[0]
    results = retrieve_from_milvus(query_embedding)
    for result in results[0]:
        print(result)
