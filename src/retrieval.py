from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np
from pymilvus import Collection, connections
import os
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Connect to Milvus
connections.connect(host='localhost', port='19530')

def bm25_retrieval(query, top_k=10):
    # Placeholder for BM25 implementation. You need to replace this with actual BM25 retrieval logic.
    # For now, we'll just return some dummy results
    return [{"text": f"Sample text {i} from BM25 result."} for i in range(top_k)]

def hybrid_retrieval(query, top_k=10):
    bm25_results = bm25_retrieval(query, top_k)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = cross_encoder.predict([(query, result['text']) for result in bm25_results])
    sorted_results = sorted(zip(bm25_results, scores), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

def retrieve_from_milvus(query_embedding, top_k=10):
    collection = Collection("raptor_index")
    collection.load()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=None,
        output_fields=["text", "metadata"]
    )
    return results

def process_milvus_results(results):
    processed_results = []
    for hit in results[0]:
        processed_results.append({
            'id': hit.id,
            'distance': hit.distance,
            'text': hit.entity.get('text'),
            'metadata': hit.entity.get('metadata')
        })
    return processed_results

def retrieve(query, top_k=10):
    # Encode query
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]

    # Retrieve from Milvus
    milvus_results = retrieve_from_milvus(query_embedding, top_k)
    processed_milvus_results = process_milvus_results(milvus_results)

    # Perform hybrid retrieval
    hybrid_results = hybrid_retrieval(query, top_k)

    # Combine results (you might want to implement a more sophisticated combination strategy)
    combined_results = processed_milvus_results + [item[0] for item in hybrid_results]

    # Deduplicate and sort (assuming you have a way to compare and sort the combined results)
    # This is a simple example, you might need a more complex sorting strategy
    deduplicated_results = list({r['text']: r for r in combined_results}.values())
    sorted_results = sorted(deduplicated_results, key=lambda x: x.get('distance', float('inf')))

    return sorted_results[:top_k]

if __name__ == "__main__":
    query = "What is the main idea of chapter 5?"
    results = retrieve(query)

    print(f"Query: {query}")
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Text: {result['text'][:100]}...")  # Print first 100 characters of text
        print(f"   Distance: {result.get('distance', 'N/A')}")
        print(f"   Metadata: {result.get('metadata', 'N/A')}")