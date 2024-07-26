import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def load_or_create_faiss_index(index_path, text_files):
    if os.path.exists(index_path):
        print(f"Loading existing index from {index_path}")
        return faiss.read_index(index_path)
    else:
        print(f"Creating new index and saving to {index_path}")
        all_embeddings = []
        for file_path in text_files:
            embeddings = np.load(file_path.replace('_chunks.txt', '_embeddings.npy'))
            all_embeddings.append(embeddings)
        combined_embeddings = np.vstack(all_embeddings)
        index = create_faiss_index(combined_embeddings)
        faiss.write_index(index, index_path)
        return index

def retrieve_from_faiss(index, query_embedding, k=10):
    return index.search(query_embedding.astype('float32').reshape(1, -1), k)

def hybrid_retrieval(query, faiss_results, texts, top_k=10):
    cross_encoder_scores = cross_encoder.predict([(query, text) for text in texts])
    
    combined_results = [
        (i, 1 / (1 + dist) * 0.5 + cross_encoder_score * 0.5)
        for i, (dist, cross_encoder_score) in enumerate(zip(faiss_results[1][0], cross_encoder_scores))
    ]
    
    sorted_results = sorted(combined_results, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

def load_texts_and_metadata(file_paths):
    all_texts = []
    all_metadata = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        all_texts.extend(texts)
        metadata = [{"title": file_path, "page": i} for i in range(len(texts))]
        all_metadata.extend(metadata)
    return all_texts, all_metadata

def retrieve(query, index_path, text_files, top_k=10):
    # Load or create FAISS index
    faiss_index = load_or_create_faiss_index(index_path, text_files)
    
    # Load texts and metadata
    all_texts, all_metadata = load_texts_and_metadata(text_files)
    
    # Encode query
    query_embedding = model.encode([query])[0]

    # Retrieve from FAISS
    faiss_results = retrieve_from_faiss(faiss_index, query_embedding, top_k)
    
    # Perform hybrid retrieval
    hybrid_results = hybrid_retrieval(query, faiss_results, all_texts, top_k)

    # Process results
    results = []
    for i, (idx, score) in enumerate(hybrid_results):
        results.append({
            'text': all_texts[idx],
            'score': score,
            'metadata': all_metadata[idx]
        })

    return results

if __name__ == "__main__":
    # Paths to your index and text files
    index_path = 'faiss_index.index'
    text_files = ['data/textbook1_chunks.txt', 'data/textbook2_chunks.txt', 'data/textbook3_chunks.txt']
    
    query = "What is the main idea of chapter 5?"
    results = retrieve(query, index_path, text_files)

    print(f"Query: {query}")
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Text: {result['text'][:100]}...")  # Print first 100 characters of text
        print(f"   Combined Score: {result['score']}")
        print(f"   Metadata: {result['metadata']}")