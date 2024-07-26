import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.mixture import GaussianMixture
from dotenv import load_dotenv
import traceback
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def summarize_cluster(cluster_texts, max_length=150):
    # Join all texts in the cluster
    full_text = ' '.join(cluster_texts)
    
    # Tokenize into sentences
    sentences = sent_tokenize(full_text)
    
    # Log the length of the text being summarized
    print(f"Summarizing text of length: {len(full_text)}")
    
    # If the text is short enough, return it as is
    if len(full_text) <= max_length:
        return full_text
    
    # Ensure the input to the summarizer is within the model's maximum input length
    max_input_length = 1024  # Adjust this based on the model's max length
    if len(full_text) > max_input_length:
        full_text = full_text[:max_input_length]
    
    # Otherwise, use the summarization pipeline
    summary = summarizer(full_text, max_length=max_length, min_length=30, do_sample=False)
    
    return summary[0]['summary_text']

def create_raptor_index(embeddings, chunks):
    gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
    gmm.fit(embeddings)
    cluster_labels = gmm.predict(embeddings)

    summarized_clusters = []
    for cluster in set(cluster_labels):
        cluster_texts = [chunks[i] for i in range(len(chunks)) if cluster_labels[i] == cluster]
        
        if not cluster_texts:
            continue  # Skip empty clusters
        
        summary = summarize_cluster(cluster_texts)
        summarized_clusters.append(summary)
    
    summarized_embeddings = model.encode(summarized_clusters)
    return summarized_embeddings, summarized_clusters

def hybrid_retrieval(query, faiss_results, texts, top_k=10):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    cross_encoder_scores = cross_encoder.predict([(query, text) for text in texts])
    
    # Combine FAISS scores (distances) with cross-encoder scores
    combined_results = [
        (i, 1 / (1 + dist) * 0.5 + cross_encoder_score * 0.5)
        for i, (dist, cross_encoder_score) in enumerate(zip(faiss_results[1][0], cross_encoder_scores))
    ]
    
    # Sort by combined score
    sorted_results = sorted(combined_results, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

def retrieve_from_faiss(index, query_embedding, k=10):
    return index.search(query_embedding.astype('float32').reshape(1, -1), k)

def load_and_process_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = f.readlines()
    embeddings = np.load(file_path.replace('_chunks.txt', '_embeddings.npy'))
    summarized_embeddings, summarized_clusters = create_raptor_index(embeddings, chunks)
    metadata = [{"title": file_path, "page": i} for i in range(len(summarized_clusters))]
    return summarized_embeddings, summarized_clusters, metadata

if __name__ == "__main__":
    try:
        print(f"Current working directory: {os.getcwd()}")

        all_embeddings = []
        all_texts = []
        all_metadata = []

        # Indexing
        text_files = ['data/textbook1_chunks.txt', 'data/textbook2_chunks.txt', 'data/textbook3_chunks.txt']
        for text_file in text_files:
            embeddings, texts, metadata = load_and_process_data(text_file)
            all_embeddings.append(embeddings)
            all_texts.extend(texts)
            all_metadata.extend(metadata)

        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)

        # Create FAISS index
        faiss_index = create_faiss_index(combined_embeddings)

        # Example query and retrieval
        query = "What is the main idea of chapter 5?"
        query_embedding = model.encode([query])[0]
        faiss_results = retrieve_from_faiss(faiss_index, query_embedding)
        
        hybrid_results = hybrid_retrieval(query, faiss_results, all_texts)
        
        print(f"Query: {query}")
        print("\nResults:")
        for i, (idx, score) in enumerate(hybrid_results, 1):
            print(f"\n{i}. Text: {all_texts[idx][:100]}...")  # Print first 100 characters of text
            print(f"   Combined Score: {score}")
            print(f"   Metadata: {all_metadata[idx]}")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
