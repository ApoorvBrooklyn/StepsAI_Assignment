# 
from sklearn.mixture import GaussianMixture
import numpy as np
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection
from openai import OpenAI
import os
from dotenv import load_dotenv

# Construct the path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')

# Load the .env file
load_dotenv(dotenv_path)

# Get the API key from the environment variables
api_key = os.getenv('API_KEY')

print(f"API Key: {api_key}")  # Debug print, remove in production

if not api_key:
    raise ValueError("API key not found. Make sure it's set in your .env file.")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def split_text(text, max_tokens=8000):
    """Split text into chunks of approximately max_tokens tokens."""
    chunks = []
    current_chunk = []
    current_size = 0
    for sentence in text.split('.'):
        sentence = sentence.strip() + '.'
        sentence_size = num_tokens_from_string(sentence)
        if current_size + sentence_size > max_tokens:
            chunks.append('.'.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    if current_chunk:
        chunks.append('.'.join(current_chunk))
    return chunks

def summarize_cluster(cluster_texts):
    full_text = ' '.join(cluster_texts)
    chunks = split_text(full_text)
    summaries = []
    
    for chunk in chunks:
        prompt = f"Summarize the following text:\n{chunk}"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            summaries.append(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error in summarize_cluster: {e}")
            summaries.append("Error in summarization")
    
    # Combine summaries
    combined_summary = " ".join(summaries)
    
    # Create a final summary if needed
    if len(summaries) > 1:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Provide a concise summary of the following text:\n{combined_summary}"}
                ],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in final summarization: {e}")
            return combined_summary
    else:
        return combined_summary
    
    
def create_raptor_index(embeddings, chunks):
    gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
    gmm.fit(embeddings)
    cluster_labels = gmm.predict(embeddings)

    summarized_clusters = []
    for cluster in set(cluster_labels):
        cluster_texts = [chunks[i] for i in range(len(chunks)) if cluster_labels[i] == cluster]
        summary = summarize_cluster(cluster_texts)
        summarized_clusters.append(summary)
    
    summarized_embeddings = embed_chunks(summarized_clusters)
    return summarized_embeddings, summarized_clusters

def store_in_milvus(embeddings, texts, metadata):
    connections.connect(host='localhost', port='19530')

    fields = [
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.STRING),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields, "RAPTOR Index")
    collection = Collection(name="raptor_index", schema=schema)
    
    entities = [
        embeddings.tolist(),
        texts,
        metadata
    ]
    collection.insert(entities)

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)


if __name__ == "__main__":
    texts = ['data/textbook1_chunks.txt', 'data/textbook2_chunks.txt', 'data/textbook3_chunks.txt']
    for text_file in texts:
        with open(text_file, 'r', encoding='utf-8') as f:
            chunks = f.readlines()
        embeddings = np.load(text_file.replace('_chunks.txt', '_embeddings.npy'))
        summarized_embeddings, summarized_clusters = create_raptor_index(embeddings, chunks)
        metadata = [{"title": text_file, "page": i} for i in range(len(chunks))]
        store_in_milvus(summarized_embeddings, summarized_clusters, metadata)