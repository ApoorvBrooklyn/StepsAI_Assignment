from sklearn.mixture import GaussianMixture
import openai
import numpy as np
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection
from sentence_transformers import SentenceTransformer

openai.api_key = 'your_openai_api_key'

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

def summarize_cluster(cluster_texts):
    prompt = "Summarize the following text:\n" + '\n'.join(cluster_texts)
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

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

if __name__ == "__main__":
    texts = ['data/textbook1_chunks.txt', 'data/textbook2_chunks.txt', 'data/textbook3_chunks.txt']
    for text_file in texts:
        with open(text_file, 'r') as f:
            chunks = f.readlines()
        embeddings = np.load(text_file.replace('_chunks.txt', '_embeddings.npy'))
        summarized_embeddings, summarized_clusters = create_raptor_index(embeddings, chunks)
        metadata = [{"title": text_file, "page": i} for i in range(len(chunks))]
        store_in_milvus(summarized_embeddings, summarized_clusters, metadata)
