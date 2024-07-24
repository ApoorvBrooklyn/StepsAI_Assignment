from sentence_transformers import SentenceTransformer
import numpy as np

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

if __name__ == "__main__":
    texts = ['data/textbook1_chunks.txt', 'data/textbook2_chunks.txt', 'data/textbook3_chunks.txt']
    for text_file in texts:
        with open(text_file, 'r') as f:
            chunks = f.readlines()
        embeddings = embed_chunks(chunks)
        np.save(text_file.replace('_chunks.txt', '_embeddings.npy'), embeddings)
