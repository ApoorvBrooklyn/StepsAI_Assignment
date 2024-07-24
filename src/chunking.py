import nltk
nltk.download('punkt')

def chunk_text(text, max_tokens=100):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(nltk.word_tokenize(sentence))
        if current_length + sentence_length > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

if __name__ == "__main__":
    texts = ['data/textbook1.txt', 'data/textbook2.txt', 'data/textbook3.txt']
    for text_file in texts:
        with open(text_file, encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        with open(text_file.replace('.txt', '_chunks.txt'), 'w', encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk + '\n')