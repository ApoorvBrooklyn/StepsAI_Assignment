import nltk
from nltk.tokenize import sent_tokenize

# Ensure the punkt tokenizer is downloaded
nltk.download('punkt')

def read_text_with_fallback(file_path, encodings=['utf-8', 'latin1', 'iso-8859-1', 'utf-16']):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to decode file {file_path} with provided encodings.")

def chunk_text(text, chunk_size=100):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

if __name__ == "__main__":
    textbooks = ['data/textbook1.txt', 'data/textbook2.txt', 'data/textbook3.txt']
    for textbook in textbooks:
        try:
            text = read_text_with_fallback(textbook)
            chunks = chunk_text(text)
            chunk_file = textbook.replace('.txt', '_chunks.txt')
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(chunk + "\n")
            print(f"Chunked text from {textbook} and saved to {chunk_file}")
        except UnicodeDecodeError as e:
            print(f"Failed to read {textbook}: {e}")
