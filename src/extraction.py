import textract

def extract_text_from_pdf(pdf_path):
    text = textract.process(pdf_path)
    return text.decode('utf-8')

if __name__ == "__main__":
    textbooks = ['data/textbook1.pdf', 'data/textbook2.pdf', 'data/textbook3.pdf']
    for textbook in textbooks:
        text = extract_text_from_pdf(textbook)
        with open(textbook.replace('.pdf', '.txt'), 'w') as f:
            f.write(text)