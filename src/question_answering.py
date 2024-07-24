def answer_question(query):
    retrieved_data = hybrid_retrieval(query)
    context = '\n'.join([data[0]['text'] for data in retrieved_data])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    query = "What is the main idea of chapter 5?"
    answer = answer_question(query)
    print(answer)
