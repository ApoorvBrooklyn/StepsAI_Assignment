import streamlit as st
from question_answering import answer_question

st.title("Textbook Question Answering System")

query = st.text_input("Enter your question:")
if query:
    answer = answer_question(query)
    st.write("Answer:", answer)
