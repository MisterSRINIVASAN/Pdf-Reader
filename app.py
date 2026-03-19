from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub


def process_pdf(pdf):
    """Extract text and create FAISS knowledge base"""
    pdf_reader = PdfReader(pdf)
    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if not text.strip():
        raise ValueError("No readable text found in PDF")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings()

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base


def main():
    load_dotenv()

    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("📄 Chat with your PDF")

    # ---------------- SESSION STATE ---------------- #
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None

    # ---------------- FILE UPLOAD ---------------- #
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:

        # Process only if new PDF
        if st.session_state.pdf_name != pdf.name:
            try:
                with st.spinner("🔄 Processing PDF..."):
                    kb = process_pdf(pdf)

                    st.session_state.knowledge_base = kb
                    st.session_state.pdf_name = pdf.name
                    st.session_state.messages = []

                st.success(f"✅ {pdf.name} processed successfully!")

            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                return

        # ---------------- CHAT HISTORY ---------------- #
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ---------------- USER INPUT ---------------- #
        user_input = st.chat_input("Ask something about your PDF...")

        if user_input:
            # Show user message
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response
            if st.session_state.knowledge_base:
                try:
                    with st.spinner("🤖 Thinking..."):

                        docs = st.session_state.knowledge_base.similarity_search(user_input)

                        llm = HuggingFaceHub(
                            repo_id="google/flan-t5-large",
                            model_kwargs={
                                "temperature": 0.5,
                                "max_length": 512
                            }
                        )

                        chain = load_qa_chain(llm, chain_type="stuff")

                        response = chain.run(
                            input_documents=docs,
                            question=user_input
                        )

                    # Show assistant response
                    with st.chat_message("assistant"):
                        st.markdown(response)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


if __name__ == "__main__":
    main()