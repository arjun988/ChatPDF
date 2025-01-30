import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up Streamlit app
st.title("Chat with PDF using LangChain and Gemini API 🚀")

# Ask user to input their Gemini API key
api_key = st.text_input("Enter your Gemini API Key:", type="password")

if not api_key:
    st.warning("Please enter your Gemini API key to proceed.")
    st.stop()

# Upload PDF file
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if pdf_file is not None:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getvalue())

    # Load the PDF using LangChain's PyPDFLoader
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Initialize Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    # Create a FAISS vector store from the text chunks
    vector_store = FAISS.from_documents(texts, embeddings)

    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        api_key=api_key,
        model="gemini-pro",
        temperature=0.7,
        convert_system_message_to_human=True
    )

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    )

    # User query
    query = st.text_input("Enter your question:")

    if query:
        # Get the answer from the QA chain
        result = qa_chain.run(query)
        st.write("### Answer:")
        st.write(result)

        # Display relevant chunks for context
        st.write("### Relevant Context:")
        relevant_docs = vector_store.similarity_search(query, k=3)
        for i, doc in enumerate(relevant_docs):
            st.write(f"**Chunk {i + 1}:**")
            st.write(doc.page_content)
            st.write("---")