from langchain.chains import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import pickle
from dotenv import load_dotenv
import streamlit as st


load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=API_KEY,
    temperature=0.6
)

file_path = "vector_index.pkl"




st.title("News Research Tool 📈")
st.sidebar.title("News Artricle URLs")


urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

proccess_button = st.sidebar.button("Proccess URLs")


placeholder = st.empty()

if proccess_button:
    # laod the data
    loader = UnstructuredURLLoader(urls=urls)
    placeholder.text("Data Loading Started...")
    data = loader.load()

    # split data
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
    )
    placeholder.text("Text Splitter Started...")
    chunks = splitter.split_documents(data)
    
    # create embeddings
    embeddings = HuggingFaceEmbeddings()
    placeholder.text("Embedding Vector Started Building...")
    vector_index = FAISS.from_documents(chunks, embeddings)
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vector_index, f)


query = placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_index = pickle.load(f)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_index.as_retriever())
        respons = chain.invoke({"question": query}, return_only_outputs=True)

        # {answer: "", sources: []}
        st.header("Answer")
        st.subheader(respons["answer"])

        sources = respons.get("sources", "")
        if sources:
            st.subheader("Sources: ")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
