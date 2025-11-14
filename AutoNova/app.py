import streamlit as st
import base64
import os
from dotenv import load_dotenv

# ✅ This MUST be the first Streamlit command
st.set_page_config(page_title="TeslaBot 🚗", layout="centered")

# ✅ Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# ✅ Embed local background image using Base64
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .main {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 15px;
        }}
        .chat-bubble-user {{
            background-color: #36454F;
            padding: 10px;
            margin: 10px 0;
            border-radius: 10px;
        }}
        .chat-bubble-bot {{
            background-color: #343434;
            padding: 10px;
            margin: 10px 0;
            border-radius: 10px;
        }}
        .title-style {{
            font-size: 2.5em;
            font-weight: bold;
            color: #E5E4E2;
            margin-bottom: 0.5em;
        }}
        .subtitle-style {{
            font-size: 1.3em;
            color: #C0C0C0;
            margin-bottom: 1.5em;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Set background image (make sure this path is correct!)
set_background("Background/back3.jpg")

# ✅ Gemini 1.5 Pro setup
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY
)

# ✅ Load vector database
@st.cache_resource
def load_vector_db():
    files = [
        "Bot data/maintenance.txt",
        "Bot data/recall_info.txt",
        "Bot data/features.txt"
    ]
    docs = []
    for file in files:
        loader = TextLoader(file)
        docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    db = Chroma.from_documents(texts, embedding=embedding, persist_directory="tAutoNova_db")
    return db.as_retriever(search_kwargs={"k": 3})

# ✅ RAG chain setup
retriever = load_vector_db()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ UI Layout
st.markdown('<div class="title-style"> AutoNova - Your Smart Tesla Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-style">Ask about Tesla maintenance, recalls, or features!</div>', unsafe_allow_html=True)

# ✅ Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ User input
user_input = st.text_input("Enter your question", placeholder="e.g., What is a battery service message?")

if user_input:
    response = rag_chain.run(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AutoNova", response))

# ✅ Display chat history
with st.container():
    for sender, msg in reversed(st.session_state.chat_history):
        if sender == "You":
            st.markdown(f'<div class="chat-bubble-user"><strong>You:</strong> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-bot"><strong>AutoNova:</strong> {msg}</div>', unsafe_allow_html=True)
