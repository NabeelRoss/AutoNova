# AutoNova 

**AutoNova** is a smart Tesla assistant powered by Google Gemini and RAG (Retrieval-Augmented Generation). It helps users quickly find answers regarding Tesla features, maintenance schedules, and recall information by retrieving data from a curated knowledge base.

## 🚀 Features
* **AI-Powered Q&A:** Uses **Gemini 1.5 Pro** to answer user queries naturally.
* **RAG Architecture:** Retrieves accurate context from local text data using **LangChain** and **ChromaDB**.
* **Interactive UI:** Built with **Streamlit**, featuring a custom chat interface and dynamic background.
* **Tesla Knowledge Base:** Covers topics like Autopilot, FSD, battery maintenance, and vehicle recalls.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **LLM:** Google Gemini 1.5 Pro
* **Embeddings:** Google Generative AI Embeddings
* **Vector Store:** ChromaDB
* **Framework:** LangChain

## 📂 Project Structure
```text
AutoNova/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # API Keys (Not uploaded to GitHub)
├── Bot data/              # Knowledge base text files
│   ├── features.txt
│   ├── maintenance.txt
│   └── recall_info.txt
└── Background/            # App background images
