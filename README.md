# 🧠 Offline RAG Assistant

A secure, privacy-focused **Retrieval-Augmented Generation (RAG)** chatbot that runs entirely offline. Built with **LangChain**, **Streamlit**, and **Phi-2**, this application allows you to chat with your PDF documents without ever sending data to the cloud.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🚀 Key Features

*   **100% Offline**: No API keys, no cloud costs, no internet required.
*   **Privacy First**: Your sensitive documents stay on your local machine.
*   **High Performance**: Uses **FAISS** for lightning-fast vector retrieval.
*   **Quantized LLM**: Runs the efficient **Phi-2** model (via `llama-cpp-python`) optimized for CPU usage.
*   **Clean UI**: A professional, minimalist interface built with Streamlit.
*   **Streaming Responses**: Real-time token streaming for a responsive chat experience.

---

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **Orchestration**: LangChain (Community & Text Splitters)
*   **LLM Inference**: Llama.cpp (Python Bindings)
*   **Vector Store**: FAISS (CPU)
*   **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
*   **Document Loading**: PyPDFLoader

---

## 📂 Project Structure

```bash
offline_rag/
├── data/                   # Place your PDF documents here
├── models/                 # Stores the quantized LLM (phi-2.gguf)
├── .streamlit/             # Streamlit configuration (Theme settings)
├── app.py                  # Main chatbot application
├── ingest.py               # Script to process PDFs & create vector index
├── requirements.txt        # Python dependencies
├── run_app.bat             # One-click launcher for the app
├── run_ingest.bat          # One-click launcher for document processing
└── README.md               # Documentation
```

---

## ⚡ Installation Guide

### Prerequisites
*   **Python 3.10** or higher installed.
*   **Visual Studio C++ Build Tools** (Required for compiling `llama-cpp-python` on Windows).

### 1. Clone the Repository
```bash
git clone https://github.com/sreemukhik/offline_rag.git
cd offline_rag
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
**Option A: Automatic (Batch Script)**
Run `install_llm.bat` (if available) or simply:
```bash
pip install -r requirements.txt
```

**Option B: Manual Installation**
If you face issues with `llama-cpp-python`, install the pre-compiled wheel specific to your system or enable standard installation:
```bash
pip install langchain langchain-community langchain-text-splitters pypdf faiss-cpu sentence-transformers streamlit
# For CPU-only installation of llama-cpp:
pip install llama-cpp-python
```

### 4. Download the Model
Download **Phi-2 GGUF** (Quantized) and place it in the `models/` folder.
*   **Recommended Model**: `phi-2.Q4_K_M.gguf`
*   **Path**: `models/phi-2.gguf`

---

## 📖 Usage

### Step 1: Ingest Documents
Place all your PDF files into the `data/` folder. Subdirectories are supported.
Then, run the ingestion script to create the vector database:

```bash
run_ingest.bat
# OR manually:
# python ingest.py
```
*You should see a message confirming the creation of `faiss_index.bin`.*

### Step 2: Launch the Chatbot
Start the application:

```bash
run_app.bat
# OR manually:
# streamlit run app.py
```

The app will open automatically in your browser (usually at `http://localhost:8501`).

---

## 🔧 Troubleshooting

### "Model not found"
Ensure the `phi-2.gguf` file is strictly visible inside the `models/` directory.

### "Llama-cpp-python failed to build"
This is common on Windows.
1. Install **Visual Studio Build Tools** with "Desktop development with C++".
2. Or, download a pre-built wheel file from [here](https://github.com/abetlen/llama-cpp-python/releases) matching your Python version and Install it via `pip install <filename>.whl`.

### "Permissions Error" or "Path too long"
Windows has a 260-character path limit.
1. Enable **Long Paths** in Windows Registry.
2. Or, move the project to a shorter path like `C:\rag\`.

---

## 📜 License
This project is licensed under the MIT License.
