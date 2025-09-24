📄 RAG Pipeline – Retrieval-Augmented Generation for PDFs

🚀 Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline in Google Colab. It processes PDF documents, extracts relevant information, and provides intelligent responses by combining document chunking, vector embeddings, semantic search, and LLM-based generation.

The pipeline enables document-based Q&A, knowledge retrieval, and intelligent search with traceability back to source documents.

⸻

🏗️ System Architecture

Pipeline Flow:
	1.	PDF Processing – Extract text & metadata using pdfplumber.
	2.	Text Chunking – Split documents into chunks with LangChain’s RecursiveCharacterTextSplitter.
	3.	Embeddings – Encode chunks with SentenceTransformers (paraphrase-mpnet-base-v2).
	4.	Vector Search – Store & query embeddings using FAISS.
	5.	LLM Response Generation – Use TinyLlama-1.1B-Chat for contextual responses.

⸻

🛠️ Key Technologies
	•	PDF Processing: pdfplumber
	•	Chunking: LangChain
	•	Embeddings: SentenceTransformers
	•	Vector DB: FAISS
	•	LLM: TinyLlama-1.1B-Chat
	•	Other: scikit-learn, numpy, Pillow

⸻

⚙️ Installation & Setup

# Install dependencies
pip install pdfplumber langchain sentence-transformers faiss-cpu scikit-learn numpy transformers accelerate pillow

Authentication

To use Hugging Face models:

from huggingface_hub import login
login()  # Requires HuggingFace token


⸻

🔑 Core Functions

1. PDF Text Extraction

page_data = load_pdf_text_with_metadata("sample.pdf")

	•	Extracts text with bounding boxes & page metadata.
	•	Supports multi-page documents.

2. Text Chunking

chunks = chunk_text_with_metadata(page_data, chunk_size=800, chunk_overlap=100)

	•	Preserves structure & position info.

3. Embeddings & Indexing

embedding_model, faiss_index, chunks = embed_and_index_chunks(chunks)

	•	Embeds text using SentenceTransformers.
	•	Stores vectors in FAISS index.

4. Semantic Search + LLM

search("What is the pipeline architecture?", embedding_model, faiss_index, chunks)

	•	Retrieves top-k relevant chunks.
	•	Generates contextual response with TinyLlama.

⸻

🧑‍💻 Usage

Basic Flow

# 1. Upload PDFs
page_data = load_pdf_text_with_metadata("document.pdf")

# 2. Chunking
chunks = chunk_text_with_metadata(page_data)

# 3. Embeddings & Index
embedding_model, faiss_index, chunks = embed_and_index_chunks(chunks)

# 4. Query
search("Explain the core components", embedding_model, faiss_index, chunks)

Advanced Options
	•	Custom chunking: Adjust chunk_size and chunk_overlap.
	•	Search parameters: Tune top_k and min_score.
	•	Visual feedback: Highlight retrieved chunks on PDF images.

⸻

🌟 Features
	•	✅ Semantic PDF search with context-aware responses
	•	✅ Visual highlighting of relevant PDF sections
	•	✅ Multi-document processing
	•	✅ Persistent storage with pickle
	•	✅ Conversation analysis for transcripts

⸻

🐞 Troubleshooting
	•	Font warnings → Cosmetic, can be ignored.
	•	Memory issues → Reduce chunk_size or process in batches.
	•	Low relevance scores → Lower min_score or refine query.
	•	Model errors → Ensure HuggingFace authentication & internet access.

⸻

📊 Performance
	•	Document processing: ~2–5 sec/page
	•	Embedding generation: ~1–2 sec/chunk
	•	Search latency: <100ms (typical queries)
	•	Memory usage: ~2–4GB (moderate docs)

⸻

📌 Best Practices
	•	Use PDFs with extractable text (not scanned images).
	•	Write specific queries for better retrieval.
	•	Tune chunk size & overlap based on document type.
	•	Validate search results before production use.

⸻

🔮 Future Enhancements
	•	Larger document scaling
	•	Domain-specific fine-tuning
	•	Advanced retrieval + re-ranking
	•	Production system integration

⸻

📖 License

This project is released under the MIT License.


