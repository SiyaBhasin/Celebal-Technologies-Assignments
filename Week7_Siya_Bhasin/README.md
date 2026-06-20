# Week 7 - Document Question Answering System using RAG

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that
answers questions based on a custom document. Instead of relying only on a
language model's internal knowledge, the system retrieves relevant
information from the document and then generates answers grounded in that
information. This improves factual accuracy and allows question answering
over private or domain-specific data.

## Dataset

Custom PDF: **Artificial Intelligence course notes** (143 pages)

RAG is meant for custom/private data — any PDF (notes, resume, research
paper, book) can be substituted by changing `PDF_PATH` in the notebook.

## Objectives

- Understand the concept of Retrieval-Augmented Generation (RAG)
- Build a pipeline combining retrieval and generation
- Enable question answering over custom documents such as PDFs or text files
- Learn how modern AI systems work internally

## Key Concepts

1. **Retrieval** — finds the most relevant chunks of text from the document
   using embeddings and vector similarity search.
2. **Augmentation** — the retrieved content is added to the model's input to
   provide context for answering.
3. **Generation** — a language model generates the final answer using the
   retrieved context, ensuring responses are grounded in actual data.

## System Architecture

1. **Document Ingestion** — the PDF is loaded and converted into raw text.
2. **Text Chunking** — the text is split into smaller overlapping chunks to
   improve retrieval accuracy.
3. **Embedding Creation** — each chunk is converted into a vector
   representation capturing its semantic meaning.
4. **Vector Database** — embeddings are stored in FAISS for efficient
   similarity search.
5. **Query Processing** — the user's question is converted into an
   embedding.
6. **Context Retrieval** — the system retrieves the most relevant chunks
   from the database.
7. **Answer Generation** — a language model (Google Gemini) generates an
   answer using the retrieved context.

## Components Used

| Component | Choice |
|---|---|
| Document loader | `PyPDFLoader` (LangChain) |
| Chunking | `RecursiveCharacterTextSplitter` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Language model | Google Gemini (`gemini-2.5-flash-lite`) |
| Orchestration | LangChain (`RetrievalQA`) |
| Platform | Google Colab |

## System Configuration

| Component | Value |
|---|---|
| Chunk Size | 500 |
| Chunk Overlap | 50 |
| Embedding Model | all-MiniLM-L6-v2 |
| Embedding Dimension | 384 |
| Vector Store | FAISS |
| Top-K Retrieval | 4 |
| Language Model | Gemini gemini-2.5-flash-lite |

## Workflow

1. Load and preprocess the PDF document
2. Split text into overlapping chunks
3. Convert chunks into embeddings
4. Store embeddings in a FAISS vector database
5. Accept a user query
6. Retrieve the most relevant chunks
7. Generate a grounded answer using the retrieved context

## Example Flow

**User Question:** "What is the main idea of the document?"

**System Process:**
- Retrieves the top-4 most relevant chunks from FAISS
- Provides them as context to Gemini
- Generates a concise, grounded answer in 2-4 sentences, citing the source
  pages used

## Results

- Successfully loaded and processed a 143-page PDF document.
- Generated 545 text chunks from the extracted content.
- Created vector embeddings (384-dim) using the all-MiniLM-L6-v2 model.
- Stored embeddings in a FAISS vector database for efficient similarity
  search.
- Retrieved contextually relevant document chunks for domain-specific
  queries (AI history, knowledge-based agents, supervised learning, etc.).
- Generated grounded, multi-sentence answers using Google Gemini, with an
  average answer latency of ~1.2 seconds across the validation suite.
- Validated retrieval and generation performance using 4 sample questions,
  logging chunk usage and latency for each.

## How to Run

1. Open `RAG_Document_QA.ipynb` in Google Colab.
2. Run the cells from top to bottom (**Runtime → Run all**).
3. The notebook loads the PDF from `sample_data/` by default; if not found,
   it will prompt you to upload one directly in Colab.
4. When prompted, enter a free Google Gemini API key (get one at
   [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
   — no billing required for the free tier).
5. Ask your own questions in the **Ask Questions** section, or review the
   automated **Validation Log** and **System Metrics Summary**.

## Improvements & Experiments (Future Work)

- Use better/larger chunking strategies (e.g. semantic chunking)
- Try different embedding models (e.g. `all-mpnet-base-v2`)
- Improve retrieval using hybrid search (keyword + vector / BM25 + embeddings)
- Add a re-ranking model layer for better relevance
- Experiment with different language models and compare answer quality

## Key Learnings

- How RAG systems combine retrieval and generation to ground answers in
  real data
- The importance of chunk size and overlap for retrieval accuracy
- Working with embeddings and vector databases (FAISS)
- Connecting a retrieval pipeline to a hosted LLM (Google Gemini) via
  LangChain
- Handling unstructured PDF text and designing a scalable, modular RAG
  pipeline

## Conclusion

This project demonstrates a complete Retrieval-Augmented Generation
pipeline — covering document ingestion, chunking, embedding, vector
storage, retrieval, and grounded answer generation. RAG systems like this
are widely used in chatbots, knowledge assistants, enterprise search
systems, and AI-powered documentation tools.
