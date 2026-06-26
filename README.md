# Celebal Technologies – Data Science & Agentic AI Assignments

This repository contains the weekly technical assignments completed during the **Celebal Technologies Data Science Internship**. The core focus of this workspace is exploring data engineering pipelines, classical machine learning architectures, and modern Agentic AI workflows.

---

## 📂 Repository Structure

The workspace is organized into weekly modular modules, culminating in a production-grade single-agent routing framework:

* **Week1_Siya_Bhasin.ipynb** to **Week5_Siya_Bhasin.ipynb** – Core Data Science foundations, preprocessing workflows, and sequence modeling.
* **Week6_Siya_Bhasin.ipynb** – Autoencoders (AE) and Generative Adversarial Networks (GANs).
* **Week7_Siya_Bhasin.ipynb** – Retrieval-Augmented Generation (RAG) pipelines and Large Language Models (LLMs).
* **Week8_Siya_Bhasin/** – **Agentic AI Workflow Framework** *(Current Focus)*
  * `week_8_assignment.ipynb` – The single-agent orchestration and tool-routing framework pipeline.

---

## 🧠 Core Highlight: Week 8 – Single-Agent Pipeline Project

The flagship implementation in this repository is a robust, tool-using **Single-Agent Smart Assistant**. It dynamically evaluates unstructured user queries, manages intent routing via pattern-matching, safely manages state trajectories, and maps outputs into clean, predictable structural schema payloads.

### 🛠️ Agent Implementation Features

1. **Intent-Based Conditional Routing:** Evaluates raw text streams via normalized case-insensitive keyword checking to determine the optimal execution path.
2. **Deterministic Tool Orchestration:**
   * **Calculator Tool:** Safely isolates and parses strings to evaluate mathematical structures.
   * **Keyword Extractor:** Isolates semantic chunks, strips syntax fillers (like *"from"*), and returns unique, length-filtered tokens.
   * **Word Counter (🚀 Bonus Tool):** Dynamically computes total word counts, character lengths, and sentence numbers for unstructured text.
3. **Observability & Trajectory Logging (🚀 Bonus Feature):** Utilizes Python’s stream-oriented `logging` layer to provide comprehensive visibility. It records timestamped runtime traces (`[INFO]`, `[WARNING]`) tracking every operational decision step.
4. **Resilient Error Fallbacks:** Gracefully catches malformed expressions or invalid strings. Instead of dropping execution threads, it intercepts runtime exceptions and wraps them in explicit error structures.

---

## 📦 Data Schema Specifications

Every pipeline terminal node guarantees a cleanly structured JSON-serializable dictionary matching the mandatory course schema layout:

```json
{
  "type": "calculation / keywords / word_count / general / error",
  "result": "..."
}
