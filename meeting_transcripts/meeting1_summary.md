# Meeting 1 Summary - MITRA Deep Learning Platform

**Date:** May 22, 2026
**Duration:** 1 hour 16 minutes
**Participants:** Sujithkumar M A, Onkar Shamsunder Biyani, Vidhi Kant Gupta, Subhasis Mahana, Sebin Francis Kannampuzha, Meena M, Kompalli Avinash Bhargav

---

## Overview

This was the first team design meeting for **MITRA**, an agentic ML platform being built as a deep learning project. Sujithkumar led the session on a whiteboard, walking the team through the high-level architecture and the detailed design of Page 1 of the platform. The meeting covered the product vision, technology stack decisions, data input design, the Metadata Generation Agent, compute strategy, and data privacy concerns.

---

## 1. Product Vision and High-Level Architecture

Sujithkumar opened the meeting by drawing the overall system design on a shared whiteboard. The core idea of MITRA is simple: the user uploads a dataset, and the platform automatically trains a set of models and returns a **leaderboard** of the best-performing models along with their hyperparameters, plus **visualizations** like epoch-wise loss curves.

The platform follows three guiding principles:
- **Bring Your Own LLM** - Users supply their own LLM credentials (OpenAI, Gemini, or Anthropic). The quality of agents depends on the LLM the user brings.
- **Bring Your Own Data** - The platform does not host or persist user data in the cloud.
- **Bring Your Own Compute** - The platform runs entirely on the user's local machine.

---

## 2. Self-Hosted Application Design (Jupyter-Like Paradigm)

Sujithkumar emphasized that MITRA should follow the **Jupyter Notebook model** rather than a cloud-hosted SaaS model like Google Colab. The key reasoning was to avoid dealing with authentication, user session management, and cloud compute allocation. Instead:

- The user receives a binary (Docker image) called `mitra`.
- Running `mitra` starts a local server on a port (e.g., `localhost:8001`).
- The user opens the browser to that port and sees the full GUI.
- All data stays on the user's machine inside a `.mitra/` directory.
- No internet connectivity is required post-installation.

Vidhi asked about building the UI in React or Next.js. Sujithkumar decided to start with **Streamlit** - a Python-native ML UI framework that provides built-in components like file uploaders, plot visualizers, dropdowns, and checkboxes. He reasoned that migrating from Streamlit to React later is easier than building from scratch, and the team's focus should first be on the agents.

---

## 3. Technology Stack Decisions

**UI Framework:** Streamlit
- Provides ML-specific UI components out of the box.
- Sujithkumar demonstrated example Streamlit apps during the meeting, showing the `st.file_uploader()` and `st.line_chart()` APIs.

**Agent Framework:** Google ADK (Agent Development Kit)
- Sujithkumar explained there are multiple options: LangGraph, Anthropic SDK, Google ADK.
- He chose Google ADK because it supports **reasoning between agents**, handles **agent-to-agent communication protocol** natively, provides a **native visualization UI** for observing agent steps, and produces concise code that looks almost like a no-code platform.
- Vidhi mentioned the professor had suggested Anthropic SDK. Sujithkumar acknowledged it but noted Anthropic SDK does not have agent-level SDKs, while Google ADK is specifically built for agents.

**Parallel Training Framework:** Ray
- Sujithkumar introduced Ray as an open-source Python framework for core-level parallel execution.
- A laptop with 8 CPU cores can run 8 ML model trainings in parallel using Ray.
- A workstation with 96 cores can run 96 in parallel.
- Ray is invoked as a server; agents ping the Ray server to dispatch training jobs.

---

## 4. LLM Interface and .env Configuration

The user must provide LLM credentials via a `.env` file before invoking the binary. The `.env` must contain three fields:
1. **LLM API Gateway URL** - The server endpoint (e.g., OpenAI, Gemini, or a company proxy).
2. **API Key** - The authentication token.
3. **LLM Type** - One of `openai`, `gemini`, or `anthropic`.

Vidhi questioned why an agent was needed to build the client; she suggested pre-building separate client scripts for each supported LLM. Sujithkumar agreed, deciding to use pre-built client scripts (`openai_client.py`, `gemini_client.py`) selected by an if-condition based on the LLM type field. A **smoke test** is performed before anything else to verify the LLM server is reachable. If this fails, the platform does not proceed.

Vidhi also raised the question of whether additional security certificates would be needed. Sujithkumar decided to limit scope to API key + gateway URL only, covering the majority of use cases.

---

## 5. Data Input Design

The team debated what file formats to support. Supported formats were narrowed down to:
- **CSV** and **XLS** as primary formats (most tools export to CSV).
- **ZIP** for multiple files (multiple CSVs, Parquet files, or image folders compressed together).

Formats that were explicitly excluded:
- **TXT** - Header detection ambiguity (what if the first row is empty?).
- **PDF** - Too complex to parse tabular data from.
- **Voice/Audio** - Requires different preprocessing layers; deferred to future work.
- **Unstructured text** - Out of scope.

**Parquet files:** Onkar raised Parquet support. The team agreed Parquet files usually come in multiple chunks, so the ZIP convention covers this case cleanly.

**Image data:** Meena asked about uploading image datasets (e.g., multiple image files). Sujithkumar decided images must be zipped into a ZIP file, with folder names acting as class labels. The platform would recursively process all valid files inside the ZIP.

**Error handling:** For unsupported file types (e.g., a ZIP containing TXT or PDF files), the platform throws a visible error message to the user.

---

## 6. Page 1 Design - Data Upload and Metadata Generation

Page 1 of the UI handles data ingestion and initial metadata extraction. The two outputs from Page 1 are:
1. `data.csv` - The raw uploaded dataset, copied to the tool workspace (`.mitra/<session_id>/data/`).
2. `metadata.json` - A structured JSON file generated by the Metadata Generation Agent.

**UI components on Page 1:**
- A file uploader (browse + upload button).
- An optional metadata file uploader.
- A description text field where the user types a natural language prompt about their dataset and problem statement.

Sujithkumar clarified that the `.env` file is handled separately at the command line and should not appear in the GUI, to avoid exposing API keys visually.

**Train/Test Split:** Vidhi asked if users should upload separate training and test sets. Sujithkumar said no - the platform always takes one dataset and handles the train/test split internally (e.g., 80/20 split). Users often do not know this distinction.

---

## 7. Metadata Generation Agent (Agent 1)

The first agent in the pipeline is the **Metadata Generation Agent**. Its role is to analyze the uploaded data and produce a standardized `metadata.json`.

**Inputs:**
- `data.csv` (the raw dataset)
- Optional metadata file uploaded by the user
- User's free-text description from the text field

**Outputs:**
- `metadata.json` with a fixed schema including:
  - List of input columns
  - Data type of each input column (categorical or numeric)
  - Output column name
  - Data type of the output column
  - Problem type: classification or regression
  - Statistical summary: count, mean, standard deviation, quartiles, most repeating values

Meena asked how the agent would be given instructions to select models. Sujithkumar explained that each agent has a `.md` file that serves as its system prompt. The Metadata Generation Agent's `.md` will instruct it to look at column names, data statistics, and the description to produce the structured JSON.

**Handling noisy user descriptions:** Vidhi raised the concern that users could type anything in the description text field. Sujithkumar's response: the agent must be smart enough to extract intent and entities from whatever the user types. If the user types random characters, the agent should ignore those parts and rely on the data alone. The agent should extract the output column if explicitly mentioned, infer it otherwise, and identify whether the problem is classification or regression.

Kompalli Avinash contributed that problem type detection (classification vs regression) can be done programmatically by analyzing the threshold of unique value repetition in the output column - a column with many repeating discrete values is classification, and a column with continuous unique values is regression.

**What the agent can and cannot see:** The agent only sees the mini stats (mean, standard deviation, quartiles, value counts, histogram) and column names - never the full raw data. This is a hard constraint for both performance and privacy reasons.

---

## 8. Data Privacy Discussion

Meena raised the question of data privacy when the data is sent to an LLM. Sujithkumar explained:
- The application is fully self-hosted, so the user's data never leaves their machine during local processing.
- The agent only sees statistical summaries (mini data), never the raw dataset rows.
- For users with strict privacy requirements, they can point to a **company-hosted local LLM** (no external internet calls).
- The LLM call happens only for metadata generation, model selection, and feature reasoning - never for the training data itself.

**PII Discussion:** Onkar raised whether the platform should detect and filter PII columns (Aadhaar number, PAN card, mobile numbers). Sujithkumar and Kompalli Avinash discussed this and concluded:
- PII columns like mobile numbers or Aadhaar IDs are unique identifiers with no statistical correlation to any meaningful output column.
- The feature engineering step will naturally drop these columns because they show zero correlation with the output.
- The platform does not need a special PII filter layer because the math handles it.

---

## 9. Compute Handling

Onkar asked about compute requirements for users with lower-end machines (e.g., 4 GB RAM laptops). Sujithkumar said the platform would:
- Detect if a GPU is available (`if torch.cuda.is_available()`) and use it; otherwise fall back to CPU.
- Use **Ray** to distribute training across all available CPU cores in parallel.
- An 8-core laptop can train 8 models in parallel; this is the USP of the platform.

---

## 10. Page 2 and Page 3 Preview

Sujithkumar gave a preview of the remaining pages before closing the meeting:
- **Page 2** will handle data preprocessing (categorical encoding, normalization, feature engineering, correlation analysis, dimensionality reduction) and model selection + training using the agent pipeline. Agents and deterministic Python scripts work together.
- **Page 3** will display the **leaderboard** of top-performing models with accuracy and other metrics, along with visualizations (loss curves, F1, etc.).

The team was asked to think about Page 2 and Page 3 designs for the next meeting. Sujithkumar recommended the team read about ML pipeline steps using ChatGPT or Gemini to get comfortable with the concepts.

---

## Key Decisions Made

| Decision | Outcome |
|---|---|
| UI Framework | Streamlit (Python-based) |
| Agent Framework | Google ADK |
| Parallel Training | Ray |
| LLM Support | OpenAI, Gemini (Anthropic optional) |
| LLM paradigm | Bring Your Own LLM |
| Data formats | CSV, XLS, ZIP |
| Hosting model | Self-hosted (Jupyter-like binary) |
| Train/test split | Platform-managed (no user input) |
| PII handling | Feature correlation naturally eliminates PII columns |
| Image data | ZIP with folder-name = class-name convention |
