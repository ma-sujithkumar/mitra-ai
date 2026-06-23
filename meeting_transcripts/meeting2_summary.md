# Meeting 2 Summary - MITRA Page 2 Backend Design

**Date:** May 24, 2026
**Duration:** 59 minutes
**Participants:** Sujithkumar M A, Kompalli Avinash Bhargav

---

## Overview

This was a focused two-person design session between Sujithkumar and Kompalli Avinash Bhargav to design the **Page 2 backend pipeline** of MITRA. The meeting picked up from where Meeting 1 left off. The team finalized the metadata schema design, walked through every step of the preprocessing pipeline, designed the feature selection agent, the model selection agent, the training sub-agent architecture, the judge/evaluator agent, and hyperparameter tuning strategy. The session ended with Avinash agreeing to take the transcripts and generate a comprehensive plan using Claude Opus with a Mermaid flow diagram.

---

## 1. Recap and Page 1 Finalization

Sujithkumar confirmed that the Page 1 design from Meeting 1 was agreed upon. The two outputs of Page 1 are:
1. `data.csv` - the raw uploaded dataset.
2. `metadata.json` - structured JSON produced by the Metadata Generation Agent (LLM call).

The key point raised immediately was the need to **fix the metadata schema** - the LLM must return only one format. A strict JSON schema must be defined upfront so that all downstream agents receive a consistent, predictable structure. Multiple formats cannot be returned.

---

## 2. Inputs to Page 2 Backend

Sujithkumar and Avinash agreed on three inputs flowing into the Page 2 backend:
1. `data.csv` - the raw dataset from the user.
2. `metadata.json` - the structured metadata from Page 1's LLM call.
3. `minidata.csv` - a statistical summary of the data (equivalent to `pandas.describe()`) including mean, standard deviation, quartiles, value counts, and type information for each column.

Avinash had prepared a diagram ahead of the meeting and shared it. He noted that the feature selection and engineering steps, along with training, should happen asynchronously in the background, with the UI receiving periodic status tokens to update a progress bar.

---

## 3. UI Progress Feedback

Avinash proposed that as each backend step completes, the system should send an asynchronous token back to the UI to update the user. Sujithkumar agreed - a progress bar or step-by-step status log (e.g., "Encoding done", "Feature selection done", "Model training started") should be shown in Page 2's UI.

Importantly, Sujithkumar pointed out that **Google ADK provides this natively** - the orchestrator's plan and step progress are already surfaced in ADK's built-in UI. The team should leverage this directly rather than building a custom status mechanism.

---

## 4. Step 1 - Categorical Encoding

The first preprocessing step is converting all columns to numeric format for ML models.

**Approach agreed upon:**
- Attempt to convert every column to `float`. If a column cannot be converted, it is a string column requiring categorical encoding.
- Use a **try-except loop** iterating over all columns to identify string columns.
- Apply **label encoding** (stable encoder, not one-hot encoding) to string columns.
- One-hot encoding was explicitly rejected to avoid dimensionality explosion.

**Outputs:**
- `data_encoded.csv` saved to `.mitra/<session_id>/` workspace.
- `encoder_map.json` - a mapping of original string values to encoded integer values, stored in the `.mitra/<session_id>/` folder (Avinash's suggestion).

**Data loading strategy:** Sujithkumar stressed that throughout all Python programs in MITRA, data must never be loaded fully into RAM via `pd.read_csv()`. Instead, all scripts must implement a **chunk-based reading strategy**: split data into chunks, apply the operation to each chunk, and write back to disk. This keeps memory usage bounded regardless of dataset size.

The UI should show a progress indicator as encoding proceeds.

---

## 5. Step 2 - Feature Selection Agent

The **Feature Selection Agent** is the most complex preprocessing component. Its job is to decide which input features (columns) to retain and which to drop before model training.

**Three mechanisms for dropping features, in priority order:**

**A. User-specified drops (explicit, mandatory):**
- If `metadata.json` contains an explicit list of columns to drop (provided by the user), those must always be dropped unconditionally - no math required.

**B. Semantic column name analysis (LLM-based):**
- Avinash raised an important point: if the metadata contains column names like `mobile_number`, `aadhaar_id`, or `customer_name`, the LLM can infer from the name alone (with domain knowledge) that these are PII-like identifiers and should be dropped.
- Sujithkumar agreed and noted this goes into the `.md` system prompt of the Feature Selection Agent: "Look at column names and metadata, and if any column appears to be a unique identifier or PII field, drop it."
- No mathematical correlation is needed for these drops; the LLM's domain intelligence handles it.

**C. Statistical correlation analysis (math-based):**
- For remaining features, compute correlation between each input feature and the output column.
- **Chi-square test** was selected as the correlation method (Avinash suggested it, Sujithkumar agreed after deliberation).
- If a feature has a low chi-square value with the output, the two are statistically independent - the feature should be dropped.
- Spearman and Pearson correlations were also briefly discussed as alternatives.

**Critical guardrail:** The Feature Selection Agent must **never read the full raw data at any point**. Sujithkumar explained that since the agent has read/write/execute-Python tools, it could write a script to `pd.read_csv()` the full file if not explicitly prevented. The agent's `.md` system prompt must contain a hard guardrail: "Do not read the full dataset at any time. Use only the minidata summary."

**Higher-order feature engineering:** Sujithkumar briefly raised the idea of having the agent experiment with combining column names to create higher-order features and check if they correlate well with the output. Avinash flagged this as a "gray area" and the team agreed to defer it.

**Output of Feature Selection Agent:**
- A list of selected features (columns to retain).
- These flow into the normalization step and then model selection.

---

## 6. Step 3 - Normalization

After categorical encoding and feature selection, normalization is applied to all retained numeric columns.

**Options discussed:**
- **Standard Scaling (Z-score / SND):** mean = 0, standard deviation = 1. This is the primary choice.
- **Log Scaling:** Avinash suggested as an option for skewed distributions.
- **Min-Max Scaling:** Mentioned as an alternative.

Sujithkumar noted that normalization helps with outlier rows but does not help decide which columns to drop. The two steps (column dropping and normalization) are independent. Normalization improves gradient descent convergence and can improve accuracy for many model types.

---

## 7. PCA / Dimensionality Reduction (Deferred)

Sujithkumar and Avinash had an extended discussion on when to apply PCA:

- **Avinash's initial position:** Apply PCA after feature selection to handle inter-feature correlation.
- **Sujithkumar's correction:** PCA is only needed when the model is **underfitting due to too many dimensions** (the algorithm takes too long to train). It is counterproductive to apply PCA when the model is already overfitting.
- **Avinash clarified:** When overfitting, you need feature extraction or data augmentation, not PCA. PCA is for the underfitting case.

Decision: PCA is deferred to a later iteration. It will be triggered conditionally by the judge/evaluator agent if underfitting is detected after the first training pass.

---

## 8. Model Selection Agent

After feature selection and preprocessing, a **Model Selection LLM call** (not a full agent loop) determines which models to experiment with.

**Inputs:**
- Reasoning output from the Feature Selection Agent.
- `metadata.json` (contains problem type: classification vs regression, output column, feature list).
- `minidata.csv` (statistical summary).

**Why an LLM call and not a hardcoded list:**
Avinash proposed hardcoding the list of possible models in the prompt. Sujithkumar pushed back: the whole point of using an LLM is that the user does not know which model to choose. The LLM's foundational intelligence should pick the appropriate candidates. If you hardcode the list, you are just doing classical modelling without intelligence.

**Output:**
- `model_config.json` - A structured list of candidate model names and configurations to experiment with.

**Orchestration after model selection:**
The orchestrator routes the pipeline into one of three sub-agent paths based on the problem type from `metadata.json`:
- **Classification Agent** - for discrete output prediction.
- **Regression Agent** - for continuous value prediction.
- **Unsupervised Learning (USL) Agent** - for clustering and association problems.

For the current scope, the team focused primarily on the classification path.

---

## 9. Model Training - Pre-built Template Library

A critical architectural decision was made around how training code is generated:

**Sujithkumar's proposal:** Build a **pre-built template library** of training scripts for all major model architectures (CNN, XGBoost, random forest, SVM, etc.). When the classification/regression agent needs to run a model, it first looks in the template library. If the template exists, it parameterizes it (number of layers, neurons, etc.) and runs it. Only if a template is absent should the agent write new code from scratch.

**Reason:** Without a template library, the agent will rewrite the same boilerplate code for every dataset and every user session, wasting tokens and introducing variability. With a library, the agent has a directed, bounded task.

Avinash suggested the templates could also serve as few-shot examples in the agent's prompt. Sujithkumar agreed: "Look at the library first. If the template exists, use it. If not, write the code. This is how it has to be prompted."

**Training execution:** Model training is dispatched to the **Ray server** for parallel execution. If the model list has 10 candidates, Ray will run training for all 10 in parallel across available CPU cores.

---

## 10. Judge Agent and Evaluation

After training, a **Judge Agent** evaluates the trained models.

**Responsibilities:**
- Calculate performance metrics (accuracy, F1 score, etc.).
- Detect **overfitting or underfitting** using variance/bias analysis.
- Use **SHAP (SHapley Additive exPlanations)** for feature importance evaluation on the trained model.

Avinash's proposed flow: the judge agent compares training vs validation metrics, detects if overfitting is occurring, and either escalates to hyperparameter tuning or, in the underfitting case, triggers PCA/dimensionality reduction before re-training.

Sujithkumar confirmed that the judge agent serves as a key decision point in the pipeline loop.

---

## 11. Hyperparameter Tuning

**Tool selected:** Optuna
- Sujithkumar explicitly ruled out LLM involvement in the hyperparameter tuning loop: "LLM in the hyperparameter loop would be very costly."
- **Grid search** was rejected as too exhaustive.
- **Random search** was accepted as feasible and fast.
- **Stratified sampling from classes** was mentioned as the sampling strategy.

Optuna handles the hyperparameter optimization loop deterministically, without LLM calls, keeping compute cost bounded.

---

## 12. Image Data Handling on Page 1 (Revisited)

During the Page 2 discussion, Avinash raised the complexity of supporting image datasets:

- **Convention agreed:** For image classification, the ZIP file must contain subfolders where the folder name is the class label (matching the ImageFolder convention in PyTorch).
- If the folder structure is not correct, the platform throws a visible error with format instructions.
- **Mini data for images is difficult:** Random sampling of images and generating statistics is non-trivial. This was noted as a hard problem.
- **New UI element for Page 1:** A dropdown was added to Page 1 allowing users to select their data type: "Image", "CSV", or "Excel". This helps the backend route the preprocessing correctly.

Avinash proposed that after the data type dropdown is selected, users should be given the choice between providing a metadata file OR a description text. Sujithkumar disagreed:
- Both should always be visible simultaneously.
- Metadata file is optional, but the description text field is **mandatory** (minimum ~20 words enforced).
- The LLM that processes Page 1 must handle both together; forcing the user to pick one creates unnecessary confusion.
- Sujithkumar's argument: "A user might have a metadata file that doesn't say which column is the output. They still need the description to clarify their problem."

---

## 13. Orchestrator Architecture

Avinash described the orchestrator as a **bidirectional conversation manager**: it does not just route one-way, but receives outputs from each agent, makes decisions, and routes to the next step. It is not a simple pipeline but a dynamic decision tree.

Key orchestrator responsibilities discussed:
- Route to classification, regression, or USL sub-agent based on problem type.
- Handle intermediate failures (e.g., Ray server not connecting) by debugging and retrying.
- Accumulate results from all training runs and pass to the judge agent.
- Not fix Mitra's own codebase under any circumstance - only fix LLM-generated scripts.

**Tool access for agents:** All agents get access to: `read_file`, `write_file`, `bash`, Python executor. This is consistent with how Claude Code operates.

---

## 14. Docker Image and Packaging

Sujithkumar confirmed the deployment model:
- All Python dependencies (PyTorch, scikit-learn, Optuna, Ray, SHAP, Streamlit, Google ADK, etc.) will be bundled inside a **Docker image**.
- The user unwraps the Docker image, sources the bundled virtual environment, and invokes the `mitra` binary.
- No internet is required after the Docker image is downloaded.
- The venv inside Docker ensures all package versions are locked and consistent across machines.

---

## 15. Action Items

At the end of the meeting, Sujithkumar asked Avinash to:
- Take both meeting transcripts.
- Feed them into **Claude Opus** (or similar strong LLM).
- Generate a **comprehensive, full-flow plan** with a **Mermaid diagram** covering the entire pipeline.
- The plan should identify all agents, their inputs/outputs, the data flow between them, and the orchestration logic.

---

## Key Decisions Made

| Component | Decision |
|---|---|
| Metadata schema | Fixed JSON schema - LLM must return only one format |
| Categorical encoding | Label encoding (not one-hot); output: data_encoded.csv + encoder_map.json |
| Data loading | Chunk-based reading always; never full pd.read_csv() |
| Feature selection | Chi-square test + LLM semantic column name analysis + user-specified drops |
| Agent guardrail | Feature selection agent must never read full raw data |
| Normalization | Standard scaling (Z-score) primary; log scale optional |
| PCA | Deferred; triggered conditionally when underfitting detected |
| Model selection | LLM call (not agentic loop); output: model_config.json |
| Training execution | Ray parallel server |
| Model code | Pre-built template library first; write from scratch only if template absent |
| Hyperparameter tuning | Optuna; no LLM in loop; random/stratified search |
| Judge agent | SHAP feature importance + overfitting/underfitting detection |
| Image data format | ZIP with folder-name = class-name convention |
| Description field | Mandatory (~20 words min); metadata file optional |
| Docker | All deps bundled; no internet needed post-install |
| Next step | Avinash to generate full Mermaid flow plan using Claude Opus |
