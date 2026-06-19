# Plan - Creating Submodule Architecture Cards (CARD.md)

This plan outlines the approach to documenting each submodule in the MITRA codebase to support future automated codebases cleanup, restructuring, and stitching without needing to read all individual source lines, thereby saving tokens.

## Proposed Token-Saving Documentation Format
Instead of generic unstructured Markdown, we will use a **YAML Frontmatter + Markdown** approach inside each `CARD.md` file. This is highly recommended because:
1. It is extremely parser-friendly: downstream harnesses can parse the frontmatter block directly using simple regex or standard YAML parsers to extract inputs, outputs, entry points, and dependencies without reading the full text description.
2. It is token-efficient: the frontmatter is compact, and the harness can skip reading the rest of the file if only structural/interface mapping is required.

### Standardized Card Format
```markdown
---
name: <Submodule Name>
path: <Relative Path>
purpose: <Brief 1-sentence description>
interfaces:
  inputs:
    - name: <artifact name or file>
      format: <JSON, CSV, ZIP, etc.>
      upstream: <Upstream submodule/agent name>
      description: <What this input represents>
  outputs:
    - name: <artifact name or file>
      format: <JSON, CSV, etc.>
      downstream: <Downstream submodule/agent name>
      description: <What this output represents>
entry_points:
  - name: <Class, script or function>
    type: <CLI / Python API / REST endpoint>
    description: <Description of how to run or import>
dependencies:
  - <internal/external dependency>
---

# Technical Architecture: <Submodule Name>

## Overview
<Detailed architectural description>

## Core Component Walkthrough
<Functions, classes, and logic breakdown>

## Interfacing Guide
- **Upstream Integration:** How it integrates with preceding steps.
- **Downstream Integration:** How it passes data to succeeding steps.

## Suggested Cleanup/Refactoring
- <Ideas for standardizing folder structure, removing duplicate code, and aligning with the v2 design plan>
```

---

## Submodules to Document
Based on exploration of the `/home/sujithma/mitra` workspace, we have identified the following submodules:

1. **`backend`**: FastAPI backend server coordinating sessions, upload, and metadata generation.
2. **`frontend`**: React/Vite web application frontend.
3. **`epic_1/mitra-ai`**: React prototype mockup of MITRA v1.
4. **`epic_3/model_selection`**: Orchestrator and agents for LLM-based model selection.
5. **`epic_3/training`**: Local training worker adapting library models to datasets.
6. **`epic_3/training_orchestrator`**: Router and manifest generator mapping selected models to training jobs.
7. **`epic_4/dataset2Vec`**: PyTorch encoder, sampler, and metadata knowledge base generator.
8. **`epic_4/judge_agent`**: Hybrid rule-based and LLM-based model evaluation/ranking agent.
9. **`epic_4/overfitting_analysis_tool`**: K-Fold cross validation and overfitting detection engine.
10. **`model_library`**: Core package containing estimators wrapper registry, metrics evaluators, and config loaders.

---

## Action Plan

### Step 1: Confirm Design Plan Details
We have read the design plan `MITRA_v2_Design_Plan 1.pdf` and its text extraction. We will align all interfaces, schemas, and flow paths with Section 4 (End-to-End Pipeline) and Section 6 (Agent I/O Contracts) of the design plan.

### Step 2: Create Submodule Cards
We will write the `CARD.md` files sequentially:
- `backend/CARD.md`
- `frontend/CARD.md`
- `epic_1/mitra-ai/CARD.md`
- `epic_3/model_selection/CARD.md`
- `epic_3/training/CARD.md`
- `epic_3/training_orchestrator/CARD.md`
- `epic_4/dataset2Vec/CARD.md`
- `epic_4/judge_agent/CARD.md`
- `epic_4/overfitting_analysis_tool/CARD.md`
- `model_library/CARD.md`

### Step 3: Verify & Update .gitignore
Ensure that these newly created `CARD.md` files are NOT gitignored, while other temporary session/harness outputs remain ignored.

### Step 4: Final Summary
Provide a comprehensive summary of all submodule interfaces, files created, and answer the user's question on token saving explicitly.
