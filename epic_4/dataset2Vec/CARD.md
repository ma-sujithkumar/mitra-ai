---
name: epic_4/dataset2Vec
path: epic_4/dataset2Vec
purpose: Permutation-invariant hierarchical set-network (Dataset2Vec) designed to compute fixed-length vector embeddings of datasets, matching similar datasets to historical leaderboard configurations.
interfaces:
  inputs:
    - name: corpus_dir
      format: Directory containing *.npz files
      upstream: Meta-dataset crawler / curation script
      description: Training set of tabular datasets converted to matrices.
    - name: config.yaml
      format: YAML
      upstream: config
      description: Hyperparameters for blocks (f_block, g_block, h_block), training loops, and store specifications.
  outputs:
    - name: encoder.pt
      format: PyTorch Model checkpoint (state dict)
      downstream: meta-knowledge database retrieval
      description: Serialized weights of the trained permutation-invariant encoder.
    - name: metadata_knowledge_base (embeddings.parquet + FAISS index)
      format: Parquet / FAISS Index
      downstream: model_selection / recommendation engine
      description: Nearest-neighbor search database linking dataset embeddings with historical model performance.
entry_points:
  - name: epic_4/dataset2Vec/train_encoder.py
    type: CLI
    description: Master script training the Dataset2Vec encoder and populating the FAISS knowledge base.
  - name: epic_4/dataset2Vec/d2v_core.encoder:Dataset2VecEncoder
    type: PyTorch nn.Module
    description: Deep learning model implementing the permutation-invariant three-layer set-network architecture.
  - name: epic_4/dataset2Vec/d2v_core.store:MetaKnowledgeStore
    type: Python API
    description: Writes embeddings, loads FAISS index, and performs similarity searches.
dependencies:
  - torch
  - faiss-cpu / faiss-gpu
  - numpy
  - pandas
  - pyyaml
---

# Technical Architecture: Dataset2Vec

## Overview
`dataset2Vec` is a deep learning submodule implementing the Dataset2Vec architecture. It learns fixed-length vector representations of entire datasets. The learned embeddings are invariant to column permutations and row sorting. Once trained, these embeddings are registered in a FAISS indexing store to retrieve similar past datasets and recommend best-performing model configurations.

## Core Component Walkthrough
1. **`d2v_core/encoder.py`**:
   - `ResidualMLPBlock`: A Multi-Layer Perceptron containing optional residual connections.
   - `Dataset2VecEncoder`: Permutation-invariant hierarchical model consisting of three stages:
     - `f_net` (cell-level embedding): maps individual data cells and targets to hidden representations.
     - `g_net` (feature-level embedding): pools cell representations column-wise, encoding column-specific distributions.
     - `h_net` (dataset-level embedding): pools feature representations, creating the final `embedding_dim` vector.
2. **`d2v_core/sampling.py`**:
   - `CorpusSampler`: Dynamically samples patches (subsets of columns/rows) during training to handle varying sizes.
3. **`d2v_core/store.py`**:
   - `MetaKnowledgeStore`: Serializes/deserializes dataset embeddings and leverages FAISS indexers to implement nearest-neighbor search.

## Interfacing Guide
- **Upstream Integration:** Feeds on pre-split `.npz` files of diverse tabular datasets.
- **Downstream Integration:** The generated metadata store is queried by the selection orchestrator to fetch the best ML models based on past runs.

## Suggested Cleanup/Refactoring
- **Integration with MetadataGen:** Standardize embedding lookups inside the backend's model recommendation routers.
- **Ray-based training:** Distribute hyperparameter sweeps in `sweep.py` over a Ray cluster rather than running them in single-threaded processes.
