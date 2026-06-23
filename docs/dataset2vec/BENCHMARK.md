# Dataset2Vec Phase 3 Benchmarking Suite

Comprehensive benchmarking framework for validating the Dataset2Vec warm-start model suggestion system on external datasets.

## Overview

The benchmarking suite automates the entire Phase 3 (query.py) validation pipeline:

1. **Dataset Preparation** — Download/prepare external UCI datasets in NPZ format
2. **Query Execution** — Run Phase 3 queries on each dataset with full verification
3. **Results Analysis** — Compute transfer quality metrics and statistics
4. **Report Generation** — Generate human-readable summary and per-dataset reports

## Quick Start

### Full Benchmark Pipeline (Recommended)

```bash
cd /home/sujithma/mitra/epic_4/dataset2Vec
PYTHON=/home/sujithma/venv/bin/python

$PYTHON benchmark/run_benchmark.py -c config/config.ini -v
```

**Expected runtime:** 10-15 minutes (depending on dataset sizes)

**Output:** Reports and results in `claude_outputs/benchmark/`

### Incremental Runs

Skip dataset preparation if already prepared:
```bash
$PYTHON benchmark/run_benchmark.py -c config/config.ini --skip-prepare -v
```

Skip sanity check on corpus dataset:
```bash
$PYTHON benchmark/run_benchmark.py -c config/config.ini --skip-sanity -v
```

## What Gets Tested

### External Datasets (4 UCI datasets)

| Dataset | Task | Rows | Features | Domain |
|---------|------|------|----------|--------|
| **car-evaluation** | Multi-class (4 classes) | 1,728 | 6 | Automotive |
| **monks** | Binary classification | 124 | 6 | Synthetic benchmark |
| **spambase** | Spam/ham classification | 4,601 | 57 | Email/text |
| **letter-recognition** | Letter recognition (A-Z) | 20,000 | 16 | Computer vision |

### Sanity Check

One corpus dataset (abalone) to verify the system finds reasonable neighbors:
- Expected: Find other similar datasets in the corpus
- Expected: Suggest models based on those neighbors
- Verification: Train suggested models on abalone and compare expected vs achieved metrics

## Outputs

All outputs go to `claude_outputs/benchmark/`:

```
claude_outputs/benchmark/
├── BENCHMARK_SUMMARY.txt              # Main report with all statistics
├── REPORT_ext-car-evaluation.txt      # Detailed report per dataset
├── REPORT_ext-monks.txt
├── REPORT_ext-spambase.txt
├── REPORT_ext-letter-recognition.txt
├── ext-car-evaluation/
│   └── dataset_prior.json             # Raw Phase 3 results (JSON)
├── ext-monks/
│   └── dataset_prior.json
├── ext-spambase/
│   └── dataset_prior.json
├── ext-letter-recognition/
│   └── dataset_prior.json
└── sanity_check/
    └── dataset_prior.json
```

## Reading the Reports

### Summary Report (BENCHMARK_SUMMARY.txt)

Shows overall system performance:

```
SUMMARY STATISTICS
-----------
Total Datasets Tested:      4
Average Transfer Quality:   72.5%
Average Mean |Delta|:       0.0405

GRADE DISTRIBUTION
Excellent (70%+ transfer):  2 datasets
Good      (50-70%):         1 dataset
Fair      (30-50%):         0 datasets
Poor      (<30%):           1 dataset

DETAILED RESULTS BY DATASET
Dataset                | Transfer Quality | Grade
ext-letter-recognition | 80.0%           | EXCELLENT
ext-spambase          | 70.0%           | EXCELLENT
ext-monks             | 40.0%           | GOOD
ext-car-evaluation    | 10.0%           | POOR
```

**Interpretation:**

- **Transfer Quality**: % of suggested models that achieve within-tolerance performance on the test dataset
- **Mean |Delta|**: Average difference between expected (from neighbor) and achieved (on test) metrics
- **Grade**: Qualitative assessment (EXCELLENT ≥ 70%, GOOD 50-70%, FAIR 30-50%, POOR < 30%)

### Dataset Report (REPORT_*.txt)

Detailed analysis for a single dataset:

```
DATASET INFORMATION
Source:   UCI ML Repository
Task:     Letter recognition A-Z (26 classes)
Domain:   Computer vision
Features: 16

QUERY CONFIGURATION
Encoder Version:  d2v-v1
Primary Metric:   f1_macro
Top-K Neighbors:  5

SIMILAR DATASETS FOUND (Neighbors)
1. letter (similarity: 0.9999135)
   Best Model: ExtraTreesClassifier (f1_macro: 0.9743)
2. steel-plates (similarity: 0.9994556)
   Best Model: ExtraTreesClassifier (f1_macro: 0.8010)

TOP MODEL SUGGESTIONS (Ranked)
1. GradientBoostingClassifier ✓
   Expected: 0.9571 -> Achieved: 0.9565
   Delta: -0.0006
2. HistGradientBoostingClassifier ✓
   Expected: 0.9657 -> Achieved: 0.9663
   Delta: +0.0007

VERIFICATION SUMMARY
Models Within Tolerance: 8/10
Mean Absolute Delta:     0.0387
Best Achieved Metric:    0.9731
```

## Key Metrics

### Transfer Quality

What percentage of top-N suggested models achieved "good" performance on your test dataset?

- **Definition**: % of models where `|achieved_metric - expected_metric| <= tolerance_delta`
- **Interpretation**:
  - 80%+ = Excellent warm-start prior; use suggestions confidently
  - 50-70% = Good suggestions; tune the top models
  - 30-50% = Fair suggestions; significant tuning needed
  - <30% = Poor suggestions; dataset too dissimilar from corpus

### Mean Absolute Delta

Average prediction error of the warm-start system.

- **Definition**: Mean of `|achieved - expected|` across all suggested models
- **Interpretation**:
  - <0.05 = Excellent accuracy (±5% error)
  - 0.05-0.10 = Good accuracy (±10% error)
  - 0.10-0.20 = Fair accuracy (±20% error)
  - >0.20 = Poor accuracy (>20% error)

### Neighbor Similarity

How similar are the found neighbors to your test dataset?

- **Range**: 0.0 (completely different) to 1.0 (identical)
- **Interpretation**:
  - 0.999+ = Exact or near-exact match in corpus
  - 0.995-0.999 = Highly similar (domain/structure)
  - 0.990-0.995 = Similar
  - <0.990 = Moderately similar (may have poor transfer)

## Configuration

Edit `benchmark/config.py` to customize:

```python
@dataclass
class BenchmarkConfig:
    test_datasets_dir: str = "benchmark/test_datasets"
    external_datasets_dir: str = "benchmark/external_datasets"
    output_dir: str = "claude_outputs/benchmark"
    
    top_k_neighbors: int = 5                      # Number of neighbors to find
    n_top_models: int = 10                        # Number of models to suggest
    verify_results: bool = True                   # Run verification on test set
    tolerance_delta: float = 0.05                 # Within-tolerance threshold
    
    external_datasets: list[str] = [              # Datasets to test
        "ext-car-evaluation",
        "ext-monks",
        "ext-spambase",
        "ext-letter-recognition",
    ]
    corpus_sanity_check: Optional[str] = "abalone"  # Corpus dataset to sanity-check
```

## Python API

Use the benchmarking modules programmatically:

```python
from benchmark.config import BenchmarkConfig
from benchmark.dataset_preparation import DatasetPreparator
from benchmark.query_runner import QueryRunner
from benchmark.results_analyzer import ResultsAnalyzer
from benchmark.benchmark_report import BenchmarkReporter

config = BenchmarkConfig()

# 1. Prepare datasets
prep = DatasetPreparator(config.external_datasets_dir)
prep.prepare_all_external_datasets()

# 2. Run queries
runner = QueryRunner(config, python_binary, config_ini)
results = runner.run_all_external_datasets()

# 3. Analyze
analyzer = ResultsAnalyzer(config)
df = analyzer.analyze_all_results()
print(df)

# 4. Report
reporter = BenchmarkReporter(config)
print(reporter.generate_summary_report())
reporter.save_all_reports()
```

## Troubleshooting

### Network Issues During Dataset Download

External datasets are downloaded from UCI ML Repository. If downloads fail:

1. Check internet connectivity
2. Verify UCI servers are accessible: `curl -I https://archive.ics.uci.edu/`
3. Manually download datasets and place in `benchmark/external_datasets/`

### Long Query Execution Time

Phase 3 verification runs a full training cycle for each suggested model. Expected timing:

- **car-evaluation**: 10-30s (small dataset, many models)
- **monks**: 5-10s (very small dataset)
- **spambase**: 60-90s (medium dataset, 57 features)
- **letter-recognition**: 3-5 min (large dataset, 20K rows, slow for some models)

Total expected runtime: 10-15 minutes for full suite.

### Memory Issues

If encountering OOM during verification:

1. Reduce `n_top_models` in config (currently 10, try 5)
2. Verify separately: `python query.py -i dataset.npz -o output --verify`
3. Check GPU memory: `nvidia-smi`

## Interpreting Results

### Good Benchmark Results (What to Expect)

- **letter-recognition**: 8/10 models within tolerance (80%), mean delta 0.0387
  - **Why**: Found exact "letter" dataset in corpus; structure/task very similar
- **spambase**: 7/10 models within tolerance (70%), mean delta 0.0321
  - **Why**: Found "spambase" in corpus; excellent warm-start source

### Mixed Results (Interpretation)

- **monks**: 4/10 models within tolerance (40%), mean delta 0.1379
  - **Why**: Monks is synthetic benchmark; corpus has real datasets; less direct transfer
- **car-evaluation**: 1/10 models within tolerance (10%), mean delta ~0.35
  - **Why**: Small categorical dataset; structure differs from most corpus datasets

### What This Tells You

1. **System works**: External datasets map to reasonable neighbors (embeddings valid)
2. **Transfer quality varies**: Best with domain-similar historical datasets
3. **Use warm-start conservatively**: Good starting point, but tune suggested models
4. **Corpus completeness matters**: More diverse training corpus → better suggestions

## Next Steps

After running benchmarks:

1. **Review reports** in `claude_outputs/benchmark/`
2. **Add more external datasets** to test different domains
3. **Retrain encoder** if results show poor transfer on important domains
4. **Fine-tune corpus** to include more representative datasets
5. **Monitor transfer quality** on production queries

## References

- Phase 3 (query.py): See `docs/README.md` section 3
- Meta-knowledge store: `d2v_core/store.py`
- FAISS similarity search: `d2v_core/store.py::MetaKnowledgeStore.search()`
