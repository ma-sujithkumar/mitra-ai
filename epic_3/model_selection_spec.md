# MITRA v2: Model Selection Call, Component Specification

| Field | Value |
|---|---|
| Component | Model Selection Call (`mitra/agents/model_selection_call/`) |
| Owner | P6 (Onkar Biyani) |
| Type | Single LLM call via LiteLLM factory; no tools, no loop, no agent state |
| Spec status | Draft v0.1 for team review |
| Sources | MITRA v2 Design Plan §2, §4.1, §4.2, §6, §10, §11, §12, §17 (R2, R12), §19; sprint call transcript |
| Suggested repo home | `mitra/agents/model_selection_call/spec.md` |

## 1. Executive summary

Model Selection converts a dataset profile into a ranked shortlist of trainable model candidates. It reads `metadata.json`, `feature_selection.json`, and the `mini_data.csv` statistics file, makes exactly one LLM call through the LiteLLM factory, and writes `model_config.json` for the Training Orchestrator, the family agents, and the HPT agent. It is a function, not an agent; the ADK runner invokes it as a single turn with no tools attached, which closes risk R2.

Two design commitments drive everything below. First, the LLM emits plain hyperparameter ranges; a deterministic Python normalizer maps them onto canonical parameter names and typed Optuna dictionaries, so prompt mistakes cannot corrupt the HPT contract. Second, this stage never halts the pipeline; invalid output earns one retry, and a rule-based fallback list guarantees a usable `model_config.json` even with the LLM fully down. Python does the enforcement; the LLM only contributes judgment. This follows the team rule from the sprint call: wherever a script can do the job, the script does the job.

## 2. Scope

In scope: the prompt, the input contracts, the output schema, validation, hp_space normalization, retry and fallback behavior, SSE events, token budget, dev fixtures, and unit tests for the single call described in Design Plan §10.

Out of scope: Training Orchestrator routing (including the R13 USL leakage check), family agents, template rendering, HPT sampling, the Judge, and BYOM key handling. Their contracts appear here only where this component must satisfy them.

## 3. Position in the pipeline

| Direction | Stage | Artifact | Owner |
|---|---|---|---|
| Upstream | Metadata Gen | `metadata.json` (validated against §4.2 schema) | P4 |
| Upstream | Feature Selection | `feature_selection.json` | P5 |
| Upstream | mini_data generator | `mini_data.csv` (per-column statistics) | P3 |
| Downstream | Training Orchestrator | reads `model_config.json`, routes by `problem_type` | P6 |
| Downstream | Family agents | read `model_config.json[i]`, render templates | P6 |
| Downstream | HPT Agent | reads `model_config.json[i].hp_space` (typed, final) | P7 |

All artifacts live under `.mitra/<session_id>/`. This component reads three files and writes one.

## 4. Inputs

### 4.1 metadata.json

Validated by P4 against the §4.2 JSON Schema before this stage runs; this component trusts it and does not revalidate. Fields consumed here:

- `problem_type`: `classification | regression | unsupervised`; selects the allowed family set.
- `data_format`: `tabular | image`; `image` restricts the family set to `cnn_image`.
- `output_cols`: non-empty forbids unsupervised families; empty forbids everything else.
- `row_count`, `col_count`, `col_types`, `class_balance`: injected into the prompt as ranking signals.
- `user_description`: injected into the prompt verbatim.

### 4.2 feature_selection.json

Shape per §6: `{keep: [], drop: [], engineered: [{name, formula}], rationale: {col: reason}}`. Consumed as: `keep` and `engineered` define the model input dimensionality and filter the mini_data excerpt; `rationale` is not forwarded to the prompt (noise, token cost).

### 4.3 mini_data.csv excerpt

`mini_data.csv` is the per-column statistics table (describe output plus dtype, null_count, unique_count, and is_pii_suspect per column; assumption A1, see §15). The prompt never receives the full file. The excerpt rule:

1. Keep only rows for columns in `keep` + `engineered[].name` + `output_cols`. This is an intersection filter: engineered features are created after P3 generates `mini_data.csv`, so they have no statistics row and silently match nothing.
2. Serialize as CSV.
3. If the serialized excerpt exceeds 2,000 tokens, drop the 25%, 50%, and 75% percentile columns first, then truncate to the first 30 rows.

## 5. Output: model_config.json

One JSON array, 1 to 8 entries, written atomically (`model_config.json.tmp`, then `os.replace`).

### 5.1 Stored schema (what consumers read)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "array",
  "minItems": 1,
  "maxItems": 8,
  "items": {
    "type": "object",
    "required": ["name", "family", "rationale", "hp_space", "priority"],
    "additionalProperties": false,
    "properties": {
      "name":      {"type": "string", "pattern": "^[a-z0-9_]{3,40}$"},
      "family":    {"enum": ["xgboost", "random_forest", "logistic_reg", "svm",
                             "mlp", "cnn_image", "kmeans", "dbscan", "isolation_forest"]},
      "rationale": {"type": "string", "maxLength": 300},
      "priority":  {"type": "integer", "minimum": 1, "maximum": 8},
      "hp_space": {
        "type": "object",
        "maxProperties": 6,
        "additionalProperties": {
          "oneOf": [
            {"type": "object", "required": ["type", "low", "high"],
             "properties": {"type": {"enum": ["int", "float"]},
                            "low": {"type": "number"}, "high": {"type": "number"},
                            "step": {"type": "number"}, "log": {"type": "boolean"}}},
            {"type": "object", "required": ["type", "choices"],
             "properties": {"type": {"const": "categorical"},
                            "choices": {"type": "array", "minItems": 1}}}
          ]
        }
      }
    }
  }
}
```

Key semantics:

- The stored `hp_space` is the final, merged, typed search space (LLM ranges intersected with template defaults; see §7). The HPT agent reads it directly and needs no further lookup; `hp_space.yaml` remains the defaults source that feeds the merge, not a runtime dependency for P7.
- `priority` is derived: always `index + 1` after validation and repair. LLM-supplied priority values are ignored. Array order is the single ranking signal; a separate trusted priority field would only invite disagreement between the two.
- `name` is a free-form label for the leaderboard. All downstream resolution keys on `family`, never on `name` (decision D6, §16).

### 5.2 Wire schema (what the LLM returns)

The prompt requests a simpler shape; the normalizer produces the stored form:

```json
[{"name": "xgboost_baseline",
  "family": "xgboost",
  "rationale": "Tabular, mixed scales, 150 rows; boosted trees are the strongest default.",
  "hp_space": {"n_estimators": [50, 300], "max_depth": [3, 8], "learning_rate": [0.01, 0.3]}}]
```

Wire conventions: a 2-element numeric array is a range; any array containing a string, and any numeric array of 3 or more elements, is a categorical choice list; `hp_space` may be omitted entirely to accept template defaults.

## 6. Selection rules (enforced in Python, injected into the prompt)

The allowed family set is computed by pure Python from metadata and injected into the prompt. The LLM never derives it.

| problem_type | data_format | Allowed families | Candidate count |
|---|---|---|---|
| classification | tabular | xgboost, random_forest, logistic_reg, svm, mlp | 3 to 8 |
| regression | tabular | xgboost, random_forest, svm, mlp | 3 to 8 |
| unsupervised | tabular | kmeans, dbscan, isolation_forest | 2 to 3 |
| classification | image | cnn_image | 1 to 3 |

Hard guards:

- `output_cols` non-empty: unsupervised families forbidden (mirrors the §2 resolved ambiguity and complements the orchestrator's R13 check).
- `output_cols` empty: only unsupervised families allowed.
- `data_format == image`: only `cnn_image`. The §10 instruction "return between 3 and 8 models" is unsatisfiable here with a closed enum of one image family; this spec relaxes it (decision D7, §16).
- `class_balance` imbalance over 5:1 is surfaced to the prompt as a ranking hint only; forcing `class_weight='balanced'` stays the Classification agent's job per §11.

## 7. hp_space normalization (deterministic Python)

Resolves the Design Plan's internal contradiction: §10 shows `"lr": [0.01, 0.3]` as a plain array, while §12.2 templates use `learning_rate` with typed Optuna dicts. The wire format is plain (§5.2); the normalizer owns typing and naming.

### 7.1 Canonical parameter names

Proposed canon, to be confirmed with P1 (template owner) by D6. The alias map absorbs drift in the meantime.

| Family | Canonical tunables |
|---|---|
| xgboost | n_estimators, max_depth, learning_rate, subsample, colsample |
| random_forest | n_estimators, max_depth, min_samples_split, max_features |
| logistic_reg | C, penalty, max_iter |
| svm | C, kernel, gamma |
| mlp | learning_rate, hidden_dim, n_layers, dropout, epochs |
| cnn_image | learning_rate, batch_size, epochs, base_channels |
| kmeans | n_clusters |
| dbscan | eps, min_samples |
| isolation_forest | n_estimators, contamination |

Alias map (applied case-insensitively before validation): `lr -> learning_rate`, `eta -> learning_rate`, `n_est | num_estimators -> n_estimators`, `max_dep -> max_depth`, `k | num_clusters -> n_clusters`, `colsample_bytree -> colsample`.

### 7.2 Merge algorithm

For each candidate, for each param after aliasing:

1. Load `mitra/templates/{family}/hp_space.yaml` if present.
2. Param exists in template: take the template dict wholesale (`type`, `log`, `step`), then narrow bounds: `low = max(low_llm, low_tpl)`, `high = min(high_llm, high_tpl)`. Empty intersection: keep the template bounds, warn `MS_HP_EMPTY_INTERSECTION`. Categorical: intersect choices; empty intersection keeps template choices, warn.
3. Param absent from an existing template: drop it, warn `MS_HP_UNKNOWN_PARAM`. Templates define the tunable set; an unknown key would silently no-op downstream, which is worse than dropping it loudly.
4. Whole template folder missing: keep all params, infer types (`int` if both bounds are ints, else `float`; `log: true` for positive float ranges with `high/low >= 100`), warn `MS_TEMPLATE_MISSING`.
5. LLM omitted `hp_space` or every param got dropped: copy the template `hp_space.yaml` wholesale.
6. Cap at 6 params per model (Optuna gets 20 trials per §14; more dimensions than that is theater). Drop extras in template order, warn.

## 8. Prompt (agents/model_selection_call/prompt.md)

```text
You are an AutoML expert selecting candidate models for an automated training pipeline.

## Dataset profile (metadata.json)
{metadata_json}

## Selected features (feature_selection.json: keep + engineered)
{features_compact}

## Column statistics (kept columns only)
{mini_data_excerpt}

## Task
Select between {n_min} and {n_max} candidate models. Order them best first by
expected validation performance on this dataset.

## Output format (JSON array only, no prose, no markdown fences)
[{"name": "xgboost_baseline",
  "family": "xgboost",
  "rationale": "Tabular, mixed scales, 150 rows; boosted trees are the strongest default.",
  "hp_space": {"n_estimators": [50, 300], "max_depth": [3, 8], "learning_rate": [0.01, 0.3]}}]

## Rules
- "family" must be one of: {allowed_families}. Anything else is discarded.
- Use only these parameter names: {canonical_params_for_allowed}.
- Numeric ranges are 2-element arrays [low, high]. Choice lists contain strings
  or 3+ numbers. Omit hp_space to accept defaults.
- 2 to 6 parameters per model. Rationale under 40 words.
- Output ONLY the JSON array.
```

Retry suffix (attempt 2 only): `Your previous response failed validation: {error}. Output only the corrected JSON array.`

LLM call parameters: client from the LiteLLM factory (the only LLM path in the architecture; no direct HTTP), `temperature=0.2`, `max_tokens=1500`, `seed=42` passed best-effort, timeout 60 s per attempt.

## 9. Control flow

```python
def select_models(session_dir: Path, cfg: SessionConfig, llm: LiteLLMClient) -> Path:
    meta  = read_json(session_dir / "metadata.json")
    fsel  = read_json(session_dir / "feature_selection.json")
    mini  = read_csv(session_dir / "mini_data.csv")

    allowed = allowed_families(meta)                  # pure Python, table in §6
    n_min, n_max = count_bounds(meta)
    prompt = render_prompt(meta, fsel, excerpt(mini, fsel, meta), allowed, n_min, n_max)
    emit_event("model_selection", "selecting candidate models", pct=0)

    cands, last_err = None, None
    for attempt in (1, 2):                            # §10: one retry on invalid output
        raw = llm.complete(prompt if attempt == 1 else prompt + retry_suffix(last_err))
        try:
            cands = parse_wire(raw)                   # V1, V2
            break
        except SpecError as e:
            last_err = e
            emit_event("model_selection", f"attempt {attempt} invalid: {e.code}", level="warn")

    if cands is None:
        cands = fallback_list(meta)                   # §11; MS_FALLBACK_USED warn
    emit_event("model_selection", "validating", pct=40)

    cands = repair(cands, allowed, n_min, n_max, meta)   # V3..V6: drop, dedupe, truncate, top up
    for c in cands:
        c.hp_space = normalize_hp(c.family, c.hp_space)  # §7
    for i, c in enumerate(cands):
        c.priority = i + 1
    emit_event("model_selection", "normalized hp spaces", pct=70)

    atomic_write_json(session_dir / "model_config.json", cands)
    emit_event("model_selection", f"wrote {len(cands)} candidates; "
               f"tokens_in={llm.usage.input} tokens_out={llm.usage.output}", pct=100)
    return session_dir / "model_config.json"
```

ADK wiring: registered on the Root Orchestrator as a FunctionTool invoked in a single turn with no tools attached (R2 mitigation, verbatim from §17). A timeout counts as a failed attempt.

## 10. Validation and repair rules

Repair beats retry for countable defects; a retry costs tokens and adds nondeterminism, a repair costs nothing (decision D4).

| ID | Check | On failure | Event code |
|---|---|---|---|
| V1 | Response parses as a JSON array | retry once, then fallback | MS_JSON_PARSE |
| V2 | Each item has name, family, rationale (hp_space optional); types correct | retry once, then fallback | MS_SCHEMA |
| V3 | `name` matches `^[a-z0-9_]{3,40}$`; missing rationale | slugify name; rationale = "" | MS_NAME_FIXED |
| V4 | `family` in the global enum | drop entry | MS_BAD_FAMILY |
| V5 | `family` in allowed set for this metadata | drop entry | MS_FAMILY_MISMATCH |
| V6 | Duplicate (family, name) | keep first | MS_DUP |
| V7 | Count > n_max | truncate tail | MS_COUNT_HIGH |
| V8 | Count < n_min after drops | top up from fallback table, skipping families already present | MS_COUNT_LOW |
| V9 | Numeric range with low >= high | swap if obvious, else drop param | MS_HP_BAD_RANGE |

All repair events are `level="warn"` SSE events; nothing here ever aborts the pipeline.

## 11. Deterministic fallback (no LLM)

Used when both attempts fail (full list) or to top up a short list (V8). Each entry ships an empty wire `hp_space`, so §7 step 5 fills it from template defaults.

| Condition | Fallback list, in priority order |
|---|---|
| classification + tabular | xgboost, random_forest, logistic_reg |
| regression + tabular | xgboost, random_forest, mlp |
| unsupervised + tabular | kmeans, dbscan, isolation_forest |
| classification + image | cnn_image |

Names take the form `{family}_fallback`; rationale: `"deterministic fallback: LLM selection unavailable or insufficient"`.

## 12. Events and token budget

- SSE stage name: `model_selection`. Progress: 0, 40, 70, 100 per §9; warns per §10.
- Token budget (supports R12 and the 50K-token session acceptance criterion): prompt hard cap 4,000 tokens (the §4.3 excerpt rule is the pressure valve), completion cap 1,500, two attempts worst case. Ceiling roughly 11K tokens; typical run near 3K.
- Usage is reported in the final SSE event (`tokens_in`, `tokens_out` in `msg`); field placement to be confirmed with P2, see §15.

## 13. Dev fixtures (unblocks development before P3, P4, and P5 deliver)

Per the sprint call: assume the upstream formats, build against dummies. Files live in `fixtures/model_selection/` next to this spec; copy into a fake session dir for tests.

| File | Purpose |
|---|---|
| `iris_metadata.json` | classification + tabular happy path |
| `iris_feature_selection.json` | keep all 4 features, one engineered (`petal_area`) |
| `iris_mini_data.csv` | per-column stats table (assumption A1 orientation) |
| `image_metadata.json` | cats-dogs image path |
| `usl_metadata.json` | unsupervised path, `output_cols: []` |
| `expected_iris_model_config.json` | canonical stored-format example; golden file for T-MS-1 and T-MS-7 |

## 14. Unit test plan (pytest, `tests/test_model_selection.py`)

LLM responses are mocked strings; no live calls in CI.

| Test | Setup | Assert |
|---|---|---|
| T-MS-1 | iris fixtures, well-formed mock response | families include xgboost and random_forest (mirrors §19.1); count in [3, 8]; priorities exactly 1..N |
| T-MS-2 | image fixtures | every family == cnn_image; count in [1, 3] |
| T-MS-3 | usl fixtures | families subset of {kmeans, dbscan, isolation_forest} |
| T-MS-4 | first mock response is broken JSON, second valid | one retry, success, MS_JSON_PARSE warn emitted |
| T-MS-5 | both responses broken | fallback config written, MS_FALLBACK_USED warn, function returns normally |
| T-MS-6 | response contains family "mystery_net" | entry dropped (V4), list topped up (V8) |
| T-MS-7 | response uses `"lr": [0.01, 0.3]` | stored param is `learning_rate`, typed dict, `log` flag taken from template |
| T-MS-8 | `n_estimators: [1, 10000]` | clamped to template bounds [50, 500] |
| T-MS-9 | unknown param `magic: [0, 1]` | dropped, MS_HP_UNKNOWN_PARAM warn |
| T-MS-10 | mock returns 12 models | truncated to 8; priorities reassigned 1..8 |
| T-MS-11 | synthetic 100-column metadata | rendered prompt under the 4,000-token cap |
| T-MS-12 | kill the process between tmp write and replace (simulated) | no partial `model_config.json` ever visible |

## 15. Dependencies and asks

- P4 (Metadata): lock the §4.2 schema and add an explicit `schema_version` key (R9 implies one). I consume `class_balance` for the imbalance hint.
- P5 (Features): confirm `feature_selection.json` keys exactly as §6; engineered column names must exist in `data_encoded.csv` (your validation, per §9 of the plan).
- P3 (mini_data): assumption A1 says `mini_data.csv` is oriented one row per source column with stat fields as columns. Confirm orientation and column names; the §4.3 excerpt rule depends on it.
- P1 (Templates): confirm the §7.1 canonical parameter tables by D6; ship `hp_space.yaml` for all nine families. Also flagging: Figure 5 resolves templates by `name` while §11.1 reads `family`; this spec assumes `family` everywhere.
- P7 (HPT): you read the stored, already-merged `hp_space` from `model_config.json[i]`; `hp_space.yaml` is no longer a runtime input for you. Recommended initial-point rule for family agents rendering templates: geometric midpoint for log floats, arithmetic midpoint rounded for ints, first choice for categoricals.
- P2 (API/SSE): confirm where token usage lives in the §5.1 event schema.
- Vidhi (P8, UI): nothing needed from page one for this component. `user_description` arrives through `metadata.json`, and the 20-word minimum is already enforced client-side and at P4.

## 16. Decisions and deviations from the v2 plan

| ID | Decision | Why |
|---|---|---|
| D1 | Wire hp_space is plain arrays; stored hp_space is typed Optuna dicts; a Python normalizer converts | §10 and §12.2 contradict each other; smallest LLM surface wins |
| D2 | Canonical param names pinned in §7.1; aliases absorbed; unknown params dropped loudly | silent key mismatch would no-op the §11.1 merge |
| D3 | `priority` derived from array order; LLM priority ignored | two ranking signals cannot disagree if one does not exist |
| D4 | Repair over retry for countable defects; retry only for parse and schema failures | tokens and determinism |
| D5 | Family enum stays as §10. Ridge and Lasso (listed in §11 regression algorithms) are unreachable from selection | enum is the contract; family agents may map mlp and xgboost to -Reg variants; raise at standup if Ridge matters |
| D6 | Template resolution keys on `family`, not `name` | Figure 5 and §11.1 disagree; names are labels |
| D7 | Image path returns 1 to 3 candidates, unsupervised 2 to 3; "3 to 8" applies only to supervised tabular | a closed enum with one image family makes 3 distinct families impossible |
| D8 | mini_data excerpt filtered to kept + engineered + output columns, 2,000-token cap | token budget, and dropped columns are pure noise to ranking |
| D9 | This stage never aborts; worst case is the deterministic fallback | the demo must not die on the cheapest stage; Python over LLM |
| D10 | `model_config.json` stores the final merged search space; HPT reads no yaml | one source of truth at the P6/P7 boundary |
| D11 | Closed enum at selection means the T-NEG-6 write-from-scratch resolver path triggers only for hand-edited configs | keep the resolver fallback as defense in depth; do not delete that test |

Open question Q1: should the user be able to pin or exclude families from the UI (a P8 page-one control feeding metadata)? Out of scope for D8; park it.

