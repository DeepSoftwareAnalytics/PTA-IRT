# PTA-IRT

Self-contained package for **PTA-IRT** (Privileged Trajectory-Aware Item Response Theory).

## Layout

```
PTA-IRT/
  run_pta.py                 # Reproduce the main PTA-IRT experiment
  prepare_summary_data.py    # Matrix + MiniLM embeddings from summaries
  traj/
    summarize.py             # LLM four-section process summary
    prompts/summary_system.txt
  pta/                       # Models, selection, estimation, e2e
  data.zip                   # Matrices, embeddings, folds, frozen S
  traj_summaries.zip         # Process summaries (Lite / Verified / Full / Pro)
  results/_hp_ref_lupi.json
```

## Setup

```bash
pip install -r requirements.txt
```

GPU (CUDA) is recommended for `run_pta.py`. CPU works but is slower.

Summarization needs a DeepSeek (or OpenAI-compatible) API key:

```bash
export DEEPSEEK_API_KEY=sk-...
# or put DEEPSEEK_API_KEY=sk-... in keys.cfg
```

## Data

Unpack the archives at the repository root, then run the experiment:

```bash
unzip data.zip -d data
unzip traj_summaries.zip -d traj_summaries
```

`data.zip` contains matrices, MiniLM embeddings, CV folds, and frozen calibration sets for Lite, Verified, Full, and Pro. `traj_summaries.zip` contains the corresponding process summaries.

## Reproduce PTA-IRT

```bash
python run_pta.py --bench lite --skip-build
python run_pta.py --bench all --skip-build
python run_pta.py --bench lite --rebuild-selection
```

Outputs are written under `results/` (or `--out-json`).

## Process summaries (optional rebuild)

After the extracted action quadruples are ready, call `traj/summarize.py`. It uses DeepSeek-V4-Flash with the paper's fixed system prompt (`traj/prompts/summary_system.txt`, temperature 0, thinking disabled).

### Input record

One JSON object (or a `.json` file per instance). Required fields:

```json
{
  "instance_id": "astropy__astropy-12907",
  "task_goal": "<issue / problem statement>",
  "ground_truth": "<reference patch, or empty>",
  "prediction": "<agent submitted patch, or empty>",
  "traj_quality": "full",
  "quadruples": [
    {
      "step": 0,
      "reasoning": "<assistant thought>",
      "tool_name": "<tool name>",
      "parameters": "<serialized arguments>",
      "result": "<tool / observation output>"
    }
  ]
}
```

`traj_quality` is one of `full` / `degraded` / `eval_only` / `no_traj` (used later as reliability weights 1.0 / 0.1 / 0.1 / 0.0). If you already serialized steps in the paper format (`### step t` / `REASONING` / `TOOL` / `PARAMETERS` / `RESULT`), pass that text as `steps_blob` instead of `quadruples`.

Prompt truncation (same as the paper): 2,000 characters per quadruple field, 4,000 for the issue, 8,000 for each patch.

### Run the summarizer

```bash
python traj/summarize.py --input record.json --out instance_summary.json
python traj/summarize.py --input-dir quads/agent_x --out-dir traj_summaries/lite/agent_x --skip-existing
python traj/summarize.py --input record.json --dry-run
```

By default the output JSON keeps `instance_id`, `n_steps`, `traj_quality`, and `summary` (quadruples and patches are dropped). Pass `--keep-quadruples` to retain them.

## Embeddings and outcome matrix (optional rebuild)

`prepare_summary_data.py` encodes summaries with `all-MiniLM-L6-v2` and writes the files `run_pta.py` reads: `matrix.csv`, `summary_embeddings.pt`, `summary_mask.npy`, `quality_weight.npy`, `metadata.json`. Question embeddings are not required for PTA-IRT.

Rebuild against the packaged Lite layout (keeps agent/item order):

```bash
python prepare_summary_data.py \
  --summary-root traj_summaries/lite \
  --metadata data/swe_lite_summary/metadata.json \
  --matrix data/swe_lite_summary/matrix.csv \
  --out-dir data/swe_lite_summary
```

From scratch, supply instance order plus resolved labels:

```bash
# SWE-bench Lite / Verified / Full: results.json per submission folder
python prepare_summary_data.py \
  --summary-root traj_summaries/lite \
  --instance-ids instance_ids.json \
  --resolved-dir /path/to/evaluation/lite \
  --out-dir data/swe_lite_summary

# SWE-bench Pro: long CSV with columns model_slug, instance_id, resolved
python prepare_summary_data.py \
  --summary-root traj_summaries/pro \
  --instance-ids instance_ids.json \
  --results-csv agent_item_long.csv \
  --out-dir data/swe_pro_summary
```
