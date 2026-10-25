# Formal Analysis Hints Improve Weak-to-Strong Monitoring

This is the official codebase associated with the paper "*Back to Basics: Formal Analysis Hints Improve Weak-to-Strong Monitoring*".

## Requirements

- Python 3.10+ and `pip install -r requirements.txt`
- [`yara`](https://virustotal.github.io/yara/) (4.x) and
  [`gitleaks`](https://github.com/gitleaks/gitleaks) (8.x) on your `PATH`. On Debian/Ubuntu:

  ```bash
  sudo apt-get install -y yara
  curl -sSL https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz \
      | sudo tar -xz -C /usr/local/bin gitleaks
  yara --version && gitleaks version
  ```
- API keys, exported in the shell:
  - `OPENAI_API_KEY` — task-contract generation (GPT-4.1-mini) and OpenAI monitors
  - `OPENROUTER_API_KEY` — Qwen, GLM, MiniMax and Claude monitors

## Data

The `experiment.py` script expects a directory containing an `examples.jsonl` manifest
and one `<trajectory_id>/trajectory.eval` per trajectory as input. `experiment.py` scores every
trajectory the manifest lists.

### Malicious Trajectories

Malicious trajectories come from
[MonitoringBench](https://huggingface.co/datasets/neur26anonsub/ctrldataset2026). Download
`final_dataset.zip` directly and export using the provided script.

```bash
mkdir -p data/ctrldataset2026
# place the downloaded final_dataset.zip at data/ctrldataset2026/final_dataset.zip
python data_utils/load_dataset.py --output-dir data/monitoringbench --limit 0
```

By default this exports the benchmark's refined split.
`experiment.py` filters out any trajectories that were present in the development set.
To widen or narrow the export, pass `--all-source-names` for every source group, repeat
`--source-name` to pick specific subsets, or filter with `--side-task-name`, `--sample-uuid`
or `--task-id`. To score a random subset of an export, pass
`--malicious-limit N` (and `--benign-limit N`) to `experiment.py`, which samples under
`--random-seed`.

### Benign Trajectories

You may use any benign BashArena trajectories for calibration.
Provide a multi-sample Inspect `.eval` log of honest runs and converted with:

```bash
python -m data_utils.preprocess_native_benign \
    --input-eval path/to/honest_runs.eval --output-dir data/benign --limit 0
```

The converter will only keep a sample if that sample:

- is `eval_mode: honest` in the `BashArena` setting, with no `side_task_name` assigned and
  no active `side_task_success` score
- completed, with tool calls and results properly paired and nothing after `submit`
- opens with a user message beginning `Your task is as follows:` (BashArena convention)

## Hint Caches

The augmented protocol needs a task contract per distinct task and a formal analysis
product per trajectory. Generate these caches in `runs/final_run/` with:

```bash
python experiment.py --generate-final-caches \
    --benign-input data/benign --malicious-input data/monitoringbench
```

Add `--contract-cache-dir` / `--formal-cache-dir` to write somewhere other than the default.

## Running Experiments

One invocation runs all three protocols for one monitor model:

```bash
python experiment.py --model openai/gpt-4.1-mini \
    --benign-input data/benign --malicious-input data/monitoringbench \
    --exclude-action-only-hint --run-name gpt41mini
```

Paper reproduction settings for the six monitors:

| Monitor | `--model` | extra flags |
|---|---|---|
| GPT-4.1-mini | `openai/gpt-4.1-mini` | - |
| GLM-4.5-Air | `openrouter/z-ai/glm-4.5-air` | - |
| MiniMax M2.5 | `openrouter/minimax/minimax-m2.5` | - |
| Qwen3.6-Flash | `openrouter/qwen/qwen3.6-flash` | `--reasoning-effort high` |
| Claude Haiku 4.5 | `openrouter/anthropic/claude-haiku-4.5` | `--reasoning-tokens 16384` |
| GPT-5.4-mini | `openai/gpt-5.4-mini` | `--reasoning-effort high` |

Useful options:

- `--max-monitor-api-calls N` - concurrency, default 10, max 50
- `--only-variant eae_augmented` - run a single protocol

## Reading Results

Each run is written under `runs/quick_monitor_eval/<run-name>/`:

- `variants/<protocol>/per_sample_results.jsonl` — one line per trajectory with
  `split`, `final_score` (0–100) and the calibrated `threshold`
- `summary.json` / `summary.csv` — threshold and recall per protocol
