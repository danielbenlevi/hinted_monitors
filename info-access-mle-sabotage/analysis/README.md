# CoT Monitoring Analysis Scripts

This folder contains analysis scripts for the Chain-of-Thought monitoring experiments described in the ICLR 2026 paper.

## Structure

- `data/` — Extracted scores and results data
- `filter_logs/` — Utilities for filtering and cleaning log files
- `plotting/` — Scripts for generating figures

Note: The EAE (Extract-and-Evaluate) monitor scripts are in `mlebench_subversion/monitoring/`.

## Usage

Analysis and plotting scripts expect log files to be available locally. Download logs from Dropbox (see main README) before running analysis.

**Note:** Some scripts have `DEFAULT_INPUT_DIR` constants pointing to `analysis/logs/`. You may need to update these paths or use command-line arguments to point to your local log directory.
