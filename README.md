# RASC: Reasoning-Aware Self-Consistency for LLM Reasoning

A novel framework that enhances sampling efficiency and reasoning faithfulness in Large Language Models (LLMs) through dynamic evaluation of outputs and rationales.

## Overview

RASC (Reasoning-Aware Self-Consistency) is designed to revolutionize LLM reasoning by:
- Dynamically evaluating both reasoning paths and final answers
- Optimizing sampling efficiency while maintaining high accuracy 
- Implementing intelligent sampling decisions and rationale selection
- Providing a comprehensive framework for reasoning assessment

## Repository Structure

```
├── src/
│   ├── experiment_collection/    # Experimental results from the paper
│   ├── prompt_file/             # System prompts and templates
│   ├── CS_based_early_stopping.py   # Early stopping implementation
│   ├── CS_feature_extractor.py      # Reasoning evaluation feature extraction
│   ├── IDV_CS_Model.py              # Core reasoning evaluation model
│   ├── LLM_agent.py                 # LLM interaction interface
│   ├── Parsers.py                   # Output parsing utilities
│   ├── SC_generator.py              # Self-consistency sample generator
│   ├── data_cleaning.py             # Data preprocessing utilities
│   ├── experiment.sh                # Full experiment execution script
│   ├── human_eval.md                # Human evaluation guidelines
│   ├── demo.ipynb                   # Getting started demonstration
│   └── utils.py                     # Utility functions
├── data/
│   ├── question_data/               # Original question datasets
│   │   └── preprocessed/            # Processed question data
│   ├── other_data.txt              # Supplementary data files
│   └── Evaluation_CoTs/sample_data  # Test sample data
├── result/
│   ├── experiments_output/          # Experimental results
│   └── Human_eval_data/             # CoT faithfulness evaluation data
└── requirements.txt                 # Project dependencies
```


## Features

- **Multi-Model Support**: Compatible with various LLM platforms including:
  - LLAMA (via ollama)
  - OpenAI models
  - Anthropic's Claude
- **Advanced Analysis Tools**: 
  - Comprehensive feature extraction for reasoning assessment
  - Optimized sampling with early stopping mechanisms
  - Customizable evaluation metrics

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Self-Consistency Chains (for extracting the answer):
```bash
python src/SC_generator.py
```
This step is only needed when you want to generate new CoT samples from scratch with an API-backed model or Ollama.
The files under `data/Evaluation_CoTs/sample_data/` are already-generated sample outputs, so you do not need an API key if you are only testing the downstream RASC pipeline on those samples.

2. Feature Extraction:
```bash
python src/CS_feature_extractor.py
```
This stage assumes you already have sampled CoTs and final answers in CSV form.
Use the provided sample data when you want to test feature extraction, scoring, and early stopping without running new LLM generations.

3. Run RASC early stopping:
```bash
python src/CS_based_early_stopping.py \
  --data_path data/Evaluation_CoTs/sample_data/new_extracted_data/test.json \
  --threshold 0.5 \
  --buffer_size 5 \
  --score_mode custom
```

4. Sweep thresholds and buffer sizes:
```bash
bash src/experiment.sh data/Evaluation_CoTs/sample_data/new_extracted_data/test.json custom
```

5. Run named reproduction presets from a config:
```bash
python src/run_paper_experiments.py --config configs/paper_reproduction_sample.json
```

6. Aggregate reproduction outputs into one table:
```bash
python src/aggregate_reproduction_results.py --input_root result/reproduction_runs
```

## Important Notes

1. **Getting Started**:
   - Begin with `demo.ipynb` before running the full experiment script
   - The notebook provides a comprehensive overview of the framework's capabilities

2. **Feature Extraction Optimization**:
   - Feature extraction can be computationally intensive
   - For initial testing, use the provided sample CoTs
   - Consider excluding "**_AGG" features to reduce processing time
   - Feel free to customize `feature_extraction.py` for additional features (update related files accordingly)

3. **Model Configuration**:
   - Current implementation uses custom LR regression on test sets
   - Coefficients are learned from training data
   - Options for improvement:
     - Train custom models for coefficient optimization (see `demo.ipynb`)
     - Implement more complex models via `IDV_CS_Model.py`

4. **Paper Alignment Notes**:
   - `CS_based_early_stopping.py` follows the paper's buffer-based Algorithm 1:
     - keep only samples with sufficiency score `>= T`
     - stop when the high-quality buffer reaches size `N`
     - choose the final answer by weighted voting over buffered samples
     - choose the best rationale as the highest-scoring buffered rationale supporting the final answer
   - Prompt templates now cover zero-shot CoT, few-shot, and least-to-most prompting.

5. **Reproduction Workflow**:
   - Put extracted feature JSONL files in a stable location, such as `data/Evaluation_CoTs/.../new_extracted_data/*.json`
   - Encode experiment grids in `configs/*.json`
   - Run all named presets through `src/run_paper_experiments.py`
   - Summarize outputs with `src/aggregate_reproduction_results.py`
   - The provided sample config is a template for structuring dataset/model/prompting sweeps; expand it with the full paper assets as they become available

6. **Sample Data vs. Fresh Generation**:
   - `data/Evaluation_CoTs/sample_data/` should be treated as already-generated CoT outputs
   - Use these sample files if you already have CoTs and want to evaluate feature extraction, sufficiency scoring, and RASC stopping
   - Use `src/SC_generator.py` only when you want to create new CoT samples from raw question datasets
   - API keys are needed for fresh generation, but not for the sample-data evaluation path


## Citation

If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{
wan2025reasoningawareselfconsistencyleveraging,
title={Reasoning Aware Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling},
author={Guangya Wan, Yuqi Wu, Jie Chen, Sheng Li},
booktitle={The 2025 Annual Conference of the Nations of the Americas Chapter of the ACL},
year={2025},
url={https://openreview.net/forum?id=ykXCRWB8DR}
}
```

## Contact

For questions and feedback, please open an issue in this repository or sent an email to wxr9et@virginia.edu for inquiry.

## Acknowledgments

We thank all ARR reviewers who helped improve this work.
