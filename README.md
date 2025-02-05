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
```

2. Feature Extraction:
```bash
python src/CS_feature_extractor.py
```

3. Run Experiments:
```bash
bash src/experiment.sh
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

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{
anonymous2025rasc,
title={{RASC}: Reasoning-Aware Self-Consistency for Efficient and Faithful {LLM} Reasoning},
author={Guangya Wan, Yuqi Wu, Jie Chen, Sheng Li},
booktitle={The 2025 Annual Conference of the Nations of the Americas Chapter of the ACL},
year={2025},
url={https://openreview.net/forum?id=ykXCRWB8DR}
}


## Citation

If you use this code in your research, please cite our paper:
```bibtex
@inproceedings{
anonymous2025rasc,
title={{RASC}: Reasoning-Aware Self-Consistency for Efficient and Faithful {LLM} Reasoning},
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



