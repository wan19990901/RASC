import json
import os
import sys
import tempfile
import unittest

import pandas as pd


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from aggregate_reproduction_results import aggregate_results
from run_paper_experiments import run_experiments


class TestReproductionRunner(unittest.TestCase):
    def test_runner_writes_summary_and_outputs(self):
        sample_rows = [
            {
                "CoT_0": "Step 1: wrong",
                "CoT_1": "Step 1: right A",
                "CoT_2": "Step 1: right B",
                "CoT answers": ["b", "a", "a"],
                "Correctness": [0, 1, 1],
                "correct answer": "a",
                "confidence_score": [0.1, 0.9, 0.8],
                "SC_correctness": 1,
                "ES_correctness": 1,
                "ES_steps": 3,
                "asc_correctness": 1,
                "asc_steps": 3,
                "LEN": [0, 1, 1],
                "QUA_IM": [0, 0, 0],
                "DIF_IV": [0, 0, 0],
                "SIM_COT_BIGRAM": [0.5, 0.7, 0.8],
                "SIM_COT_AGG": [0.5, 0.8, 0.8],
                "SIM_AC_BIGRAM": [0, 0, 1],
                "SIM_AC_AGG": [0, 0, 1],
                "SIM_INPUT": [0.6, 0.7, 0.8],
                "STEP_COUNT": [1, 1, 1],
                "STEP_COHERENCE": [0.1, 0.2, 0.3],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test.json")
            pd.DataFrame(sample_rows).to_json(data_path, orient="records", lines=True)

            config = {
                "output_root": os.path.join(tmpdir, "outputs"),
                "shared": {
                    "data_path": data_path,
                    "score_mode": "custom",
                    "feature_list": ["LEN", "SIM_INPUT"],
                    "custom_intercept": -0.2,
                    "custom_coefficients": [0.5, 0.5],
                },
                "experiments": [
                    {
                        "name": "toy_run",
                        "dataset": "toy",
                        "model": "toy-model",
                        "prompting": "zero_cot",
                        "thresholds": [0.5],
                        "buffer_sizes": [2],
                    }
                ],
            }

            summary_df, summary_path = run_experiments(config)
            self.assertTrue(os.path.exists(summary_path))
            self.assertEqual(len(summary_df), 1)
            self.assertEqual(summary_df.iloc[0]["experiment_name"], "toy_run")

            aggregated = aggregate_results(config["output_root"])
            self.assertEqual(len(aggregated), 1)
            self.assertIn("CS_ACC", aggregated.columns)


if __name__ == "__main__":
    unittest.main()
