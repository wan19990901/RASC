import os
import sys
import unittest

import pandas as pd


SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from CS_based_early_stopping import CS_early_stopping


class TestRASCAlgorithm(unittest.TestCase):
    def test_buffer_stopping_and_best_rationale_selection(self):
        df = pd.DataFrame(
            [
                {
                    "CoT_0": "Step 1: Wrong reasoning",
                    "CoT_1": "Step 1: Correct reasoning A",
                    "CoT_2": "Step 1: Correct reasoning B",
                    "CoT answers": ["b", "a", "a"],
                    "confidence_score": [0.20, 0.92, 0.81],
                    "correct answer": "a",
                    "SC_correctness": 1,
                    "ES_correctness": 1,
                    "ES_steps": 3,
                    "asc_correctness": 1,
                    "asc_steps": 3,
                }
            ]
        )

        result = CS_early_stopping(df.copy(), threshold=0.8, N=2)
        row = result.iloc[0]

        self.assertEqual(row["CS_steps"], 3)
        self.assertEqual(row["CS_buffer_size"], 2)
        self.assertEqual(row["CS_stop_reason"], "buffer_full")
        self.assertEqual(row["CS_Answer"], "a")
        self.assertEqual(row["CS_correctness"], 1)
        self.assertEqual(row["CS_Best_RP"], "Step 1: Correct reasoning A")
        self.assertAlmostEqual(row["CS_Best_Score"], 0.92, places=6)


if __name__ == "__main__":
    unittest.main()
