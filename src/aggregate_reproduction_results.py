import argparse
import os

import pandas as pd


def aggregate_results(input_root, output_path=None):
    summary_path = os.path.join(input_root, "experiment_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
    else:
        rows = []
        for dirpath, _, filenames in os.walk(input_root):
            for filename in filenames:
                if not filename.endswith(".csv"):
                    continue
                if filename == "aggregated_summary.csv":
                    continue
                csv_path = os.path.join(dirpath, filename)
                df_run = pd.read_csv(csv_path)
                if not {"SC_correctness", "ES_correctness", "CS_correctness", "asc_correctness"}.issubset(
                    df_run.columns
                ):
                    continue
                rows.append(
                    {
                        "output_path": csv_path,
                        "SC_ACC": df_run["SC_correctness"].mean(),
                        "ES_ACC": df_run["ES_correctness"].mean(),
                        "CS_ACC": df_run["CS_correctness"].mean(),
                        "ASC_ACC": df_run["asc_correctness"].mean(),
                        "SC_Avg_Steps": 40,
                        "ES_Avg_Steps": df_run["ES_steps"].mean(),
                        "CS_Avg_Steps": df_run["CS_steps"].mean(),
                        "ASC_Avg_Steps": df_run["asc_steps"].mean(),
                        "num_questions": len(df_run),
                    }
                )
        df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No experiment results found under: {input_root}")

    ordered_cols = [
        col
        for col in [
            "experiment_name",
            "dataset",
            "model",
            "prompting",
            "threshold",
            "buffer_size",
            "score_mode",
            "num_questions",
            "SC_ACC",
            "ES_ACC",
            "CS_ACC",
            "ASC_ACC",
            "SC_Avg_Steps",
            "ES_Avg_Steps",
            "CS_Avg_Steps",
            "ASC_Avg_Steps",
            "data_path",
            "output_path",
        ]
        if col in df.columns
    ]
    df = df[ordered_cols]

    if output_path is None:
        output_path = os.path.join(input_root, "aggregated_summary.csv")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved aggregated summary to: {output_path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate RASC reproduction runs into a single CSV summary."
    )
    parser.add_argument(
        "--input_root",
        required=True,
        help="Directory containing experiment outputs or experiment_summary.csv.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional destination CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    aggregate_results(args.input_root, args.output_path)


if __name__ == "__main__":
    main()
