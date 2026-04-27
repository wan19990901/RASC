import argparse
import json
import os
from copy import deepcopy

import pandas as pd

from CS_based_early_stopping import evaluate_rasc_pipeline


DEFAULT_FEATURE_LIST = [
    "LEN",
    "QUA_IM",
    "DIF_IV",
    "SIM_COT_BIGRAM",
    "SIM_COT_AGG",
    "SIM_AC_BIGRAM",
    "SIM_AC_AGG",
    "SIM_INPUT",
    "STEP_COUNT",
    "STEP_COHERENCE",
]


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def slugify_experiment_name(name):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def build_runs(config):
    shared = deepcopy(config.get("shared", {}))
    experiments = config.get("experiments", [])

    for experiment in experiments:
        merged = deepcopy(shared)
        merged.update(experiment)

        thresholds = merged.pop("thresholds", None)
        if thresholds is None:
            thresholds = [merged.pop("threshold")]

        buffer_sizes = merged.pop("buffer_sizes", None)
        if buffer_sizes is None:
            buffer_sizes = [merged.pop("buffer_size", 5)]

        for threshold in thresholds:
            for buffer_size in buffer_sizes:
                run_config = deepcopy(merged)
                run_config["threshold"] = threshold
                run_config["buffer_size"] = buffer_size
                run_config.setdefault("feature_list", DEFAULT_FEATURE_LIST)
                yield run_config


def run_experiments(config):
    output_root = config.get("output_root", "../result/reproduction_runs")
    os.makedirs(output_root, exist_ok=True)

    summaries = []
    for run in build_runs(config):
        experiment_name = run["name"]
        experiment_slug = slugify_experiment_name(experiment_name)
        output_dir = os.path.join(output_root, experiment_slug)
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"threshold_{run['threshold']}_buffer_{run['buffer_size']}.csv"
        output_path = os.path.join(output_dir, output_filename)

        _, summary = evaluate_rasc_pipeline(
            data_path=run["data_path"],
            threshold=run["threshold"],
            buffer_size=run["buffer_size"],
            feature_list=run.get("feature_list", DEFAULT_FEATURE_LIST),
            score_mode=run.get("score_mode", "custom"),
            custom_intercept=run.get("custom_intercept", -0.6),
            custom_coefficients=run.get("custom_coefficients"),
            output_path=output_path,
        )

        summary["experiment_name"] = experiment_name
        summary["prompting"] = run.get("prompting", "unspecified")
        summary["dataset"] = run.get("dataset", "unspecified")
        summary["model"] = run.get("model", "unspecified")
        summary["data_path"] = run["data_path"]
        summary["output_path"] = output_path
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(output_root, "experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved experiment summary to: {summary_path}")
    return summary_df, summary_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run named paper-style RASC experiment presets from a JSON config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON experiment config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    run_experiments(config)


if __name__ == "__main__":
    main()
