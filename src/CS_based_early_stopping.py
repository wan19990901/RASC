import argparse
import os
from collections import Counter

import pandas as pd

from IDV_CS_Model import customized_LR_model, trained_LR_model
from utils import calculate_ASC_correctness, calculate_ES_correctness, calculate_SC_correctness


def normalize_answer(answer):
    try:
        return float(answer)
    except (TypeError, ValueError):
        return str(answer).strip().lower()


def compare_answers(answer1, answer2):
    return normalize_answer(answer1) == normalize_answer(answer2)


def build_high_quality_buffer(row, threshold, capacity):
    buffer_entries = []
    total_seen = 0

    cot_cols = [col for col in row.index if col.startswith("CoT_")]

    for idx, (answer, score) in enumerate(zip(row["CoT answers"], row["confidence_score"])):
        total_seen = idx + 1
        cot_col = f"CoT_{idx}"
        rationale = row[cot_col] if cot_col in cot_cols else None

        if score >= threshold:
            buffer_entries.append(
                {
                    "step": idx,
                    "answer": answer,
                    "normalized_answer": normalize_answer(answer),
                    "score": score,
                    "rationale": rationale,
                }
            )

        if len(buffer_entries) >= capacity:
            break

    return buffer_entries, total_seen


def select_answer_and_rationale(buffer_entries, fallback_entries=None):
    candidates = buffer_entries if buffer_entries else (fallback_entries or [])
    if not candidates:
        return None, None, None

    weighted_votes = Counter()
    for entry in candidates:
        weighted_votes[entry["normalized_answer"]] += entry["score"]

    best_answer = max(weighted_votes, key=weighted_votes.get)
    supporting_entries = [
        entry for entry in candidates if entry["normalized_answer"] == best_answer
    ]
    best_entry = max(supporting_entries, key=lambda entry: entry["score"])

    return best_entry["answer"], best_entry["rationale"], best_entry["score"]


def build_fallback_entries(row):
    fallback_entries = []
    cot_cols = [col for col in row.index if col.startswith("CoT_")]
    for idx, (answer, score) in enumerate(zip(row["CoT answers"], row["confidence_score"])):
        cot_col = f"CoT_{idx}"
        rationale = row[cot_col] if cot_col in cot_cols else None
        fallback_entries.append(
            {
                "step": idx,
                "answer": answer,
                "normalized_answer": normalize_answer(answer),
                "score": score,
                "rationale": rationale,
            }
        )
    return fallback_entries


def CS_early_stopping(df, threshold, N=5):
    cs_answers = []
    cs_correctness = []
    cs_steps = []
    cs_best_rp = []
    cs_best_score = []
    cs_buffer_size = []
    cs_stop_reason = []

    for _, row in df.iterrows():
        buffer_entries, num_of_steps = build_high_quality_buffer(row, threshold, N)
        fallback_entries = build_fallback_entries(row)

        selected_answer, selected_rationale, selected_score = select_answer_and_rationale(
            buffer_entries,
            fallback_entries=fallback_entries,
        )

        cs_answers.append(selected_answer)
        cs_best_rp.append(selected_rationale)
        cs_best_score.append(selected_score)
        cs_steps.append(num_of_steps)
        cs_buffer_size.append(len(buffer_entries))
        cs_stop_reason.append(
            "buffer_full" if len(buffer_entries) >= N else "max_samples_reached"
        )
        cs_correctness.append(
            1 if compare_answers(selected_answer, row["correct answer"]) else 0
        )

    df["CS_Answer"] = cs_answers
    df["CS_correctness"] = cs_correctness
    df["CS_steps"] = cs_steps
    df["CS_Best_RP"] = cs_best_rp
    df["CS_Best_Score"] = cs_best_score
    df["CS_buffer_size"] = cs_buffer_size
    df["CS_stop_reason"] = cs_stop_reason

    df_model_comp_dict = {
        "SC_ACC": df.SC_correctness.sum() / len(df),
        "ES_ACC": df.ES_correctness.sum() / len(df),
        "CS_ACC": df.CS_correctness.sum() / len(df),
        "SC_Avg_Steps": 40,
        "ES_Avg_Steps": df.ES_steps.mean(),
        "CS_Avg_Steps": df.CS_steps.mean(),
        "ASC_Avg_Steps": df.asc_steps.mean(),
        "ASC_ACC": df.asc_correctness.sum() / len(df),
    }
    for key, val in df_model_comp_dict.items():
        print(f"{key} : {val}")

    return df


def evaluate_rasc_pipeline(
    data_path,
    threshold,
    buffer_size=5,
    feature_list=None,
    score_mode="custom",
    custom_intercept=-0.6,
    custom_coefficients=None,
    output_path=None,
):
    if feature_list is None:
        feature_list = [
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
    if custom_coefficients is None:
        custom_coefficients = [
            -0.17887917,
            -2.47526597,
            2.57520725,
            0.68997781,
            1.65216567,
            -2.61836719,
            -0.04469021,
            3.54958297,
            0.0,
            0.0,
        ]

    df_with_features = pd.read_json(data_path, lines=True)

    if score_mode == "trained":
        df_cs = trained_LR_model(
            df_with_features,
            feature_list,
            report_auroc=False,
        )
    else:
        if len(custom_coefficients) != len(feature_list):
            raise ValueError(
                "The number of custom coefficients must match the feature list length."
            )
        df_cs = customized_LR_model(
            df_with_features,
            feature_list,
            custom_coefficients,
            custom_intercept,
            report_auroc=False,
        )

    df_cs = calculate_SC_correctness(df_cs)
    df_cs = calculate_ES_correctness(df_cs, window_size=5)
    df_cs = calculate_ASC_correctness(df_cs)
    df_final = CS_early_stopping(df=df_cs, threshold=threshold, N=buffer_size)

    summary = {
        "threshold": threshold,
        "buffer_size": buffer_size,
        "score_mode": score_mode,
        "num_questions": len(df_final),
        "SC_ACC": df_final["SC_correctness"].mean(),
        "ES_ACC": df_final["ES_correctness"].mean(),
        "CS_ACC": df_final["CS_correctness"].mean(),
        "ASC_ACC": df_final["asc_correctness"].mean(),
        "SC_Avg_Steps": 40,
        "ES_Avg_Steps": df_final["ES_steps"].mean(),
        "CS_Avg_Steps": df_final["CS_steps"].mean(),
        "ASC_Avg_Steps": df_final["asc_steps"].mean(),
    }

    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"Saved results to: {output_path}")

    return df_final, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paper-faithful RASC early stopping on extracted features."
    )
    parser.add_argument("--data_path", required=True, help="Path to extracted feature JSONL.")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Sufficiency-score threshold T from the paper.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=5,
        help="High-quality buffer capacity N from the paper.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Optional CSV output path. Defaults to result/experiments_output/test_N_threshold/df_threshold_<T>_N_<N>.csv",
    )
    parser.add_argument(
        "--feature_list",
        nargs="+",
        default=[
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
        ],
        help="Feature columns used for sufficiency scoring.",
    )
    parser.add_argument(
        "--score_mode",
        choices=["custom", "trained"],
        default="custom",
        help="Use paper/demo coefficients or fit a scorer on a train split.",
    )
    parser.add_argument(
        "--custom_intercept",
        type=float,
        default=-0.6,
        help="Intercept for custom logistic scoring.",
    )
    parser.add_argument(
        "--custom_coefficients",
        nargs="+",
        type=float,
        default=[
            -0.17887917,
            -2.47526597,
            2.57520725,
            0.68997781,
            1.65216567,
            -2.61836719,
            -0.04469021,
            3.54958297,
            0.0,
            0.0,
        ],
        help="Coefficients for custom logistic scoring. Must match feature_list length.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output_path
    if output_path is None:
        storage_dir = "../result/experiments_output/test_N_threshold/"
        os.makedirs(storage_dir, exist_ok=True)
        output_path = os.path.join(
            storage_dir,
            f"df_threshold_{args.threshold}_N_{args.buffer_size}.csv",
        )
    evaluate_rasc_pipeline(
        data_path=args.data_path,
        threshold=args.threshold,
        buffer_size=args.buffer_size,
        feature_list=args.feature_list,
        score_mode=args.score_mode,
        custom_intercept=args.custom_intercept,
        custom_coefficients=args.custom_coefficients,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
