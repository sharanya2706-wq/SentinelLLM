import time
from pathlib import Path

import pandas as pd

from utils.dataset_loader import load_benchmark
from llm.client import generate_response
from llm.careershield import CAREERSHIELD_SYSTEM_PROMPT

from evaluation.hallucination import evaluate_hallucination
from evaluation.bias import evaluate_bias
from evaluation.toxicity import evaluate_toxicity
from evaluation.jailbreak import evaluate_jailbreak
from evaluation.prompt_injection import evaluate_prompt_injection
from evaluation.reasoning import evaluate_reasoning

from evaluation.response_storage import (
    save_response,
    get_completed_ids
)


# ========================================
# SELECT TEST CASES FOR TEST MODE
# ========================================

def select_test_cases(benchmark):
    """
    Select exactly 15 test cases for test mode.

    Coverage:
    - 3 Hallucination
    - 3 Bias
    - 3 Toxicity
    - 2 Jailbreak
    - 2 Prompt Injection
    - 2 Reasoning
    """

    category_limits = {
        "hallucination": 3,
        "bias": 3,
        "toxicity": 3,
        "jailbreak": 2,
        "prompt_injection": 2,
        "reasoning": 2
    }

    selected_cases = []

    for category, limit in category_limits.items():

        category_cases = benchmark[
            benchmark["category"] == category
        ].head(limit)

        selected_cases.append(
            category_cases
        )

    if not selected_cases:

        return benchmark.head(0)

    return pd.concat(
        selected_cases,
        ignore_index=True
    )


# ========================================
# RUN EVALUATION
# ========================================

def run_evaluation(
    dataset_path="data/benchmark.csv",
    endpoint=None,
    api_key=None,
    model_name=None,
    test_mode=False,
    progress_callback=None
):

    print(
        "\n===== Starting SentinelLLM Evaluation =====\n"
    )

    # ========================================
    # VALIDATE MODEL CONFIGURATION
    # ========================================

    if not endpoint:

        raise ValueError(
            "API endpoint is required to evaluate an LLM."
        )

    if not api_key:

        raise ValueError(
            "API key is required to evaluate an LLM."
        )

    if not model_name:

        raise ValueError(
            "Model name is required to evaluate an LLM."
        )

    # ========================================
    # VALIDATE DATASET PATH
    # ========================================

    dataset_path = Path(
        dataset_path
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    # ========================================
    # LOAD DATASET
    # ========================================

    print(
        f"Loading dataset: {dataset_path}"
    )

    benchmark = load_benchmark(
        dataset_path
    )

    print(
        f"Total test cases loaded: {len(benchmark)}\n"
    )

    # ========================================
    # TEST MODE
    # ========================================

    if test_mode:

        benchmark = select_test_cases(
            benchmark
        )

        print(
            "TEST MODE ENABLED"
        )

        print(
            f"Selected {len(benchmark)} "
            "test cases across multiple "
            "evaluation categories.\n"
        )

    else:

        print(
            "FULL EVALUATION MODE ENABLED"
        )

        print(
            f"Running all {len(benchmark)} "
            "test cases.\n"
        )

    # ========================================
    # INITIALIZE RESULTS
    # ========================================

    results = []

    completed_ids = set()

    # ========================================
    # CHECK PREVIOUSLY COMPLETED TESTS
    # ========================================

    if (
        dataset_path.name == "benchmark.csv"
        and not test_mode
    ):

        completed_ids = get_completed_ids()

        if completed_ids:

            print(
                f"Found {len(completed_ids)} "
                "previously completed test cases."
            )

            print(
                "They will be skipped.\n"
            )

    # ========================================
    # CREATE ACTUAL TEST LIST
    # ========================================

    tests_to_run = benchmark[
        ~benchmark["id"].isin(
            completed_ids
        )
    ]

    total_tests = len(
        tests_to_run
    )

    completed_count = 0

    # ========================================
    # INITIAL PROGRESS UPDATE
    # ========================================

    if progress_callback:

        progress_callback({
            "status": "running",
            "completed": 0,
            "total": total_tests,
            "percentage": 0,
            "current_test": None,
            "current_category": None
        })

    # ========================================
    # HANDLE EMPTY DATASET
    # ========================================

    if total_tests == 0:

        print(
            "No test cases available to evaluate."
        )

        if progress_callback:

            progress_callback({
                "status": "completed",
                "completed": 0,
                "total": 0,
                "percentage": 100,
                "current_test": None,
                "current_category": None
            })

        return results

    # ========================================
    # RUN EACH TEST CASE
    # ========================================

    for _, test_case in tests_to_run.iterrows():

        test_id = test_case[
            "id"
        ]

        category = test_case[
            "category"
        ]

        prompt = test_case[
            "prompt"
        ]

        # ========================================
        # UPDATE PROGRESS BEFORE TEST
        # ========================================

        if progress_callback:

            percentage = round(
                (
                    completed_count /
                    total_tests
                ) * 100,
                2
            )

            progress_callback({
                "status": "running",
                "completed": completed_count,
                "total": total_tests,
                "percentage": percentage,
                "current_test": test_id,
                "current_category": category
            })

        # ========================================
        # DISPLAY CURRENT TEST
        # ========================================

        print(
            f"Running test: {test_id} | "
            f"Category: {category}"
        )

        try:

            # ========================================
            # GENERATE MODEL RESPONSE
            # ========================================

            response = generate_response(
                user_prompt=prompt,
                endpoint=endpoint,
                api_key=api_key,
                model_name=model_name,
                system_instruction=(
                    CAREERSHIELD_SYSTEM_PROMPT
                )
            )

            evaluation_result = None

            # ========================================
            # HALLUCINATION
            # ========================================

            if category == "hallucination":

                evaluation_result = (
                    evaluate_hallucination(
                        response,
                        test_case[
                            "expected_answer"
                        ]
                    )
                )

            # ========================================
            # BIAS
            # ========================================

            elif category == "bias":

                evaluation_result = (
                    evaluate_bias(
                        response,
                        test_case[
                            "evaluation_criteria"
                        ]
                    )
                )

            # ========================================
            # TOXICITY
            # ========================================

            elif category == "toxicity":

                evaluation_result = (
                    evaluate_toxicity(
                        response,
                        test_case[
                            "evaluation_criteria"
                        ]
                    )
                )

            # ========================================
            # JAILBREAK
            # ========================================

            elif category == "jailbreak":

                evaluation_result = (
                    evaluate_jailbreak(
                        response,
                        test_case[
                            "evaluation_criteria"
                        ]
                    )
                )

            # ========================================
            # PROMPT INJECTION
            # ========================================

            elif category == "prompt_injection":

                evaluation_result = (
                    evaluate_prompt_injection(
                        response,
                        test_case[
                            "evaluation_criteria"
                        ]
                    )
                )

            # ========================================
            # REASONING
            # ========================================

            elif category == "reasoning":

                evaluation_result = (
                    evaluate_reasoning(
                        response,
                        test_case[
                            "expected_answer"
                        ]
                    )
                )

            # ========================================
            # UNKNOWN CATEGORY
            # ========================================

            else:

                evaluation_result = {
                    "score": None,
                    "status": "NOT_EVALUATED",
                    "reason": (
                        "No evaluator implemented "
                        f"for category: {category}"
                    )
                }

            # ========================================
            # CREATE RESULT
            # ========================================

            result = {
                "id": test_id,
                "category": category,
                "prompt": prompt,
                "expected_answer": (
                    test_case[
                        "expected_answer"
                    ]
                ),
                "evaluation_criteria": (
                    test_case[
                        "evaluation_criteria"
                    ]
                ),
                "difficulty": (
                    test_case[
                        "difficulty"
                    ]
                ),
                "response": response,
                "status": "completed",
                "evaluation": (
                    evaluation_result
                )
            }

            # ========================================
            # SAVE RESPONSE
            # ========================================

            save_response(
                result
            )

            results.append(
                result
            )

            print(
                "Status: Completed"
            )

            if evaluation_result:

                print(
                    "Evaluation Score: "
                    f"{evaluation_result.get('score')}"
                )

                print(
                    "Evaluation Status: "
                    f"{evaluation_result.get('status')}"
                )

        # ========================================
        # HANDLE TEST FAILURE
        # ========================================

        except Exception as e:

            print(
                f"Status: Failed - {e}"
            )

            results.append({
                "id": test_id,
                "category": category,
                "prompt": prompt,
                "response": None,
                "status": (
                    f"failed: {e}"
                )
            })

        # ========================================
        # INCREMENT COMPLETED COUNT
        # ========================================

        completed_count += 1

        # ========================================
        # UPDATE PROGRESS AFTER TEST
        # ========================================

        if progress_callback:

            percentage = round(
                (
                    completed_count /
                    total_tests
                ) * 100,
                2
            )

            progress_callback({
                "status": "running",
                "completed": completed_count,
                "total": total_tests,
                "percentage": percentage,
                "current_test": test_id,
                "current_category": category
            })

        # ========================================
        # WAIT BEFORE NEXT REQUEST
        # ========================================

        if completed_count < total_tests:

            print(
                "\nWaiting 12 seconds before "
                "next request..."
            )

            time.sleep(
                12
            )

    # ========================================
    # FINAL PROGRESS UPDATE
    # ========================================

    if progress_callback:

        progress_callback({
            "status": "completed",
            "completed": completed_count,
            "total": total_tests,
            "percentage": 100,
            "current_test": None,
            "current_category": None
        })

    print(
        "\n===== Evaluation Completed =====\n"
    )

    return results