import json
from pathlib import Path
from datetime import datetime

from evaluation.recommendations import (
    generate_recommendations
)


HISTORY_FILE = Path(
    "data/evaluation_history.json"
)


def load_history():
    """
    Load all previous evaluation runs.
    """

    if not HISTORY_FILE.exists():
        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        FileNotFoundError
    ):

        return []


def save_evaluation_history(
    model_name,
    dataset_type,
    results
):
    """
    Save a completed evaluation run to history.
    """

    history = load_history()

    # ----------------------------------------
    # TOTAL TESTS
    # ----------------------------------------

    total_tests = len(results)


    # ----------------------------------------
    # COMPLETED RESULTS
    # ----------------------------------------

    completed_results = [

        result

        for result in results

        if result.get("status") == "completed"

    ]


    # ----------------------------------------
    # PASSED TESTS
    # ----------------------------------------

    passed_tests = sum(

        1

        for result in completed_results

        if (
            result
            .get("evaluation", {})
            .get("status")
            == "PASS"
        )

    )


    # ----------------------------------------
    # NEEDS ATTENTION
    # ----------------------------------------

    failed_tests = (
        total_tests - passed_tests
    )


    # ----------------------------------------
    # COLLECT SCORES
    # ----------------------------------------

    scores = []

    for result in completed_results:

        evaluation = (
            result.get("evaluation", {})
        )

        score = evaluation.get("score")

        if score is not None:

            scores.append(score)


    # ----------------------------------------
    # OVERALL SCORE
    # ----------------------------------------

    overall_score = (

        round(
            sum(scores) / len(scores),
            2
        )

        if scores

        else 0

    )


    # ----------------------------------------
    # CATEGORY SCORES
    # ----------------------------------------

    category_scores = {}


    for result in completed_results:

        category = (
            result.get("category")
        )

        score = (

            result
            .get("evaluation", {})
            .get("score")

        )


        if (
            category
            and score is not None
        ):

            if category not in category_scores:

                category_scores[
                    category
                ] = []


            category_scores[
                category
            ].append(score)


    # ----------------------------------------
    # CATEGORY SCORE SUMMARY
    # ----------------------------------------

    category_summary = {}


    for (
        category,
        scores_list
    ) in category_scores.items():

        category_summary[
            category
        ] = round(

            sum(scores_list)
            /
            len(scores_list),

            2

        )


    # ----------------------------------------
    # GENERATE AI RECOMMENDATIONS
    # ----------------------------------------

    recommendations = (
        generate_recommendations(
            category_summary
        )
    )


    # ----------------------------------------
    # CREATE EVALUATION HISTORY ENTRY
    # ----------------------------------------

    evaluation_run = {

        "evaluation_id":
            f"eval_{len(history) + 1}",


        "timestamp":

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),


        "model_name":
            model_name,


        "dataset_type":
            dataset_type,


        "total_tests":
            total_tests,


        "passed_tests":
            passed_tests,


        "needs_attention":
            failed_tests,


        "overall_score":
            overall_score,


        "category_scores":
            category_summary,


        # AI recommendations
        "recommendations":
            recommendations

    }


    # ----------------------------------------
    # ADD NEW EVALUATION TO TOP
    # ----------------------------------------

    history.insert(
        0,
        evaluation_run
    )


    # ----------------------------------------
    # ENSURE DATA DIRECTORY EXISTS
    # ----------------------------------------

    HISTORY_FILE.parent.mkdir(

        parents=True,
        exist_ok=True

    )


    # ----------------------------------------
    # SAVE HISTORY
    # ----------------------------------------

    with open(

        HISTORY_FILE,
        "w",
        encoding="utf-8"

    ) as file:

        json.dump(

            history,

            file,

            indent=4

        )


    return evaluation_run