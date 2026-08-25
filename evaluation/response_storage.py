import csv
import os


RESULTS_FILE = "data/responses.csv"


def save_response(result):
    """
    Save one completed LLM response and its evaluation
    permanently to responses.csv.
    """

    os.makedirs("data", exist_ok=True)

    file_exists = os.path.exists(RESULTS_FILE)

    evaluation = result.get("evaluation") or {}

    fieldnames = [

        # ----------------------------------------
        # BASIC TEST INFORMATION
        # ----------------------------------------

        "id",
        "category",
        "prompt",
        "expected_answer",
        "evaluation_criteria",
        "difficulty",
        "response",
        "status",

        # ----------------------------------------
        # COMMON EVALUATION FIELDS
        # ----------------------------------------

        "score",
        "evaluation_status",
        "reason",

        # ----------------------------------------
        # HALLUCINATION FIELDS
        # ----------------------------------------

        "answer_validation",
        "contradiction_detected",
        "semantic_similarity",
        "word_match_percentage",

        # ----------------------------------------
        # BIAS FIELDS
        # ----------------------------------------

        "bias_detected",
        "detected_indicator",
        "bias_similarity",
        "fairness_similarity",

        # ----------------------------------------
        # TOXICITY FIELDS
        # ----------------------------------------

        "toxicity_detected",
        "toxicity_similarity",
        "safety_similarity",

        # ----------------------------------------
        # JAILBREAK FIELDS
        # ----------------------------------------

        "jailbreak_detected",
        "jailbreak_compliance_similarity",
        "jailbreak_resistance_similarity",

        # ----------------------------------------
        # PROMPT INJECTION FIELDS
        # ----------------------------------------

        "prompt_injection_detected",
        "prompt_injection_compliance_similarity",
        "prompt_injection_resistance_similarity",

        # ----------------------------------------
        # REASONING FIELDS
        # ----------------------------------------

        "reasoning_correct",
        "reasoning_correct_similarity",
        "reasoning_incorrect_similarity"
    ]

    with open(
        RESULTS_FILE,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        # Write header only for a new file
        if not file_exists:

            writer.writeheader()

        writer.writerow({

            # ----------------------------------------
            # BASIC TEST INFORMATION
            # ----------------------------------------

            "id":
                result.get("id"),

            "category":
                result.get("category"),

            "prompt":
                result.get("prompt"),

            "expected_answer":
                result.get("expected_answer"),

            "evaluation_criteria":
                result.get("evaluation_criteria"),

            "difficulty":
                result.get("difficulty"),

            "response":
                result.get("response"),

            "status":
                result.get("status"),

            # ----------------------------------------
            # COMMON EVALUATION FIELDS
            # ----------------------------------------

            "score":
                evaluation.get("score"),

            "evaluation_status":
                evaluation.get("status"),

            "reason":
                evaluation.get("reason"),

            # ----------------------------------------
            # HALLUCINATION FIELDS
            # ----------------------------------------

            "answer_validation":
                evaluation.get("validation"),

            "contradiction_detected":
                evaluation.get(
                    "contradiction_detected"
                ),

            "semantic_similarity":
                evaluation.get(
                    "semantic_similarity"
                ),

            "word_match_percentage":
                evaluation.get(
                    "word_match_percentage"
                ),

            # ----------------------------------------
            # BIAS FIELDS
            # ----------------------------------------

            "bias_detected":
                evaluation.get(
                    "bias_detected"
                ),

            "detected_indicator":
                evaluation.get(
                    "detected_indicator"
                ),

            "bias_similarity":
                evaluation.get(
                    "bias_similarity"
                ),

            "fairness_similarity":
                evaluation.get(
                    "fairness_similarity"
                ),

            # ----------------------------------------
            # TOXICITY FIELDS
            # ----------------------------------------

            "toxicity_detected":
                evaluation.get(
                    "toxicity_detected"
                ),

            "toxicity_similarity":
                evaluation.get(
                    "toxicity_similarity"
                ),

            "safety_similarity":
                evaluation.get(
                    "safety_similarity"
                ),

            # ----------------------------------------
            # JAILBREAK FIELDS
            # ----------------------------------------

            "jailbreak_detected":
                evaluation.get(
                    "jailbreak_detected"
                ),

            "jailbreak_compliance_similarity":
                evaluation.get(
                    "compliance_similarity"
                )
                if result.get("category") == "jailbreak"
                else None,

            "jailbreak_resistance_similarity":
                evaluation.get(
                    "resistance_similarity"
                )
                if result.get("category") == "jailbreak"
                else None,

            # ----------------------------------------
            # PROMPT INJECTION FIELDS
            # ----------------------------------------

            "prompt_injection_detected":
                evaluation.get(
                    "prompt_injection_detected"
                ),

            "prompt_injection_compliance_similarity":
                evaluation.get(
                    "compliance_similarity"
                )
                if result.get("category") == "prompt_injection"
                else None,

            "prompt_injection_resistance_similarity":
                evaluation.get(
                    "resistance_similarity"
                )
                if result.get("category") == "prompt_injection"
                else None,

            # ----------------------------------------
            # REASONING FIELDS
            # ----------------------------------------

            "reasoning_correct":
                evaluation.get(
                    "reasoning_correct"
                ),

            "reasoning_correct_similarity":
                evaluation.get(
                    "correct_similarity"
                ),

            "reasoning_incorrect_similarity":
                evaluation.get(
                    "incorrect_similarity"
                )
        })


def load_responses():
    """
    Load all previously saved responses.
    """

    if not os.path.exists(RESULTS_FILE):

        return []

    with open(
        RESULTS_FILE,
        mode="r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        return list(reader)


def get_completed_ids():
    """
    Return IDs of all successfully completed test cases.
    """

    responses = load_responses()

    completed_ids = set()

    for response in responses:

        if response.get("status") == "completed":

            completed_ids.add(
                response.get("id")
            )

    return completed_ids
    