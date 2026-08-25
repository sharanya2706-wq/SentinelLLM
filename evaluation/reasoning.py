import re
from sentence_transformers import SentenceTransformer, util


# Load the semantic model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize_text(text):
    """
    Normalize text for consistent evaluation.
    """

    if not text:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def calculate_semantic_similarity(response, examples):
    """
    Compare the response with example statements and
    return the highest semantic similarity score.
    """

    response_embedding = model.encode(
        response,
        convert_to_tensor=True
    )

    example_embeddings = model.encode(
        examples,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        response_embedding,
        example_embeddings
    )[0]

    return float(similarities.max())


def detect_direct_answer(response):
    """
    Detect a direct yes/no answer at the beginning
    of the response.
    """

    response = normalize_text(response)

    if re.match(r"^yes\b", response):
        return "yes"

    if re.match(r"^no\b", response):
        return "no"

    return None


def evaluate_reasoning(
    response,
    expected_answer="",
    evaluation_criteria=""
):
    """
    Evaluate whether an LLM response reaches the correct
    logical conclusion.

    Evaluation order:

    1. Empty response
    2. Explicit correct answer
    3. Explicit contradictory answer
    4. Semantic reasoning comparison
    5. Uncertain response
    """

    # ----------------------------------------
    # EMPTY RESPONSE
    # ----------------------------------------

    if not response or not str(response).strip():

        return {
            "score": 0,
            "status": "FAIL",
            "reasoning_correct": False,
            "detected_indicator": "",
            "correct_similarity": 0.0,
            "incorrect_similarity": 0.0,
            "reason": "The model did not provide a response."
        }

    normalized_response = normalize_text(response)
    normalized_expected = normalize_text(expected_answer)

    # ----------------------------------------
    # CORRECT REASONING EXAMPLES
    # ----------------------------------------

    correct_examples = [

        "Yes.",

        "Yes, the conclusion follows logically.",

        "Since all roses are flowers and all flowers are plants, all roses are plants.",

        "All roses must be plants because every rose is a flower and every flower is a plant.",

        "The conclusion follows through a transitive relationship.",

        "The logical reasoning is valid and the conclusion is correct.",

        "The answer follows logically from the given statements."
    ]

    if normalized_expected:
        correct_examples.append(normalized_expected)

    # ----------------------------------------
    # INCORRECT REASONING EXAMPLES
    # ----------------------------------------

    incorrect_examples = [

        "No.",

        "No, the conclusion does not follow.",

        "We cannot conclude that all roses are plants.",

        "There is not enough information to determine the answer.",

        "The reasoning is incorrect.",

        "The logical conclusion is invalid."
    ]

    # ----------------------------------------
    # CALCULATE SIMILARITIES
    # ----------------------------------------

    correct_similarity = calculate_semantic_similarity(
        normalized_response,
        correct_examples
    )

    incorrect_similarity = calculate_semantic_similarity(
        normalized_response,
        incorrect_examples
    )

    correct_similarity = round(
        correct_similarity,
        3
    )

    incorrect_similarity = round(
        incorrect_similarity,
        3
    )

    # ----------------------------------------
    # DIRECT ANSWER VALIDATION
    # ----------------------------------------

    direct_answer = detect_direct_answer(
        normalized_response
    )

    # Expected answer is yes/no
    if normalized_expected in ["yes", "no"]:

        # Correct direct answer
        if direct_answer == normalized_expected:

            return {
                "score": 100,
                "status": "PASS",
                "reasoning_correct": True,
                "detected_indicator":
                    "Correct logical conclusion detected",
                "correct_similarity": correct_similarity,
                "incorrect_similarity": incorrect_similarity,
                "reason": (
                    "The response reaches the expected "
                    "logical conclusion."
                )
            }

        # Contradictory direct answer
        if (
            direct_answer
            and direct_answer != normalized_expected
        ):

            return {
                "score": 0,
                "status": "FAIL",
                "reasoning_correct": False,
                "detected_indicator":
                    "Contradictory logical conclusion detected",
                "correct_similarity": correct_similarity,
                "incorrect_similarity": incorrect_similarity,
                "reason": (
                    "The response gives a logical conclusion "
                    "that contradicts the expected answer."
                )
            }

    # ----------------------------------------
    # EXPLICIT EXPECTED ANSWER
    # ----------------------------------------

    if (
        normalized_expected
        and normalized_expected in normalized_response
    ):

        return {
            "score": 100,
            "status": "PASS",
            "reasoning_correct": True,
            "detected_indicator":
                "Expected answer detected",
            "correct_similarity": correct_similarity,
            "incorrect_similarity": incorrect_similarity,
            "reason": (
                "The response contains the expected answer "
                "and is consistent with the correct conclusion."
            )
        }

    # ----------------------------------------
    # STRONG CORRECT REASONING
    # ----------------------------------------

    if (
        correct_similarity >= 0.60
        and (
            correct_similarity
            - incorrect_similarity
        ) >= 0.05
    ):

        return {
            "score": 100,
            "status": "PASS",
            "reasoning_correct": True,
            "detected_indicator":
                "Strong semantic evidence of correct reasoning",
            "correct_similarity": correct_similarity,
            "incorrect_similarity": incorrect_similarity,
            "reason": (
                "The response is more consistent with "
                "correct logical reasoning."
            )
        }

    # ----------------------------------------
    # STRONG INCORRECT REASONING
    # ----------------------------------------

    if (
        incorrect_similarity >= 0.60
        and (
            incorrect_similarity
            - correct_similarity
        ) >= 0.05
    ):

        return {
            "score": 0,
            "status": "FAIL",
            "reasoning_correct": False,
            "detected_indicator":
                "Strong semantic evidence of incorrect reasoning",
            "correct_similarity": correct_similarity,
            "incorrect_similarity": incorrect_similarity,
            "reason": (
                "The response is more consistent with "
                "incorrect logical reasoning."
            )
        }

    # ----------------------------------------
    # UNCERTAIN RESPONSE
    # ----------------------------------------

    return {
        "score": 50,
        "status": "WARNING",
        "reasoning_correct": False,
        "detected_indicator": "",
        "correct_similarity": correct_similarity,
        "incorrect_similarity": incorrect_similarity,
        "reason": (
            "The response does not contain enough clear evidence "
            "to confidently determine whether the reasoning "
            "is correct or incorrect."
        )
    }