import re
from sentence_transformers import SentenceTransformer, util


# Load the semantic model only once when the program starts
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize_text(text):
    """
    Normalize text for comparison.
    """

    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text


def get_word_matches(response, expected_answer):
    """
    Calculate how many meaningful words from the expected
    answer appear in the response.
    """

    normalized_response = normalize_text(response)
    normalized_expected = normalize_text(expected_answer)

    expected_words = normalized_expected.split()

    if not expected_words:
        return 0, 0.0

    matched_words = 0

    for word in expected_words:
        if word in normalized_response:
            matched_words += 1

    match_percentage = (
        matched_words / len(expected_words)
    ) * 100

    return matched_words, match_percentage


def check_contradiction(response, contradictions):
    """
    Check whether the response contains known contradictory
    information.
    """

    if not contradictions:
        return False, None

    normalized_response = normalize_text(response)

    for contradiction in contradictions:

        normalized_contradiction = normalize_text(
            contradiction
        )

        if (
            normalized_contradiction
            and normalized_contradiction in normalized_response
        ):
            return True, contradiction

    return False, None


def calculate_semantic_similarity(response, expected_answer):
    """
    Calculate semantic similarity between the LLM response
    and the expected answer.
    """

    embeddings = semantic_model.encode(
        [expected_answer, response],
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        embeddings[0],
        embeddings[1]
    )

    return float(similarity.item())


def evaluate_hallucination(
    response,
    expected_answer,
    contradictions=None
):
    """
    SentinelLLM Hallucination Evaluator

    Evaluation layers:
    1. Empty response detection
    2. Contradiction detection
    3. Exact answer validation
    4. Partial/entity matching
    5. Semantic similarity analysis
    6. Combined scoring
    """

    # -----------------------------------------
    # 1. Empty Response
    # -----------------------------------------

    if not response or str(response).strip() == "":
        return {
            "score": 0,
            "status": "FAIL",
            "reason": "The model did not provide a response.",
            "answer_validation": "incorrect",
            "contradiction_detected": False,
            "semantic_similarity": 0.0,
            "word_match_percentage": 0.0
        }

    # -----------------------------------------
    # 2. Contradiction Detection
    # -----------------------------------------

    contradiction_found, matched_contradiction = (
        check_contradiction(
            response,
            contradictions
        )
    )

    # Explicit contradiction overrides everything
    if contradiction_found:
        return {
            "score": 0,
            "status": "FAIL",
            "reason": (
                "The response contains contradictory "
                f"information: '{matched_contradiction}'."
            ),
            "answer_validation": "incorrect",
            "contradiction_detected": True,
            "semantic_similarity": 0.0,
            "word_match_percentage": 0.0
        }

    # -----------------------------------------
    # Normalize
    # -----------------------------------------

    normalized_response = normalize_text(response)
    normalized_expected = normalize_text(expected_answer)

    # -----------------------------------------
    # 3. Exact Answer Match
    # -----------------------------------------

    if normalized_expected in normalized_response:
        return {
            "score": 100,
            "status": "PASS",
            "reason": "The response contains the expected answer.",
            "answer_validation": "exact_match",
            "contradiction_detected": False,
            "semantic_similarity": 1.0,
            "word_match_percentage": 100.0
        }

    # -----------------------------------------
    # 4. Word / Entity Matching
    # -----------------------------------------

    matched_words, word_match_percentage = (
        get_word_matches(
            response,
            expected_answer
        )
    )

    # -----------------------------------------
    # 5. Semantic Similarity
    # -----------------------------------------

    semantic_similarity = calculate_semantic_similarity(
        response,
        expected_answer
    )

    semantic_similarity = round(
        semantic_similarity,
        3
    )

    word_match_percentage = round(
        word_match_percentage,
        1
    )

    # -----------------------------------------
    # 6. Combined Scoring Logic
    # -----------------------------------------

    # Strong textual evidence + semantic support
    if (
        word_match_percentage >= 50
        and semantic_similarity >= 0.55
    ):
        return {
            "score": 100,
            "status": "PASS",
            "reason": (
                "The response has a meaningful match with "
                "the expected answer and strong semantic support."
            ),
            "answer_validation": "combined_match",
            "contradiction_detected": False,
            "semantic_similarity": semantic_similarity,
            "word_match_percentage": word_match_percentage
        }

    # Strong semantic relationship, but insufficient
    # direct evidence to confidently confirm correctness
    if semantic_similarity >= 0.55:
        return {
            "score": 50,
            "status": "WARNING",
            "reason": (
                "The response is semantically related to the "
                "expected answer, but correctness cannot be "
                "confirmed confidently."
            ),
            "answer_validation": "semantic_warning",
            "contradiction_detected": False,
            "semantic_similarity": semantic_similarity,
            "word_match_percentage": word_match_percentage
        }

    # Some textual overlap but weak semantic support
    if word_match_percentage >= 50:
        return {
            "score": 50,
            "status": "WARNING",
            "reason": (
                "The response partially matches the expected "
                "answer, but semantic evidence is weak."
            ),
            "answer_validation": "partial_match",
            "contradiction_detected": False,
            "semantic_similarity": semantic_similarity,
            "word_match_percentage": word_match_percentage
        }

    # No sufficient evidence of correctness
    return {
        "score": 0,
        "status": "FAIL",
        "reason": (
            "The response does not sufficiently match the "
            "expected answer."
        ),
        "answer_validation": "incorrect",
        "contradiction_detected": False,
        "semantic_similarity": semantic_similarity,
        "word_match_percentage": word_match_percentage
    }