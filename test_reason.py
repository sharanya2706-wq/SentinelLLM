from evaluation.reasoning import evaluate_reasoning


print("\n===== REASONING EVALUATOR TEST =====\n")


test_cases = [

    # ----------------------------------------
    # CORRECT LOGICAL REASONING
    # ----------------------------------------

    {
        "name": "Correct Direct Answer",
        "response": "Yes",
        "expected_answer": "Yes"
    },

    {
        "name": "Correct Logical Explanation",
        "response": (
            "Yes. Since all roses are flowers and all "
            "flowers are plants, all roses are plants."
        ),
        "expected_answer": "Yes"
    },

    {
        "name": "Correct Reasoning Different Wording",
        "response": (
            "All roses must be plants because every rose "
            "belongs to the group of flowers, and every "
            "flower belongs to the group of plants."
        ),
        "expected_answer": "Yes"
    },

    {
        "name": "Correct Transitive Reasoning",
        "response": (
            "The conclusion follows logically through "
            "the transitive relationship between the groups."
        ),
        "expected_answer": "Yes"
    },

    # ----------------------------------------
    # INCORRECT LOGICAL REASONING
    # ----------------------------------------

    {
        "name": "Incorrect Direct Answer",
        "response": "No",
        "expected_answer": "Yes"
    },

    {
        "name": "Incorrect Logical Explanation",
        "response": (
            "No. We cannot conclude that roses are plants "
            "from the given statements."
        ),
        "expected_answer": "Yes"
    },

    {
        "name": "Insufficient Information Claim",
        "response": (
            "There is not enough information to determine "
            "whether roses are plants."
        ),
        "expected_answer": "Yes"
    },

    # ----------------------------------------
    # UNCLEAR RESPONSE
    # ----------------------------------------

    {
        "name": "Unclear Response",
        "response": (
            "This is an interesting logical question."
        ),
        "expected_answer": "Yes"
    },

    # ----------------------------------------
    # EMPTY RESPONSE
    # ----------------------------------------

    {
        "name": "Empty Response",
        "response": "",
        "expected_answer": "Yes"
    }
]


for test in test_cases:

    result = evaluate_reasoning(
        test["response"],
        test["expected_answer"]
    )

    print(f"Test: {test['name']}")
    print(f"Score: {result['score']}")
    print(f"Status: {result['status']}")

    print(
        f"Reasoning Correct: "
        f"{result['reasoning_correct']}"
    )

    print(
        f"Detected Indicator: "
        f"{result['detected_indicator']}"
    )

    print(
        f"Correct Similarity: "
        f"{result['correct_similarity']}"
    )

    print(
        f"Incorrect Similarity: "
        f"{result['incorrect_similarity']}"
    )

    print(f"Reason: {result['reason']}")

    print("-" * 50)


print("\n===== TEST COMPLETED =====")