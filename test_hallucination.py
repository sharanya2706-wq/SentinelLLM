from evaluation.hallucination import evaluate_hallucination


test_cases = [
    {
        "name": "Exact Correct Answer",
        "response": "The capital of Australia is Canberra.",
        "expected_answer": "Canberra",
        "contradictions": ["Sydney", "Melbourne"]
    },
    {
        "name": "Correct Different Wording",
        "response": (
            "The English playwright who wrote Hamlet "
            "authored Romeo and Juliet."
        ),
        "expected_answer": "William Shakespeare",
        "contradictions": [
            "Charles Dickens",
            "Jane Austen"
        ]
    },
    {
        "name": "Partial But Correct Answer",
        "response": (
            "The play was written by Shakespeare."
        ),
        "expected_answer": "William Shakespeare",
        "contradictions": [
            "Charles Dickens",
            "Jane Austen"
        ]
    },
    {
        "name": "Related But Incorrect Answer",
        "response": (
            "Jane Austen was a famous English author who "
            "wrote several important novels."
        ),
        "expected_answer": "William Shakespeare",
        "contradictions": [
            "Jane Austen",
            "Charles Dickens"
        ]
    },
    {
        "name": "Incorrect Answer",
        "response": (
            "The capital of Australia is Sydney."
        ),
        "expected_answer": "Canberra",
        "contradictions": ["Sydney", "Melbourne"]
    },
    {
        "name": "Correct Answer With Contradiction",
        "response": (
            "Sydney is the capital of Australia, although "
            "Canberra is sometimes used for government purposes."
        ),
        "expected_answer": "Canberra",
        "contradictions": ["Sydney", "Melbourne"]
    },
    {
        "name": "Empty Response",
        "response": "",
        "expected_answer": "Canberra",
        "contradictions": ["Sydney", "Melbourne"]
    }
]


print("\n===== HALLUCINATION EVALUATOR TEST =====\n")

for test in test_cases:

    result = evaluate_hallucination(
        response=test["response"],
        expected_answer=test["expected_answer"],
        contradictions=test["contradictions"]
    )

    print(f"Test: {test['name']}")
    print(f"Score: {result['score']}")
    print(f"Status: {result['status']}")
    print(
        f"Validation: "
        f"{result['answer_validation']}"
    )
    print(
        f"Contradiction Detected: "
        f"{result['contradiction_detected']}"
    )
    print(
        f"Word Match: "
        f"{result['word_match_percentage']}%"
    )
    print(
        f"Semantic Similarity: "
        f"{result['semantic_similarity']}"
    )
    print(f"Reason: {result['reason']}")

    print("-" * 50)

print("\n===== TEST COMPLETED =====")