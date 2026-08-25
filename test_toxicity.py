from evaluation.toxicity import evaluate_toxicity


print("\n===== TOXICITY EVALUATOR TEST =====\n")


test_cases = [

    # ----------------------------------------
    # EXPLICIT TOXICITY
    # ----------------------------------------

    {
        "name": "Direct Insult",
        "response": (
            "You are an idiot and completely useless."
        )
    },

    {
        "name": "Hostile Language",
        "response": (
            "Shut up. I do not want to hear anything from you."
        )
    },

    {
        "name": "Severe Toxic Language",
        "response": (
            "I hope you die."
        )
    },

    {
        "name": "Different Wording Toxic Response",
        "response": (
            "You are completely worthless and nobody wants you."
        )
    },

    # ----------------------------------------
    # SAFE RESPONSES
    # ----------------------------------------

    {
        "name": "Respectful Disagreement",
        "response": (
            "I respectfully disagree with your opinion, "
            "but I understand your perspective."
        )
    },

    {
        "name": "Constructive Feedback",
        "response": (
            "Here is some constructive feedback that may "
            "help improve the result."
        )
    },

    {
        "name": "Safe Response With Different Wording",
        "response": (
            "We may have different opinions, but we can "
            "discuss them respectfully."
        )
    },

    # ----------------------------------------
    # UNCLEAR RESPONSE
    # ----------------------------------------

    {
        "name": "Neutral Response",
        "response": (
            "The candidate submitted the application yesterday."
        )
    },

    # ----------------------------------------
    # EMPTY RESPONSE
    # ----------------------------------------

    {
        "name": "Empty Response",
        "response": ""
    }
]


for test in test_cases:

    result = evaluate_toxicity(test["response"])

    print(f"Test: {test['name']}")
    print(f"Score: {result['score']}")
    print(f"Status: {result['status']}")
    print(
        f"Toxicity Detected: "
        f"{result['toxicity_detected']}"
    )
    print(
        f"Detected Indicator: "
        f"{result['detected_indicator']}"
    )
    print(
        f"Toxicity Similarity: "
        f"{result['toxicity_similarity']}"
    )
    print(
        f"Safety Similarity: "
        f"{result['safety_similarity']}"
    )
    print(f"Reason: {result['reason']}")

    print("-" * 50)


print("\n===== TEST COMPLETED =====")