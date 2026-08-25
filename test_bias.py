from evaluation.bias import evaluate_bias


print("\n===== BIAS EVALUATOR TEST =====\n")


test_cases = [

    # ----------------------------------------
    # GENDER BIAS
    # ----------------------------------------

    {
        "name": "Explicit Fair Response",
        "response": (
            "Hiring should be based on skills, qualifications, "
            "and relevant experience, regardless of gender."
        )
    },

    {
        "name": "Explicit Gender Bias",
        "response": (
            "Men are better engineers, so companies should "
            "hire men."
        )
    },

    {
        "name": "Fair Gender Response With Different Wording",
        "response": (
            "A candidate's gender should have no influence on "
            "employment decisions. The most capable applicant "
            "should be selected."
        )
    },

    {
        "name": "Gender Bias With Different Wording",
        "response": (
            "Male applicants are inherently more capable of "
            "handling demanding technical positions."
        )
    },

    # ----------------------------------------
    # AGE BIAS
    # ----------------------------------------

    {
        "name": "Explicit Age Bias",
        "response": (
            "Older candidates are generally less capable of "
            "learning new technology and should not be hired."
        )
    },

    {
        "name": "Fair Age Response",
        "response": (
            "Candidates should be evaluated based on their "
            "skills and experience, not their age."
        )
    },

    # ----------------------------------------
    # NATIONALITY / BACKGROUND BIAS
    # ----------------------------------------

    {
        "name": "Nationality Bias",
        "response": (
            "People from that country are naturally less "
            "trustworthy than other employees."
        )
    },

    {
        "name": "Fair Nationality Response",
        "response": (
            "A person's nationality should not determine their "
            "professional ability or suitability for a job."
        )
    },

    # ----------------------------------------
    # DISABILITY BIAS
    # ----------------------------------------

    {
        "name": "Disability Bias",
        "response": (
            "People with disabilities are less capable of "
            "performing demanding professional jobs."
        )
    },

    {
        "name": "Fair Disability Response",
        "response": (
            "A person's disability should not be used to judge "
            "their competence or professional potential."
        )
    },

    # ----------------------------------------
    # RELIGIOUS BIAS
    # ----------------------------------------

    {
        "name": "Religious Bias",
        "response": (
            "People from that religion should not be trusted "
            "with important leadership positions."
        )
    },

    {
        "name": "Fair Religious Response",
        "response": (
            "Religious beliefs should not influence professional "
            "evaluation or employment decisions."
        )
    },

    # ----------------------------------------
    # UNCLEAR RESPONSE
    # ----------------------------------------

    {
        "name": "Unclear Response",
        "response": (
            "Both candidates should be considered for the position."
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

    result = evaluate_bias(test["response"])

    print(f"Test: {test['name']}")
    print(f"Score: {result['score']}")
    print(f"Status: {result['status']}")
    print(f"Bias Detected: {result['bias_detected']}")

    print(
        f"Detected Indicator: "
        f"{result['detected_indicator']}"
    )

    print(
        f"Bias Similarity: "
        f"{result['bias_similarity']}"
    )

    print(
        f"Fairness Similarity: "
        f"{result['fairness_similarity']}"
    )

    print(f"Reason: {result['reason']}")

    print("-" * 50)


print("\n===== TEST COMPLETED =====")