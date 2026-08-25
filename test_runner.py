from evaluation.runner import run_evaluation


dataset_path = "data/benchmark.csv"

results = run_evaluation(dataset_path)


print("\n===== EVALUATION SUMMARY =====\n")

for result in results:
    print(f"Test ID: {result['id']}")
    print(f"Category: {result['category']}")
    print(f"Status: {result['status']}")
    print(f"Response: {result.get('response')}")

    if result.get("evaluation"):
        print("\nEvaluation:")
        print(f"Score: {result['evaluation']['score']}")
        print(f"Result: {result['evaluation']['status']}")
        print(f"Reason: {result['evaluation']['reason']}")

    print("-" * 50)
    