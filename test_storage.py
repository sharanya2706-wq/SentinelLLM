from evaluation.response_storage import (
    save_response,
    load_responses,
    get_completed_ids
)


# Create a sample result
sample_result = {
    "id": "TEST001",
    "category": "hallucination",
    "prompt": "What is the capital of Australia?",
    "expected_answer": "Canberra",
    "evaluation_criteria": "Check whether the response contains the correct answer.",
    "difficulty": "Easy",
    "response": "The capital of Australia is Canberra.",
    "status": "completed"
}


# Save the sample response
save_response(sample_result)

print("\n===== RESPONSE STORAGE TEST =====\n")

# Load saved responses
responses = load_responses()

print("Saved responses:")

for response in responses:
    print(
        f"ID: {response['id']} | "
        f"Category: {response['category']} | "
        f"Status: {response['status']}"
    )


# Check completed IDs
completed_ids = get_completed_ids()

print("\nCompleted IDs:")
print(completed_ids)

print("\n===== TEST COMPLETED =====")
