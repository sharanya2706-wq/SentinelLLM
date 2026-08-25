from llm.client import generate_response


prompt = """
I have a software engineering interview in 15 days.
How should I prepare?
"""

response = generate_response(prompt)

print("\n===== CareerShield AI Response =====\n")
print(response)
