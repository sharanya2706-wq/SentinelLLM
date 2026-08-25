from openai import OpenAI


def generate_response(
    user_prompt,
    endpoint,
    api_key,
    model_name,
    system_instruction=None
):
    """
    Send a prompt to a user-provided OpenAI-compatible
    LLM endpoint and return the response.
    """

    # Create OpenAI-compatible client
    client = OpenAI(
        api_key=api_key,
        base_url=endpoint
    )

    messages = []

    # Add system instruction if provided
    if system_instruction:
        messages.append(
            {
                "role": "system",
                "content": system_instruction
            }
        )

    # Add benchmark prompt
    messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    # Call the selected LLM
    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    # Return generated response
    return response.choices[0].message.content