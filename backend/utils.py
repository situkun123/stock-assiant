import os

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI


def create_state_graph(app, save_path="financial_agent_graph.png"):
    """Visualize the agent graph and save to file."""
    try:

        png_data = app.get_graph().draw_mermaid_png()
        # Save to file
        with open(save_path, "wb") as f:
            f.write(png_data)
        print(f"✓ Graph saved to {save_path}")


    except ImportError:
        # Fallback to ASCII if dependencies missing
        print("Install graphviz for PNG output: pip install pygraphviz")
        print("\nASCII representation:")
        print(app.get_graph().draw_ascii())

def open_ai_key_test(model="gpt-4o-mini"):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env or export it in your shell.")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say this is a test"}
            ]
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_number_of_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

if __name__ == "__main__":
    text = "This is a test message to calculate the number of tokens."
    num_tokens = calculate_number_of_tokens(text)
    print(f"Number of tokens in the text: {num_tokens}")
