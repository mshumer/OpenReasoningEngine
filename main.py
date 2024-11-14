# main.py
from dotenv import load_dotenv
load_dotenv()

from engine import complete_reasoning_task
from mixture import ensemble
import chain_store

def main():
    # Initialize store
    chain_store.init_store()

    # API keys and configurations
    cohere_api_key = "[REDACTED]"

    # Define task
    task = """
    Create a Python implementation of a Red-Black Tree with the following operations:
    1. Insert a node
    2. Delete a node
    3. Search for a node
    4. Print the tree in-order

    The implementation should maintain all Red-Black Tree properties:
    - Every node is either red or black
    - The root is black
    - All leaves (NIL) are black
    - If a node is red, then both its children are black
    - Every path from root to leaves contains the same number of black nodes

    Test the implementation by:
    1. Inserting the numbers [7, 3, 18, 10, 22, 8, 11, 26, 2, 6, 13]
    2. Printing the tree structure showing colors
    3. Deleting nodes 18 and 11
    4. Printing the final tree structure
    5. Searching for both present and non-present values

    Use the Python interpreter tool to implement and test this data structure.
    """

    # Simple mode: just run one model
    response, history, tools = complete_reasoning_task(
        task=task,
        model_config={
            "model": "openai/gpt-4o-mini",
            "api_url": "https://openrouter.ai/api/v1/chat/completions",
            "api_key": "[REDACTED]"
        },
        verbose=True
    )

    # Ensemble mode: run multiple models in parallel
    # ensemble_response = ensemble(
    #     task=task,
    #     agents=[
    #         {
    #             "model": "openai/gpt-4o",
    #             "api_key": "[REDACTED]",
    #             "api_url": "https://openrouter.ai/api/v1/chat/completions"
    #         },
    #         {
    #             "model": "anthropic/claude-3.5-sonnet",
    #             "api_key": "[REDACTED]",
    #             "api_url": "https://openrouter.ai/api/v1/chat/completions"
    #         }
    #     ],
    #     coordinator={
    #         "model": "anthropic/claude-3.5-sonnet",
    #         "api_key": "[REDACTED]",
    #         "api_url": "https://openrouter.ai/api/v1/chat/completions"
    #     },
    #     verbose=True,
    #     chain_store_api_key=cohere_api_key,
    #     max_workers=3  # Optional: control parallel execution
    # )

    # print("\nEnsemble Response:")
    # print(ensemble_response)

if __name__ == "__main__":
    main()
