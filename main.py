# main.py
from dotenv import load_dotenv
import os
from engine import complete_reasoning_task
from mixture import ensemble
import chain_store

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
JINA_API_KEY = os.environ.get("JINA_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

def save_chain_prompt() -> bool:
    """Prompt user if they want to save the chain."""
    while True:
        response = input("\nWould you like to save this reasoning chain for future reference? (y/n): ").lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("Please answer 'y' or 'n'")

def main():
    # Initialize store
    chain_store.init_store()

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

    model = "anthropic/claude-3.5-sonnet"
    api_url = "https://openrouter.ai/api/v1/chat/completions"

    # Run the engine
    response, conversation_history, thinking_tools, output_tools = complete_reasoning_task(
        task=task,
        api_key=OPENROUTER_API_KEY,
        model=model,
        api_url=api_url,
        verbose=True,
        use_planning=False,
        jina_api_key=JINA_API_KEY
    )

    # Check if the run was successful (no errors in response)
    if isinstance(response, dict) and not response.get('error'):
        # Ask user if they want to save the chain
        if save_chain_prompt():
            try:
                # Save the chain
                chain_store.save_successful_chain(
                    task=task,
                    conversation_history=conversation_history,
                    final_response=response,
                    cohere_api_key=COHERE_API_KEY,
                    thinking_tools=thinking_tools,
                    output_tools=output_tools,
                    metadata={"model": model, "api_url": api_url}
                )
                print("Chain saved successfully!")
            except Exception as e:
                print(f"Error saving chain: {str(e)}")
    else:
        print("Run contained errors - skipping chain save prompt")

    return response, conversation_history, thinking_tools, output_tools

if __name__ == "__main__":
    main()