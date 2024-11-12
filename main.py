# main.py

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
    Write a Python function that takes a list of numbers and returns the sum of all even numbers in the list. 
    Test it with the list [1, 2, 3, 4, 5, 6] using the interpreter tool.
    """
    
    # Simple mode: just run one model
    # response, history, tools = complete_reasoning_task(
    #     task=task,
    #     api_key="[REDACTED]",
    #     model="anthropic/claude-3.5-sonnet",
    #     api_url="https://openrouter.ai/api/v1/chat/completions",
    #     verbose=True
    # )
    
    # Ensemble mode: run multiple models in parallel
    ensemble_response = ensemble(
        task=task,
        agents=[
            {
                "model": "openai/gpt-4o",
                "api_key": "[REDACTED]",
                "api_url": "https://openrouter.ai/api/v1/chat/completions"
            },
            {
                "model": "anthropic/claude-3.5-sonnet",
                "api_key": "[REDACTED]",
                "api_url": "https://openrouter.ai/api/v1/chat/completions"
            }
        ],
        coordinator={
            "model": "anthropic/claude-3.5-sonnet",
            "api_key": "[REDACTED]",
            "api_url": "https://openrouter.ai/api/v1/chat/completions"
        },
        verbose=True,
        chain_store_api_key=cohere_api_key,
        max_workers=3  # Optional: control parallel execution
    )
    
    print("\nEnsemble Response:")
    print(ensemble_response)

if __name__ == "__main__":
    main()
