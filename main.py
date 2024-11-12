# main.py

from engine import complete_reasoning_task
import chain_store

def main():
    # Initialize store
    chain_store.init_store()
    
    # API keys
    cohere_api_key = "REDACTED"
    api_key = 'REDACTED'
    
    # task = "Alice has N brothers and she also has M sisters. How many sisters does Alice's brother Andrew have? Think it through deeply, step by step. Ensure each step is very brief."
    # task = "Create a voice LLM chatbot powered by GPT-4 that is interruptible and knows when the user is interrupting. This should be insanely detailed and fully built. Hint: use two threads. Start by planning the architecture, then plan the code, then write each individual component, and finally integrate them."
    # task = "Assume the laws of physics on Earth. A small marble is put into a normal cup and the cup is placed upside down on a table. Someone then takes the cup without changing its orientation and puts it inside the microwave. Where is the marble now?"
    task = '2 + 2 * 2'
    model = 'anthropic/claude-3.5-sonnet'
    api_url = 'https://openrouter.ai/api/v1/chat/completions'
    temperature = 0.7
    top_p = 1.0
    max_tokens = 3000
    verbose = True
    log_conversation = False

    final_response, conversation_history, tools = complete_reasoning_task(
        task=task,
        api_key=api_key,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_url=api_url,
        verbose=verbose,
        log_conversation=log_conversation,
        chain_store_api_key=cohere_api_key
    )

    print("\nFinal Response:")
    print(final_response)
    
    # Handle chain saving
    save_chain = input("\nWas this response successful? Would you like to save this chain? (y/n): ").lower().strip()
    if save_chain == 'y':
        if chain_store.save_successful_chain(
            task, 
            conversation_history, 
            final_response, 
            cohere_api_key,
            tools
        ):
            print("Chain saved successfully!")
        else:
            print("Failed to save chain.")

if __name__ == "__main__":
    main()
