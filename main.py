# main.py

from engine import complete_reasoning_task

def main():
    # Let's use a task that requires calculation
    task = "Calculate the compound interest on $1000 invested for 5 years at 5% annual interest rate, compounded annually."
    api_key = 'YOUR-API-KEY-HERE'
    model = 'anthropic/claude-3.5-sonnet'
    api_url = 'https://openrouter.ai/api/v1/chat/completions'
    temperature = 0.7
    top_p = 1.0
    max_tokens = 500
    verbose = True
    log_conversation = True

    final_response = complete_reasoning_task(
        task=task,
        api_key=api_key,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_url=api_url,
        verbose=verbose,
        log_conversation=log_conversation
    )

    print("\nFinal Response:")
    print(final_response)

if __name__ == "__main__":
    main()
