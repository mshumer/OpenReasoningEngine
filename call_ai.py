from colorama import Fore, Style
import requests
from typing import List, Dict
import concurrent.futures

def send_message_to_api(
    task: str,
    messages: List[Dict],
    api_key: str,
    tools: List[Dict],
    model: str = 'gpt-4o-mini',
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 500,
    api_url: str = 'https://api.openai.com/v1/chat/completions',
    verbose: bool = False,
    is_first_step: bool = False
) -> Dict:
    """
    Send a message to the OpenAI API and return the assistant's response.
    """
    if verbose and is_first_step:
        print(f"\n{Fore.CYAN}╭──────────────────────────────────────────{Style.RESET_ALL}")
        print(f"{Fore.CYAN}│ Sending Request to API{Style.RESET_ALL}")
        print(f"{Fore.CYAN}├──────────────────────────────────────────{Style.RESET_ALL}")
        print(f"{Fore.CYAN}│ Model: {Style.RESET_ALL}{model}")
        print(f"{Fore.CYAN}│ URL: {Style.RESET_ALL}{api_url}")
        print(f"{Fore.CYAN}│ Temperature: {Style.RESET_ALL}{temperature}")
        print(f"{Fore.CYAN}╰──────────────────────────────────────────{Style.RESET_ALL}\n")

    try:
        response = requests.post(
            api_url,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': model,
                'messages': messages,
                'tools': tools,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
            },
            timeout=60
        )
        print(f"Response: {response.json()}")

        if verbose:
            print(f"{Fore.YELLOW}Response status: {response.status_code}{Style.RESET_ALL}")

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        response_data = response.json()
        return response_data['choices'][0]['message']

    except Exception as error:
        raise Exception(f'Error sending message to API: {str(error)}')

def generate_multiple_candidates(
    task: str,
    messages: List[Dict],
    api_key: str,
    tools: List[Dict],
    num_candidates: int = 3,
    model: str = 'gpt-4o-mini',
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 500,
    api_url: str = 'https://api.openai.com/v1/chat/completions',
    verbose: bool = False,
    is_first_step: bool = False
) -> List[Dict]:
    """
    Generate multiple candidate responses in parallel using concurrent.futures.
    Returns a list of candidate responses.
    """
    def generate_candidate():
        return send_message_to_api(
            task=task,
            messages=messages,
            api_key=api_key,
            tools=tools,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            api_url=api_url,
            verbose=verbose,
            is_first_step=is_first_step
        )

    candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_candidates) as executor:
        future_to_candidate = {executor.submit(generate_candidate): i for i in range(num_candidates)}
        for future in concurrent.futures.as_completed(future_to_candidate):
            try:
                candidate = future.result()
                candidates.append(candidate)
            except Exception as e:
                if verbose:
                    print(f"{Fore.RED}Error generating candidate: {str(e)}{Style.RESET_ALL}")

    return candidates