from colorama import Fore, Style
import requests
from typing import List, Dict
import concurrent.futures
import os


OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def send_message_to_api(
    task: str,
    messages: List[Dict],
    api_key: str,
    tools: List[Dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 500,
    api_url: str = "https://openrouter.ai/api/v1/chat/completions",
    verbose: bool = False,
    is_first_step: bool = False,
) -> Dict:
    """
    Send a message to the OpenRouter API and return the assistant's response.
    Will retry up to 3 times with increasing delay between retries.
    """
    if verbose and is_first_step:
        print(
            f"\n{Fore.CYAN}╭──────────────────────────────────────────{Style.RESET_ALL}"
        )
        print(f"{Fore.CYAN}│ Sending Request to API{Style.RESET_ALL}")
        print(
            f"{Fore.CYAN}├──────────────────────────────────────────{Style.RESET_ALL}"
        )
        print(f"{Fore.CYAN}│ Model: {Style.RESET_ALL}{model}")
        print(f"{Fore.CYAN}│ URL: {Style.RESET_ALL}{api_url}")
        print(f"{Fore.CYAN}│ Temperature: {Style.RESET_ALL}{temperature}")
        print(
            f"{Fore.CYAN}╰──────────────────────────────────────────{Style.RESET_ALL}\n"
        )

    retries = 0
    max_retries = 3
    delay = 1  # Initial delay in seconds

    while retries <= max_retries:
        try:
            print(
                f"\n{Fore.BLUE}Making API Request (Attempt {retries + 1}/{max_retries + 1})...{Style.RESET_ALL}"
            )
            response = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "tools": tools if tools else None,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                timeout=60,
            )
            print(f"{Fore.GREEN}Response received:{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{response.json()}{Style.RESET_ALL}")

            if verbose:
                print(
                    f"{Fore.YELLOW}Response status: {response.status_code}{Style.RESET_ALL}"
                )

            if response.status_code != 200:
                raise Exception(
                    f"API request failed with status {response.status_code}: {response.text}"
                )

            response_data = response.json()
            print(f"{Fore.GREEN}Successfully parsed response data{Style.RESET_ALL}")
            return response_data["choices"][0]["message"]

        except Exception as error:
            print(
                f"{Fore.RED}Error occurred during API call (Attempt {retries + 1})!{Style.RESET_ALL}"
            )
            print(f"{Fore.RED}{str(error)}{Style.RESET_ALL}")

            if retries == max_retries:
                raise Exception(
                    f"Error sending message to API after {max_retries + 1} attempts: {str(error)}"
                )

            import time

            wait_time = delay * (2**retries)  # Exponential backoff
            print(
                f"{Fore.YELLOW}Waiting {wait_time} seconds before retrying...{Style.RESET_ALL}"
            )
            time.sleep(wait_time)
            retries += 1


def generate_multiple_candidates(
    task: str,
    messages: List[Dict],
    api_key: str,
    tools: List[Dict],
    num_candidates: int = 3,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 500,
    api_url: str = "https://openrouter.ai/api/v1/chat/completions",
    verbose: bool = False,
    is_first_step: bool = False,
) -> List[Dict]:
    """
    Generate multiple candidate responses in parallel using concurrent.futures.
    Returns a list of candidate responses.
    """
    print(
        f"\n{Fore.MAGENTA}╭──────────────────────────────────────────{Style.RESET_ALL}"
    )
    print(f"{Fore.MAGENTA}│ Generating {num_candidates} Candidates{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}╰──────────────────────────────────────────{Style.RESET_ALL}")

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
            is_first_step=is_first_step,
        )

    candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_candidates) as executor:
        print(f"{Fore.CYAN}Starting parallel candidate generation...{Style.RESET_ALL}")
        future_to_candidate = {
            executor.submit(generate_candidate): i for i in range(num_candidates)
        }
        for future in concurrent.futures.as_completed(future_to_candidate):
            try:
                candidate = future.result()
                candidates.append(candidate)
                print(
                    f"{Fore.GREEN}Successfully generated candidate {len(candidates)}/{num_candidates}{Style.RESET_ALL}"
                )
            except Exception as e:
                print(
                    f"{Fore.RED}Error generating candidate: {str(e)}{Style.RESET_ALL}"
                )

    print(
        f"{Fore.GREEN}Generated {len(candidates)} candidates successfully{Style.RESET_ALL}"
    )
    return candidates


def generate_best_candidate(
    task: str,
    messages: List[Dict],
    api_key: str,
    tools: List[Dict],
    num_candidates: int = 3,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 500,
    api_url: str = "https://openrouter.ai/api/v1/chat/completions",
    verbose: bool = False,
    is_first_step: bool = False,
) -> Dict:
    """
    Generate a list of candidate responses and return the best one.
    """
    print(f"\n{Fore.CYAN}╭──────────────────────────────────────────{Style.RESET_ALL}")
    print(f"{Fore.CYAN}│ Starting Best Candidate Selection{Style.RESET_ALL}")
    print(f"{Fore.CYAN}╰──────────────────────────────────────────{Style.RESET_ALL}")

    candidates = generate_multiple_candidates(
        task,
        messages,
        api_key,
        tools,
        num_candidates,
        model,
        temperature,
        top_p,
        max_tokens,
        api_url,
        verbose,
        is_first_step,
    )

    print(f"\n{Fore.YELLOW}Generated Candidates:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{candidates}{Style.RESET_ALL}")

    print(f"\n{Fore.MAGENTA}Preparing evaluation prompt...{Style.RESET_ALL}")
    evaluation_prompt = ""

    i = 1
    for candidate in candidates:
        evaluation_prompt += f"Candidate {i}:\n{candidate}\n\n"
        i += 1

    SYSTEM_PROMPT = """You are a judge tasked with evaluating the viability of multiple candidate responses to a given task. Your goal is to identify the candidate that is most likely to lead to solving the task properly.

You will be given a <task> which describes the task at hand, a <previous_thoughts> section which contains the thoughts of the assistant before receiving the candidate responses, and a <next_thought_candidates> section which contains the candidate responses to be evaluated.

Evaluate the viability of each candidate response and output the number of the candidate that is most likely to lead to solving the task properly.

Do so in the following format:
<thinking>
Think through the viability of each candidate here.
</thinking>

<best_candidate_number>
Number of the best candidate
</best_candidate_number>
"""

    evaluation_prompt += f"""<task>{task}</task>

<previous_thoughts>
{messages}
</previous_thoughts>

<next_thought_candidates>
{evaluation_prompt}
</next_thought_candidates>

Think it through inside the <thinking> section, and then output the number of the candidate that is most likely to lead to solving the <task> properly in the <best_candidate_number> section. In the <best_candidate_number> section, only output the number, nothing else. Possible numbers are: {', '.join(str(i) for i in range(1, num_candidates + 1))}"""

    print(f"\n{Fore.BLUE}Sending evaluation request to API...{Style.RESET_ALL}")
    best_candidate_response = send_message_to_api(
        task="",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": evaluation_prompt},
        ],
        api_key=api_key,
        tools=tools,
    )

    # Parse the best candidate number from the response
    best_candidate_number = int(
        best_candidate_response["content"]
        .split("<best_candidate_number>")[1]
        .split("</best_candidate_number>")[0]
        .strip()
    )

    print(f"\n{Fore.GREEN}Selected best candidate:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{best_candidate_number}{Style.RESET_ALL}")

    # Return the best candidate
    return candidates[best_candidate_number - 1]
