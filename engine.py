import os
import requests
from typing import List, Dict, Optional, Tuple
from colorama import init, Fore, Style
from tools import execute_tool
import json
from datetime import datetime
from chain_store import (
    get_similar_chains, 
    prepare_examples_messages
)

# Initialize colorama for cross-platform colored output
init()

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

def thinking_loop(
    task: str,
    api_key: str,
    tools: List[Dict],
    model: str = 'gpt-4o-mini',
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 500,
    api_url: str = 'https://api.openai.com/v1/chat/completions',
    verbose: bool = False,
    chain_store_api_key: Optional[str] = None
) -> List[Dict]:
    """
    Execute the thinking loop and return the conversation history.
    """
    conversation_history = []
    continue_loop = True
    step_count = 1

    if verbose:
        print(f"\n{Fore.MAGENTA}╭──────────────────────────────────────────{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}│ Starting Thinking Loop{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}╰──────────────────────────────────────────{Style.RESET_ALL}\n")

    # Get similar chains if chain_store_api_key is provided
    example_messages = []
    if chain_store_api_key:
        similar_chains = get_similar_chains(task, chain_store_api_key)
        example_messages = prepare_examples_messages(similar_chains, tools)

    # Create the system message for the current task
    system_message = {
        'role': 'system',
        'content': (
            f"<CURRENT_TASK>\n{task}\n\n"
            "<INSTRUCTIONS>\n"
            "Slow down your thinking by breaking complex questions into multiple reasoning steps.\n"
            "Each individual reasoning step should be brief.\n"
            "When you need to perform calculations, use the calculator tool.\n"
            "Return <DONE> after the last step.\n"
            "The EXAMPLE_TASK(s) above are examples of how to break complex questions into multiple reasoning steps. Use these examples to guide your own thinking for the CURRENT_TASK."
        )
    }

    # Start with example messages and system message in the history
    full_conversation_history = example_messages + [system_message]

    while continue_loop:
        if verbose:
            print(f"\n{Fore.BLUE}Step {step_count}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{'─' * 40}{Style.RESET_ALL}")

        # Add user message
        user_message = {
            'role': 'user',
            'content': (
                'Think about your next reasoning step to perform the CURRENT_TASK. '
                'Return just the next step. If you need to calculate something, use the calculator tool. '
                'Remember, steps should be very brief. '
                'If this is the final step, return <DONE>.'
            )
        }
        
        # Add to both conversation histories
        conversation_history.append(user_message)
        full_conversation_history.append(user_message)

        # Get response from AI API
        response = send_message_to_api(
            task,
            full_conversation_history,
            api_key,
            tools,
            model,
            temperature,
            top_p,
            max_tokens,
            api_url,
            verbose,
            is_first_step=(step_count == 1)
        )

        # Add assistant's response to both histories
        assistant_message = {
            'role': 'assistant',
            'content': response.get('content'),
            'tool_calls': response.get('tool_calls', None)
        }
        conversation_history.append(assistant_message)
        full_conversation_history.append(assistant_message)

        if verbose and response.get('content'):
            print(f"\n{Fore.GREEN}Assistant: {Style.RESET_ALL}{response['content']}")

        # Handle tool calls
        if 'tool_calls' in response and response['tool_calls']:
            for tool_call in response['tool_calls']:
                if verbose:
                    print(f"\n{Fore.YELLOW}╭──────────────────────────────────────────{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}│ Tool Call Detected{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}├──────────────────────────────────────────{Style.RESET_ALL}")
                
                # Execute tool and get result
                tool_name = tool_call['function']['name']
                arguments = json.loads(tool_call['function']['arguments'])
                
                if verbose:
                    print(f"{Fore.YELLOW}│ Tool: {Style.RESET_ALL}{tool_name}")
                    print(f"{Fore.YELLOW}│ Arguments: {Style.RESET_ALL}{json.dumps(arguments, indent=2)}")
                
                result = execute_tool(tool_name, arguments)
                
                # Add tool result to both histories
                tool_message = {
                    'role': 'tool',
                    'tool_call_id': tool_call['id'],
                    'content': str(result)
                }
                conversation_history.append(tool_message)
                full_conversation_history.append(tool_message)
                
                if verbose:
                    print(f"{Fore.YELLOW}│ Result: {Style.RESET_ALL}{result}")
                    print(f"{Fore.YELLOW}╰──────────────────────────────────────────{Style.RESET_ALL}\n")

        # Check for termination conditions
        if response.get('content'):
            termination_phrases = [
                '<done>', 'done', 'there is no next step.',
                'this conversation is complete', 'the conversation has ended.',
                'this conversation is finished.', 'the conversation has concluded.'
            ]

            if any(term in response['content'].lower() for term in termination_phrases):
                if verbose:
                    print(f"\n{Fore.MAGENTA}╭──────────────────────────────────────────{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}│ Thinking Loop Complete{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}│ Total Steps: {step_count}{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}╰──────────────────────────────────────────{Style.RESET_ALL}\n")
                continue_loop = False

        step_count += 1

    if verbose:
        print(f"Returning conversation history: {full_conversation_history}")

    return full_conversation_history

def complete_reasoning_task(
    task: str,
    api_key: Optional[str] = None,
    model: str = 'gpt-4o-mini',
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 500,
    api_url: str = 'https://api.openai.com/v1/chat/completions',
    verbose: bool = False,
    log_conversation: bool = False,
    chain_store_api_key: Optional[str] = None
) -> Tuple[str, List[Dict]]:
    """
    Complete a task using the step-by-step thinking process.
    Returns both the final response and the conversation history.
    """
    if api_key is None:
        raise ValueError('API key not provided.')

    if verbose:
        print(f"\n{Fore.MAGENTA}╭──────────────────────────────────────────{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}│ Starting Task{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}├──────────────────────────────────────────{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}│ {task}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}╰──────────────────────────────────────────{Style.RESET_ALL}\n")

    # Define available tools
    tools = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2' or '1000 * (1 + 0.05) ** 5')"
                    }
                },
                "required": ["operation"]
            }
        }
    }]

    # Run the thinking loop
    conversation_history = thinking_loop(
        task,
        api_key,
        tools,
        model,
        temperature,
        top_p,
        max_tokens,
        api_url,
        verbose,
        chain_store_api_key=chain_store_api_key
    )

    # Add final completion request
    final_user_message = {
        'role': 'user',
        'content': 'Complete the <CURRENT_TASK>. Do not return <DONE>. Note that the user will only see what you return here. None of the steps you have taken will be shown to the user, so ensure you return the final answer.'
    }
    conversation_history.append(final_user_message)

    if verbose:
        print(f"{Fore.CYAN}Requesting final response...{Style.RESET_ALL}\n")

    # Get final response
    final_response = send_message_to_api(
        task,
        conversation_history,
        api_key,
        tools,
        model,
        temperature,
        top_p,
        max_tokens,
        api_url,
        verbose
    )

    # Print final response if verbose
    if verbose and 'content' in final_response:
        print(f'\n{Fore.GREEN}Final Response:{Style.RESET_ALL}')
        print(final_response['content'])

    final_response = final_response.get('content', '')

    # Log conversation history if logging is enabled
    if log_conversation:
        # Remove example chains from conversation history by removing everything prior to the bottom-most system message
        bottom_system_message_index = next((i for i, msg in enumerate(reversed(conversation_history)) if msg.get('role') == 'system'), None)
        if bottom_system_message_index is not None:
            conversation_history = conversation_history[-bottom_system_message_index:]
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'logs/conversation_{timestamp}.json'
        
        # Prepare log data
        log_data = {
            'task': task,
            'model': model,
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'api_url': api_url,
            'conversation_history': conversation_history,
            'final_response': final_response
        }
        
        # Write to file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"\n{Fore.CYAN}Conversation history logged to: {Style.RESET_ALL}{filename}")
        except Exception as e:
            if verbose:
                print(f"\n{Fore.RED}Failed to log conversation history: {Style.RESET_ALL}{str(e)}")

    return final_response, conversation_history, tools
