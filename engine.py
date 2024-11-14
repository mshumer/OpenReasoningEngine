import os
import requests
from classes import ModelConfig
from e2b_code_interpreter import Sandbox
from typing import List, Dict, Optional, Tuple, Union
from colorama import init, Fore, Style
from tools import execute_tool, clear_interpreter_state
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
    tools: List[Dict],
    model_config: ModelConfig,
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
        print(f"{Fore.CYAN}│ Model: {Style.RESET_ALL}{model_config['model']}")
        print(f"{Fore.CYAN}│ URL: {Style.RESET_ALL}{model_config['api_url']}")
        print(f"{Fore.CYAN}│ Temperature: {Style.RESET_ALL}{model_config['temperature']}")
        print(f"{Fore.CYAN}╰──────────────────────────────────────────{Style.RESET_ALL}\n")

    try:
        response = requests.post(
            model_config['api_url'],
            headers={
                'Authorization': f'Bearer {model_config['api_key']}',
                'Content-Type': 'application/json'
            },
            json={
                'model': model_config['model'],
                'messages': messages,
                'tools': tools,
                'max_tokens': model_config['max_tokens'],
                'temperature': model_config['temperature'],
                'top_p': model_config['top_p'],
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

def thinking_loop(
    task: str,
    model_config: ModelConfig,
    tools: List[Dict],
    verbose: bool = False,
    chain_store_api_key: Optional[str] = None,
    wolfram_app_id: Optional[str] = None,
    max_reasoning_steps: Optional[int] = None,
    sandbox: Optional[Sandbox] = None,
    image: Optional[str] = None,
    reflection_mode: bool = False
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
        if max_reasoning_steps:
            print(f"{Fore.MAGENTA}│ Maximum steps: {max_reasoning_steps}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}╰──────────────────────────────────────────{Style.RESET_ALL}\n")

    # Get similar chains if chain_store_api_key is provided
    example_messages = []
    if chain_store_api_key:
        similar_chains = get_similar_chains(task, chain_store_api_key)
        example_messages = prepare_examples_messages(similar_chains, tools)

    # Create the system message for the current task
    tools_description = (
        "You have access to these tools:\n"
        "1. find_datapoint_on_web: Search the web for information using Perplexity\n"
        "2. python: For executing Python code"
    )

    if wolfram_app_id:
        tools_description += "\n3. wolfram: Query Wolfram Alpha for precise mathematical, scientific, and factual computations"

    system_message = {
        'role': 'system',
        'content': (
            f"<CURRENT_TASK>\n{task}\n\n"
            "<INSTRUCTIONS>\n"
            "Slow down your thinking by breaking complex questions into multiple reasoning steps.\n"
            "Each individual reasoning step should be brief.\n"
            f"{tools_description}\n\n"
            "When you need to write or test Python code, use the python tool.\n"
            "When you need to search the web for information, use the find_datapoint_on_web tool.\n"
            + (
                "When you need precise mathematical or scientific computations, use the wolfram tool.\n"
                if wolfram_app_id else ""
            ) +
            "\nWhen searching the web:\n"
            "- The find_datapoint_on_web tool uses human MTurk workers who do quick research and return what they find\n"
            "- Only ask simple, factual questions that can be directly looked up\n"
            "- Queries must be single, straightforward questions - no compound questions\n"
            "- Do not ask workers to make logical inferences or analyze information\n"
            "- If a query is rejected, simplify it to ask for just one basic fact\n"
            "- Keep queries focused on finding specific, verifiable information\n"
            "- If the worker notes data isn't directly available or makes logic jumps, break down into simpler questions to get just the raw facts and do the analysis yourself\n"
            "\nWhen writing Python code:\n"
            "- If your code produces an error, add print statements to debug the issue\n"
            "- Use assertions/prints to validate inputs, intermediate results, and outputs\n"
            "- Print the state to see what's happening\n"
            "- When an error occurs, systematically add checks to identify where the problem is\n"
            "- Structure your debugging process step by step\n"
            + (
                "\nWhen using Wolfram Alpha:\n"
                "- Use for precise mathematical calculations and scientific data\n"
                "- Phrase queries clearly and specifically\n"
                "- Great for unit conversions, equations, and factual data\n"
                if wolfram_app_id else ""
            ) +
            "\nReturn <DONE> after the last step.\n"
            + (
                "The EXAMPLE_TASK(s) above are examples of how to break complex questions into multiple reasoning steps. Use these examples to guide your own thinking for the CURRENT_TASK."
                if example_messages else ""
            )
        )
    }

    # Start with example messages and system message in the history
    full_conversation_history = example_messages + [system_message]

    if image:
        full_conversation_history.append({
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': f"Here is the image the user provided:"
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': image
                    }
                }
            ]
        })

    while continue_loop:
        # Check if we've exceeded max steps
        if max_reasoning_steps and step_count > max_reasoning_steps:
            if verbose:
                print(f"\n{Fore.YELLOW}Maximum reasoning steps ({max_reasoning_steps}) reached. Forcing completion.{Style.RESET_ALL}")

            # Add a system message explaining the forced stop
            force_stop_message = {
                'role': 'system',
                'content': (
                    f"Maximum reasoning steps ({max_reasoning_steps}) reached. "
                )
            }
            conversation_history.append(force_stop_message)
            full_conversation_history.append(force_stop_message)

            # Add a user message requesting the final answer
            final_user_message = {
                'role': 'user',
                'content': (
                    'Based on your reasoning so far, provide your final answer to the CURRENT_TASK. '
                    'Make your response complete and self-contained since this will be shown to the user.'
                    "Please provide your final answer based on what you've learned so far. "
                    "Do not return <DONE>, and **you are not allowed to use any tools**. Just respond with your final answer."
                )
            }
            conversation_history.append(final_user_message)
            full_conversation_history.append(final_user_message)

            # Get final response when hitting max steps
            response = send_message_to_api(
                task,
                full_conversation_history,
                tools,
                model_config,
                verbose
            )
            print('Final response:', response)

            # Add the final response to histories
            assistant_message = {
                'role': 'assistant',
                'content': response.get('content'),
                'tool_calls': response.get('tool_calls', None)
            }
            conversation_history.append(assistant_message)
            full_conversation_history.append(assistant_message)

            if verbose and response.get('content'):
                print(f"\n{Fore.GREEN}Final Response after max steps:{Style.RESET_ALL}")
                print(response.get('content'))

            # Return here to skip the additional final response request
            return full_conversation_history

        if verbose:
            print(f"\n{Fore.BLUE}Step {step_count}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{'─' * 40}{Style.RESET_ALL}")

        # Determine which message to send based on reflection mode and step count
        if reflection_mode and step_count % 2 == 0:
            # Even steps in reflection mode are for reflection
            user_message = {
                'role': 'user',
                'content': (
                    'Reflect on your last step — check for mistakes. '
                    'Consider:\n'
                    '1. Are your assumptions valid and well-justified?\n'
                    '2. Did you make any logical errors or jumps in reasoning?\n'
                    '3. Is there a more effective or efficient approach?\n'
                    'Explain your analysis, whether you find issues or confirm the step was sound.\n'
                    'Do not make a snap decision. Think carefully before deciding if the step is free of mistakes.\n'
                    'Be brief and to the point.\n'
                    'If this is the final step, return <DONE>.'
                )
            } # Note — these reflection steps are often a bit long, which may lead to the non-reflection steps doing more work per step than they should. Figure this out later.
        else:
            # Odd steps or non-reflection mode use the original message
            user_message = {
                'role': 'user',
                'content': (
                    'Think about your next reasoning step to perform the CURRENT_TASK. '
                    'Return just the next step. '
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
            tools,
            model_config,
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

                try:
                    # Execute tool and get result
                    tool_name = tool_call['function']['name']

                    # Add error handling for argument parsing
                    try:
                        if 'arguments' not in tool_call['function'] or not tool_call['function']['arguments']:
                            error_msg = "No arguments provided in tool call"
                            if verbose:
                                print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
                            raise ValueError(error_msg)

                        arguments = json.loads(tool_call['function']['arguments'])

                    except json.JSONDecodeError as e:
                        error_msg = f"Invalid JSON in tool arguments: {tool_call['function'].get('arguments', 'NO_ARGS')}"
                        if verbose:
                            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
                            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                        raise ValueError(error_msg)

                    if verbose:
                        print(f"{Fore.YELLOW}│ Tool: {Style.RESET_ALL}{tool_name}")
                        print(f"{Fore.YELLOW}│ Arguments: {Style.RESET_ALL}{json.dumps(arguments, indent=2)}")

                    result = execute_tool(
                        tool_name,
                        arguments,
                        task=task,
                        model_config=model_config,
                        wolfram_app_id=wolfram_app_id,
                        sandbox=sandbox,
                    )

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

                except Exception as e:
                    error_msg = str(e)
                    if verbose:
                        print(f"{Fore.RED}Error executing tool: {error_msg}{Style.RESET_ALL}")

                    # Add error message to conversation history so model can correct its approach
                    error_message = {
                        'role': 'system',
                        'content': (
                            f"Error using {tool_name} tool: {error_msg}\n"
                            "Please correct your approach and try again."
                        )
                    }
                    conversation_history.append(error_message)
                    full_conversation_history.append(error_message)
                    continue

        # Check for termination conditions
        if response.get('content'):
            termination_phrases = [
                '<done>', 'done', 'there is no next step.',
                'this conversation is complete', 'the conversation has ended.',
                'this conversation is finished.', 'the conversation has concluded.'
            ]

            if any(term in response['content'].lower() for term in termination_phrases):
                if verbose:
                    print(f"\n{Fore.MAGENTA}╭─────────────────────────────��────────────{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}│ Thinking Loop Complete{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}│ Total Steps: {step_count}{Style.RESET_ALL}")
                    print(f"{Fore.MAGENTA}╰──────────────────────────────────────────{Style.RESET_ALL}\n")
                continue_loop = False

        step_count += 1

    return full_conversation_history

def complete_reasoning_task(
    task: str,
    model_config: ModelConfig,
    verbose: bool = False,
    log_conversation: bool = False,
    chain_store_api_key: Optional[str] = None,
    wolfram_app_id: Optional[str] = None,
    max_reasoning_steps: Optional[int] = None,
    image: Optional[str] = None,
    output_tools: Optional[List[Dict]] = None,
    reflection_mode: bool = False
) -> Tuple[Union[str, Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Execute the reasoning task and return the final response.
    Now supports optional structured output via output_tools and reflection mode.
    """
    # Clear Python interpreter state for just this task
    clear_interpreter_state(task=task)

    if model_config['api_key'] is None:
        raise ValueError('API key not provided.')
    
    if model_config['api_url'] is None:
        model_config['api_url'] = 'https://api.openai.com/v1/chat/completions'
    if model_config['model'] is None:
        model_config['model'] = 'gpt-4o-mini'
    if model_config['temperature'] is None:
        model_config['temperature'] = 0.7
    if model_config['top_p'] is None:
        model_config['top_p'] = 1.0
    if model_config['max_tokens'] is None:
        model_config['max_tokens'] = 500

    if verbose:
        print(f"\n{Fore.MAGENTA}╭──────────────────────────────────────────{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}│ Starting Task{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}├──────────────────────────────────────────{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}│ {task}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}╰──────────────────────────────────────────{Style.RESET_ALL}\n")

    # Initialize E2B sandbox for Python code execution
    # Timeout says how long the sandbox can stay alive
    # You can extend the sandbox lifetime by calling sandbox.set_timeout
    # to reset the timeout to a new value.
    timeout = 60 * 10 # 10 minutes
    sandbox = Sandbox(timeout=timeout)


    # Define thinking tools (internal tools that can be used during reasoning)
    thinking_tools = [
        {
            "type": "function",
            "function": {
                "name": "python",
                "description": "Execute Python code and return the output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Maximum execution time in seconds",
                            "default": 5
                        }
                    },
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_datapoint_on_web",
                "description": "Search the web for a datapoint using Perplexity. Returns findings with citations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The specific question"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # Add Wolfram tool only if wolfram_app_id is provided
    if wolfram_app_id:
        thinking_tools.append({
            "type": "function",
            "function": {
                "name": "wolfram",
                "description": "Query Wolfram Alpha for computations, math, science, and knowledge. Great for mathematical analysis, scientific calculations, data analysis, and fact-checking.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to send to Wolfram Alpha. Be specific and precise."
                        },
                        "include_pods": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Optional list of pod names to include (e.g., ['Result', 'Solution', 'Plot']). Leave empty for all pods.",
                            "default": None
                        },
                        "max_width": {
                            "type": "integer",
                            "description": "Maximum width for plots/images",
                            "default": 1000
                        }
                    },
                    "required": ["query"]
                }
            }
        })

    # Add output tools description to system message
    output_tools_description = ""
    if output_tools:
        output_tools_description = "\n\nWhen providing your final response, you can use these output functions (but you don't have access to them during reasoning steps):\n"
        for tool in output_tools:
            output_tools_description += f"- {tool['function']['name']}: {tool['function']['description']}\n"

    # Modify system message to include output tools if provided
    system_message = {
        'role': 'system',
        'content': (
            f"<CURRENT_TASK>\n{task}\n\n"
            "<INSTRUCTIONS>\n"
            "Slow down your thinking by breaking complex questions into multiple reasoning steps.\n"
            "Each individual reasoning step should be brief.\n"
            f"{output_tools_description}\n"  # Add output tools description if any
            "When you need to write or test Python code, use the python tool.\n"
            "When you need to search the web for information, use the find_datapoint_on_web tool.\n"
            + (
                "When you need precise mathematical or scientific computations, use the wolfram tool.\n"
                if wolfram_app_id else ""
            ) +
            "\nWhen searching the web:\n"
            "- The find_datapoint_on_web tool uses human MTurk workers who do quick research and return what they find\n"
            "- Only ask simple, factual questions that can be directly looked up\n"
            "- Queries must be single, straightforward questions - no compound questions\n"
            "- Do not ask workers to make logical inferences or analyze information\n"
            "- If a query is rejected, simplify it to ask for just one basic fact\n"
            "- Keep queries focused on finding specific, verifiable information\n"
            "- If the worker notes data isn't directly available or makes logic jumps, break down into simpler questions to get just the raw facts and do the analysis yourself\n"
            "\nWhen writing Python code:\n"
            "- If your code produces an error, add print statements to debug the issue\n"
            "- Use assertions/prints to validate inputs, intermediate results, and outputs\n"
            "- Print the state to see what's happening\n"
            "- When an error occurs, systematically add checks to identify where the problem is\n"
            "- Structure your debugging process step by step\n"
            + (
                "\nWhen using Wolfram Alpha:\n"
                "- Use for precise mathematical calculations and scientific data\n"
                "- Phrase queries clearly and specifically\n"
                "- Great for unit conversions, equations, and factual data\n"
                if wolfram_app_id else ""
            ) +
            "\nReturn <DONE> after the last step.\n"
            "The EXAMPLE_TASK(s) above are examples of how to break complex questions into multiple reasoning steps. Use these examples to guide your own thinking for the CURRENT_TASK."
        )
    }

    # Run thinking loop with only thinking tools
    conversation_history = thinking_loop(
        task,
        model_config,
        thinking_tools,
        verbose,
        chain_store_api_key=chain_store_api_key,
        wolfram_app_id=wolfram_app_id,
        max_reasoning_steps=max_reasoning_steps,
        sandbox=sandbox,
        image=image,
        reflection_mode=reflection_mode
    )

    # Only request final response if we didn't hit max steps
    final_response = None
    if not max_reasoning_steps or len([m for m in conversation_history if m['role'] == 'system' and 'Maximum reasoning steps' in m.get('content', '')]) == 0:
        # Add final completion request
        final_user_message = {
            'role': 'user',
            'content': (
                'Complete the <CURRENT_TASK>. Do not return <DONE>. '
                'Note that the user will only see what you return here. '
                'None of the steps you have taken will be shown to the user, so ensure you return the final answer. '
                + ('You can return a text response and/or use one of the available output functions.' if output_tools else '')
            )
        }
        conversation_history.append(final_user_message)

        if verbose:
            print(f"{Fore.CYAN}Requesting final response...{Style.RESET_ALL}\n")

        # Get final response with output tools if provided
        final_response = send_message_to_api(
            task,
            conversation_history,
            model_config,
            output_tools if output_tools else thinking_tools,  # Use output tools for final response if provided
            verbose
        )
    else:
        # Use the last assistant message as the final response
        final_response = next(
            (msg for msg in reversed(conversation_history)
             if msg['role'] == 'assistant' and msg.get('content')),
            {'content': None}
        )

    # Print final response if verbose
    if verbose and ('content' in final_response or 'tool_calls' in final_response):
        print(f'\n{Fore.GREEN}Final Response:{Style.RESET_ALL}')
        if 'content' in final_response and 'tool_calls' in final_response:
            print(f"Content: {final_response['content']}")
            print(f"Tool Calls: {final_response['tool_calls']}")
        elif 'content' in final_response:
            print(final_response['content'])
        else:
            print(final_response['tool_calls'])

    if 'tool_calls' in final_response:
        final_response_tool_calls = final_response['tool_calls']
    else:
        final_response_tool_calls = None

    if 'content' in final_response:
        final_response_content = final_response['content']
    else:
        final_response_content = None

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
            'model': model_config['model'],
            'temperature': model_config['temperature'],
            'top_p': model_config['top_p'],
            'max_tokens': model_config['max_tokens'],
            'api_url': model_config['api_url'],
            'conversation_history': conversation_history,
            'final_response': final_response_content,
            'final_response_tool_calls': final_response_tool_calls,
            'thinking_tools': thinking_tools,
            'output_tools': output_tools
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

    return {'content': final_response_content, 'tool_calls': final_response_tool_calls}, conversation_history, thinking_tools, output_tools
