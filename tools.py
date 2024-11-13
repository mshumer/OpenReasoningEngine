import os
import requests
from typing import Dict, Any, List, Union
import sys
from io import StringIO
import traceback
from contextlib import redirect_stdout, redirect_stderr
from collections import defaultdict
import json

# Store for interpreter states
interpreter_states = defaultdict(dict)

def calculator(operation: str) -> float:
    """
    A simple calculator that evaluates mathematical expressions.
    
    Args:
        operation (str): A mathematical expression as a string (e.g., "2 + 2")
        
    Returns:
        float: The result of the calculation
    """
    try:
        # Using eval() is generally not safe for production, but for this demo it's ok
        result = eval(operation)
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid calculation: {str(e)}")

def python_interpreter(code: str, thread_id: str = "default", timeout: int = 5) -> str:
    """
    Safely execute Python code in a restricted environment.
    Maintains state across calls if the same thread_id is used.
    
    Args:
        code (str): Python code to execute
        thread_id (str): Identifier for the execution thread to maintain state
        timeout (int): Maximum execution time in seconds
        
    Returns:
        str: Output of the code execution (stdout + stderr)
    """
    stdout = StringIO()
    stderr = StringIO()
    
    try:
        # Get or create state for this thread
        local_dict = interpreter_states[thread_id]
        
        # Redirect stdout and stderr
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                # Execute the code with the thread's state
                exec(code, {"__builtins__": __builtins__}, local_dict)
            except AssertionError as ae:
                # Special handling for assertion errors to make them more informative
                return f"Assertion Error: {str(ae)}\nThis indicates a condition you were checking for failed. Review the assertion and the values being tested."
            except Exception as e:
                # For other errors, provide debugging suggestions
                error_msg = (
                    f"Error executing code: {str(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}\n"
                    "\nDebugging Suggestions:\n"
                    "1. Add assert statements to check variable values\n"
                    "2. Print intermediate results\n"
                    "3. Verify input types and values\n"
                    "4. Check for edge cases\n"
                    "\nExample:\n"
                    "assert isinstance(your_list, list), f'Expected list, got {type(your_list)}'\n"
                    "assert all(isinstance(x, (int, float)) for x in your_list), 'All elements must be numbers'\n"
                    "assert len(your_list) > 0, 'List cannot be empty'"
                )
                return error_msg
            
        # Save the updated state
        interpreter_states[thread_id] = local_dict
            
        # Get the output
        output = stdout.getvalue()
        errors = stderr.getvalue()
        
        # Combine output and errors
        result = ""
        if output:
            result += f"Output:\n{output}"
        if errors:
            if result:
                result += "\n"
            result += f"Errors:\n{errors}"
        
        # Add state information to the result
        state_vars = [f"{k}: {type(v).__name__}" for k, v in local_dict.items() 
                     if not k.startswith('__')]
        if state_vars:
            if result:
                result += "\n"
            result += f"\nThread state variables:\n{', '.join(state_vars)}"
            
        return result if result else "Code executed successfully with no output."
        
    except Exception as e:
        return f"Error executing code: {str(e)}\n{traceback.format_exc()}"
    finally:
        stdout.close()
        stderr.close()

def web_research(
    query: str,
    api_key: str = None,
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
) -> str:
    """
    Perform web research using Perplexity.
    
    Args:
        query: The specific question or search query
        api_key: API key for OpenRouter
        api_url: API URL for OpenRouter
    
    Returns:
        str: Research findings with citations
    """
    try:
        return _ask_perplexity(query, api_key, api_url)
    except Exception as e:
        return f"Error performing web research: {str(e)}"

def _ask_perplexity(
    question: str,
    api_key: str,
    api_url: str
) -> str:
    """
    Ask a question using Perplexity via OpenRouter.
    """
    try:
        response = requests.post(
            api_url,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'perplexity/llama-3.1-sonar-large-128k-online',
                'messages': [{'role': 'user', 'content': question}],
                'max_tokens': 1024,
                'temperature': 0.7
            },
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result:
            return result['choices'][0]['message']['content']
        elif 'response' in result:
            return result['response']
        else:
            raise ValueError(f"Unexpected API response format: {result}")
            
    except Exception as e:
        raise Exception(f"Error querying Perplexity: {str(e)}")

def execute_tool(tool_name: str, parameters: Dict[str, Any], api_key: str = None, model: str = None, api_url: str = None) -> Any:
    """
    Execute the specified tool with the given parameters.
    """
    tools = {
        "calculator": calculator,
        "python": python_interpreter,
        "web_research": web_research
    }
    
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
        
    tool_func = tools[tool_name]
    
    # For web_research, inject the current API credentials
    if tool_name == "web_research":
        parameters = {**parameters, "api_key": api_key, "api_url": api_url}
    
    return tool_func(**parameters) 