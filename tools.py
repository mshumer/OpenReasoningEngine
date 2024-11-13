import os
import requests
from typing import Dict, Any, List, Union
import sys
from io import StringIO
import traceback
from contextlib import redirect_stdout, redirect_stderr
import json
import wolframalpha
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sympy
import scipy
import sklearn
from sympy import symbols, solve, simplify
from scipy import stats
from sklearn import preprocessing
import math

# Dictionary of interpreter states, keyed by task hash
interpreter_states = {}

def get_task_hash(task: str) -> str:
    """Generate a unique hash for a task."""
    import hashlib
    return hashlib.md5(task.encode()).hexdigest()

def clear_interpreter_state(task: str = None):
    """
    Clear the interpreter state.
    If task is provided, only clear that task's state.
    If no task is provided, clear all states.
    """
    global interpreter_states
    if task:
        task_hash = get_task_hash(task)
        if task_hash in interpreter_states:
            del interpreter_states[task_hash]
    else:
        interpreter_states = {}

def python_interpreter(code: str, task: str, timeout: int = 5) -> str:
    """
    Safely execute Python code in a restricted environment.
    Maintains separate state for each task.
    """
    stdout = StringIO()
    stderr = StringIO()
    
    try:
        # Get task-specific state
        task_hash = get_task_hash(task)
        if task_hash not in interpreter_states:
            interpreter_states[task_hash] = {}
            
        state = interpreter_states[task_hash]
        
        # Initialize state if empty
        if not state:
            # Configure matplotlib to use non-interactive backend
            import matplotlib
            matplotlib.use('Agg')
            
            state.update({
                'np': np,
                'pd': pd,
                'plt': plt,
                'sns': sns,
                'sympy': sympy,
                'symbols': symbols,
                'solve': solve,
                'simplify': simplify,
                'scipy': scipy,
                'stats': stats,
                'sklearn': sklearn,
                'preprocessing': preprocessing,
                'math': math,
                'log': math.log,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'pi': math.pi,
                'e': math.e,
                'sqrt': math.sqrt,
                'exp': math.exp
            })
        
        # Take snapshot of state before execution
        pre_exec_state = {k: str(v) for k, v in state.items()}
        
        # Redirect stdout and stderr
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                # Execute the code with the state
                exec_result = eval(code, {"__builtins__": __builtins__}, state)
                
                # If there's a return value from eval, print it
                if exec_result is not None:
                    print(f"Result: {exec_result}")
                    
            except SyntaxError:
                # If eval fails (e.g., for statements), fall back to exec
                exec(code, {"__builtins__": __builtins__}, state)
            except Exception as e:
                error_msg = (
                    f"Error executing code: {str(e)}\n"
                    f"Traceback:\n{traceback.format_exc()}\n"
                    "\nDebugging Suggestions:\n"
                    "1. Add print statements to debug the issue\n"
                    "2. Use assertions to validate inputs and outputs\n"
                    "3. Check variable types with print(type(var))\n"
                    "4. For numerical computations, verify inputs are numbers\n"
                    "5. For symbolic math, ensure variables are properly defined with symbols()\n"
                    "\nNote: Plotting is currently not supported. Instead of visualizing data, consider:\n"
                    "1. Printing key numerical results\n"
                    "2. Showing data statistics\n"
                    "3. Printing array slices or samples\n"
                    "\nAvailable packages:\n"
                    "- numpy (np): Numerical computing\n"
                    "- pandas (pd): Data manipulation\n"
                    "- sympy: Symbolic mathematics\n"
                    "- scipy: Scientific computing\n"
                    "- sklearn: Machine learning"
                )
                return error_msg
            
        # Get the output
        output = stdout.getvalue()
        errors = stderr.getvalue()
        
        # Combine output and errors
        result = []
        if output:
            result.append(f"Output:\n{output.rstrip()}")
        if errors:
            result.append(f"Errors:\n{errors.rstrip()}")
        
        # Track new and modified variables
        new_vars = {}
        modified_vars = {}
        excluded_vars = {
            'np', 'pd', 'plt', 'sns', 'sympy', 'scipy', 'sklearn',
            'symbols', 'solve', 'simplify', 'stats', 'preprocessing',
            'math', 'log', 'sin', 'cos', 'tan', 'pi', 'e', 'sqrt', 'exp'
        }
        
        for k, v in state.items():
            if not k.startswith('__') and k not in excluded_vars:
                str_val = str(v)
                if len(str_val) > 1500:
                    str_val = str_val[:1500] + "..."
                    
                if k not in pre_exec_state:
                    new_vars[k] = f"{type(v).__name__} = {str_val}"
                elif pre_exec_state[k] != str_val:
                    modified_vars[k] = f"{type(v).__name__} = {str_val}"
        
        # Add variable information to result
        if new_vars:
            result.append(f"New variables:\n{', '.join(f'{k}: {v}' for k, v in new_vars.items())}")
        if modified_vars:
            result.append(f"Modified variables:\n{', '.join(f'{k}: {v}' for k, v in modified_vars.items())}")
            
        return "\n\n".join(result) if result else "Code executed successfully with no output."
        
    except Exception as e:
        return f"Error executing code: {str(e)}\n{traceback.format_exc()}"
    finally:
        stdout.close()
        stderr.close()
        if 'plt' in locals():
            plt.close('all')

def find_datapoint_on_web(
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

def wolfram(
    query: str,
    wolfram_app_id: str,
    include_pods: List[str] = None,  # e.g., ["Result", "Solution", "Plot"]
    max_width: int = 1000
) -> str:
    """
    Query Wolfram Alpha for computations, math, science, and knowledge.
    
    Args:
        query: The query to send to Wolfram Alpha
        wolfram_app_id: Your Wolfram Alpha API key
        include_pods: List of pod names to include in result (None for all)
        max_width: Maximum width for plots/images
    
    Returns:
        str: Formatted response from Wolfram Alpha
    """
    try:
        client = wolframalpha.Client(wolfram_app_id)
        res = client.query(query, width=max_width)
        
        # Format the response
        result = []
        for pod in res.pods:
            # Skip if we're only interested in specific pods and this isn't one of them
            if include_pods and pod.title not in include_pods:
                continue
                
            if pod.title and pod.text:
                result.append(f"{pod.title}:\n{pod.text}")
        
        return "\n\n".join(result) if result else "No results found"
        
    except Exception as e:
        return f"Error querying Wolfram Alpha: {str(e)}"

def execute_tool(tool_name: str, parameters: Dict[str, Any], task: str = None, api_key: str = None, model: str = None, api_url: str = None, wolfram_app_id: str = None) -> Any:
    """
    Execute the specified tool with the given parameters.
    """
    tools = {
        "python": python_interpreter,
        "find_datapoint_on_web": find_datapoint_on_web,
        "wolfram": wolfram
    }
    
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
        
    tool_func = tools[tool_name]
    
    # Remove thread_id from parameters if it exists
    if 'thread_id' in parameters:
        del parameters['thread_id']
    
    # Inject appropriate credentials and task
    if tool_name == "python":
        parameters = {**parameters, "task": task}
    elif tool_name == "find_datapoint_on_web":
        parameters = {**parameters, "api_key": api_key, "api_url": api_url}
    elif tool_name == "wolfram":
        parameters = {**parameters, "wolfram_app_id": wolfram_app_id}
    
    return tool_func(**parameters) 