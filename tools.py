import os
import requests
from typing import Dict, Any, List, Union, Optional
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
from e2b_code_interpreter import Sandbox
from serpapi import GoogleSearch

serpapi_api_key = os.getenv("SERPAPI_API_KEY")

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

def python_interpreter(code: str, task: str, timeout: int = 5, sandbox: Optional[Sandbox] = None) -> str:
    """
    Safely execute Python code in a restricted environment.
    Maintains separate state for each task.
    """
    if sandbox is None:
        raise ValueError("E2B Sandbox is required for Python code execution but none was provided.")

    print(f"Executing code:\n{code}")
    execution = sandbox.run_code(
        code,
        # timeout=timeout, # Timeout to wait for the whole request to complete
        on_stdout=lambda x: print('[stdout]', x),
        on_stderr=lambda x: print('[stderr]', x)
    )

    if execution.error:
        e = execution.error

        error_msg = (
            f"Error executing code: {e.value}\n"
            f"Error type: {type(e.name)}\n"
            f"Traceback:\n{e.traceback}\n"
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
            "- scipy: Scientific computing\n"
            "- sklearn: Machine learning"
        )
        return error_msg

    result = []

    # Results are the output of the code execution besides stdout and stderr
    # Can be text, PNG, JPG, JSON, html, markdown, etc.
    # Results are based on executing code inside the headless Jupyter notebook
    # that's running inside the sandbox.
    # The same way, you'd get result from a Jupyter notebook cell, you get results here.
    # That means any display() calls in the code will be captured as a result,
    # and also the last expression in the code, if there is one.
    code_exec_results = execution.results
    for ce_result in code_exec_results:
        print(ce_result.formats()) # Raw data of results
        # if 'png' in ce_result.formats:
            # Handle PNG images
        # if 'json' in ce_result.formats:
            # Handle JSON
        # ...
        #
        # Text is always present for every result.
        result.append(ce_result.text)

    stdout = execution.logs.stdout
    stderr = execution.logs.stderr
    if stdout:
        result.append(f"Output:\n{''.join(stdout)}")
    if stderr:
        result.append(f"Errors:\n{''.join(stderr)}")
    return "\n\n".join(result) if result else "Code executed successfully with no output."

def find_datapoint_on_web(
    query: str,
    api_key: str = None,
) -> str:
    """
    Perform web search using SERPAPI Google Search.

    Args:
        query: The specific search query
        api_key: API key for SERPAPI
        api_url: Not used for SERPAPI

    Returns:
        str: Search results with citations
    """
    try:
        # Configure the search
        search = GoogleSearch({
            "q": query,
            "api_key": api_key,
            "num": 5  # Get top 5 results
        })
        
        # Get the results
        results = search.get_dict()
        
        if "error" in results:
            return f"Error performing search: {results['error']}"
            
        # Format organic results
        formatted_results = []
        
        if "organic_results" in results:
            for result in results["organic_results"]:
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description available")
                link = result.get("link", "No link available")
                formatted_results.append(f"Source: {title}\nSummary: {snippet}\nURL: {link}\n")
                
        if formatted_results:
            return "\n".join(formatted_results)
        else:
            return "No relevant results found for the query."
            
    except Exception as e:
        return f"Error performing web search: {str(e)}"

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

def execute_tool(
    tool_name: str,
    parameters: Dict[str, Any],
    task: str = None,
    api_key: str = None,
    model: str = None,
    api_url: str = None,
    wolfram_app_id: str = None,
    sandbox: Optional[Sandbox] = None
) -> Any:
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
        parameters = {**parameters, "task": task, "sandbox": sandbox}
    elif tool_name == "find_datapoint_on_web":
        parameters = {**parameters, "api_key": serpapi_api_key}
    elif tool_name == "wolfram":
        parameters = {**parameters, "wolfram_app_id": wolfram_app_id}

    return tool_func(**parameters)