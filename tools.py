from typing import Dict, Any

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

def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Any:
    """
    Execute the specified tool with the given parameters.
    
    Args:
        tool_name (str): The name of the tool to execute
        parameters (Dict[str, Any]): The parameters for the tool
        
    Returns:
        Any: The result of the tool execution
    """
    tools = {
        "calculator": calculator
    }
    
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
        
    tool_func = tools[tool_name]
    return tool_func(**parameters) 