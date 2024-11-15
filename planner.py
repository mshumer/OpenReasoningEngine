from typing import List, Dict, Optional
from colorama import Fore, Style
import json

def format_tools_for_context(tools: List[Dict]) -> str:
    """Format tools list into a readable string for context."""
    tools_str = "Available Tools:\n"
    for tool in tools:
        if tool.get('type') == 'function':
            func = tool['function']
            tools_str += f"- {func['name']}: {func['description']}\n"
            
            # Add parameter details if they exist
            if 'parameters' in func and 'properties' in func['parameters']:
                tools_str += "  Parameters:\n"
                for param_name, param_details in func['parameters']['properties'].items():
                    tools_str += f"    - {param_name}: {param_details.get('description', 'No description')}\n"
    
    return tools_str

def format_chain_for_planning(
    chain: Dict,
    include_tool_calls: bool = True
) -> str:
    """
    Format a single chain into a concise summary focusing on key patterns and outcomes.
    """
    formatted = f"\nTask: {chain.get('task', 'Unknown task')}\n"
    
    # Add metadata if it exists
    if 'metadata' in chain:
        formatted += "Context:\n"
        for key, value in chain['metadata'].items():
            formatted += f"- {key}: {value}\n"
    
    # Add tools that were available
    if 'thinking_tools' in chain:
        formatted += "\n" + format_tools_for_context(chain['thinking_tools'])
    
    formatted += "\nSteps taken:\n"
    for msg in chain.get('conversation_history', []):
        if msg['role'] == 'assistant':
            step = f"- {msg.get('content', '')}"
            
            # Include tool calls if requested and they exist
            if include_tool_calls and msg.get('tool_calls'):
                for tool_call in msg['tool_calls']:
                    if tool_call['type'] == 'function':
                        func = tool_call['function']
                        step += f"\n  Tool used: {func['name']}"
                        try:
                            args = json.loads(func['arguments'])
                            step += f"\n  Arguments: {json.dumps(args, indent=2)}"
                        except:
                            step += f"\n  Arguments: {func['arguments']}"
            
            formatted += step + "\n"
        # Include tool responses for context
        elif msg['role'] == 'tool':
            content = msg.get('content', '')
            first_line = content.split('\n')[0] if content else ''
            formatted += f"  Result: {first_line}...\n"
    
    return formatted

def generate_plan(
    task: str,
    similar_chains: List[Dict],
    current_tools: List[Dict],
    api_key: str,
    model: str,
    api_url: str,
    verbose: bool = False,
    metadata: Optional[Dict] = None
) -> str:
    """
    Generate a plan of action based on similar chains from memory.
    Takes into account available tools and other context.
    """
    from engine import send_message_to_api
    
    if verbose:
        print(f"\n{Fore.CYAN}Extracting patterns from {len(similar_chains)} similar chains...{Style.RESET_ALL}")
    
    # Format current context
    current_context = f"Current Task: {task}\n"
    if metadata:
        current_context += "Current Context:\n"
        for key, value in metadata.items():
            current_context += f"- {key}: {value}\n"
    current_context += "\n" + format_tools_for_context(current_tools)
    
    # Format similar chains
    examples_context = ""
    for i, chain in enumerate(similar_chains, 1):
        examples_context += f"\nExample {i}:"
        examples_context += format_chain_for_planning(chain)
    
    # Create planning prompt
    planning_messages = [
        {
            'role': 'system',
            'content': (
                "You are an expert at breaking down complex tasks into clear steps and leveraging available tools effectively. "
                "Focus on providing strategic guidance about HOW to approach problems rather than specific solutions. "
                "Key aspects to consider:\n"
                "- How to break the problem into manageable steps\n"
                "- Which tools would be most helpful at each stage\n" 
                "- How to validate progress and handle potential issues\n"
                "- What patterns from past experiences could be applied"
            )
        },
        {
            'role': 'user',
            'content': (
                f"{current_context}\n"
                f"Similar Examples:{examples_context}\n\n"
                "Based on these examples and the available tools/resources, outline a strategic approach for this task:\n"
                "1. How would you break this down into clear steps?\n"
                "2. Which tools would be most valuable at each stage?\n"
                "3. What key checkpoints or validation should be included?\n"
                "4. What patterns from similar past tasks could guide the approach?\n\n"
                "Focus on the process and methodology rather than specific implementation details.\n"
                "Keep it concise and super high-level, like you're having a quick chat with a colleague. Maximum 200 words."
            )
        }
    ]

    if verbose:
        print(f"{Fore.CYAN}Analyzing patterns and generating plan...{Style.RESET_ALL}")

    try:
        response = send_message_to_api(
            task,
            planning_messages,
            api_key,
            [],  # No tools needed for planning
            model,
            temperature=0.7,
            top_p=1.0,
            max_tokens=1000,  # Increased for more detailed plans
            api_url=api_url,
            verbose=verbose
        )
        
        plan = response.get('content', '')
        
        if verbose:
            print(f"\n{Fore.GREEN}Generated Plan:{Style.RESET_ALL}")
            print(plan)
            
        return plan
        
    except Exception as e:
        if verbose:
            print(f"\n{Fore.RED}Error generating plan: {str(e)}{Style.RESET_ALL}")
        return "Failed to generate plan from similar examples." 