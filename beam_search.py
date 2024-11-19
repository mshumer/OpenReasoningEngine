from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from api_utils import send_message_to_api
from colorama import Fore, Style
import time
import json
from e2b_code_interpreter import Sandbox
from tools import (
    interpreter_branches,
    create_branch,
    cleanup_unused_branches,
    execute_tool,
    record_command,
    execute_branch_history,
    cleanup_sandbox
)

def format_conversation_history(history: List[Dict]) -> str:
    """Format conversation history for evaluation prompt."""
    formatted = []
    for msg in history:
        if msg['role'] != 'system':  # Skip system messages
            content = msg.get('content', '')
            if content:
                formatted.append(f"{msg['role'].upper()}: {content}")
    return "\n".join(formatted)

def format_candidates(candidates: List[Dict]) -> str:
    """Format candidate responses for evaluation."""
    formatted = []
    for i, candidate in enumerate(candidates):
        content = candidate.get('content', '')
        tool_calls = candidate.get('tool_calls', [])
        
        formatted.append(f"CANDIDATE {i}:")
        formatted.append(f"Thought: {content}")
        
        if tool_calls:
            formatted.append("Tool Calls:")
            for tool in tool_calls:
                formatted.append(f"- {tool['function']['name']}: {tool['function']['arguments']}")
        formatted.append("---")
    
    return "\n".join(formatted)

def generate_single_candidate(
    task: str,
    conversation_history: List[Dict],
    api_key: str,
    model: str,
    api_url: str,
    tools: List[Dict],
    verbose: bool = False
) -> Dict:
    """Generate a single candidate thought."""
    temperature = 0.7 + (time.time() % 0.3)
    
    messages = conversation_history + [{
        'role': 'user',
        'content': 'Think about your next reasoning step to perform the CURRENT_TASK. Return just the next step.'
    }]

    if verbose:
        print(f"\n{Fore.CYAN}├── Generating candidate with temperature {temperature}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}├── Message count: {len(messages)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}├── Full message history:{Style.RESET_ALL}")
        for i, msg in enumerate(messages):
            # Truncate content for readability but show full role and type info
            content = msg.get('content', 'No content')[:100] + '...' if msg.get('content') else 'No content'
            print(f"{Fore.CYAN}│   [{i}] Role: {msg.get('role')}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}│       Content: {content}{Style.RESET_ALL}")
            if 'tool_calls' in msg:
                print(f"{Fore.CYAN}│       Has tool_calls: {bool(msg['tool_calls'])}{Style.RESET_ALL}")
            if 'tool_result' in msg:
                print(f"{Fore.CYAN}│       Has tool_result: True{Style.RESET_ALL}")
    
    try:
        response = send_message_to_api(
            task=task,
            messages=messages,
            api_key=api_key,
            model=model,
            api_url=api_url,
            tools=tools,
            temperature=temperature,
            max_tokens=500
        )
        
        if verbose:
            print(f"{Fore.CYAN}├── API Response:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}│   Status: Success{Style.RESET_ALL}")
            print(f"{Fore.CYAN}│   Full Response: {json.dumps(response, indent=2)}{Style.RESET_ALL}")
        
        return response
        
    except Exception as e:
        if verbose:
            print(f"{Fore.RED}├── API Error Details:{Style.RESET_ALL}")
            print(f"{Fore.RED}│   Error Type: {type(e)}{Style.RESET_ALL}")
            print(f"{Fore.RED}│   Error Message: {str(e)}{Style.RESET_ALL}")
            if hasattr(e, 'response'):
                print(f"{Fore.RED}│   Response Status: {e.response.status_code}{Style.RESET_ALL}")
                print(f"{Fore.RED}│   Response Body: {e.response.text}{Style.RESET_ALL}")
            print(f"{Fore.RED}│   Messages sent:{Style.RESET_ALL}")
            for msg in messages:
                print(f"{Fore.RED}│     {msg.get('role')}: {msg.get('content', '')[:100]}{Style.RESET_ALL}")
        raise

def generate_candidates(
    task: str,
    conversation_history: List[Dict],
    num_candidates: int,
    api_key: str,
    model: str,
    api_url: str,
    tools: List[Dict],
    verbose: bool = False
) -> List[Dict]:
    """Generate multiple candidates concurrently using ThreadPoolExecutor."""
    candidates = []

    if verbose:
        print(f"\n{Fore.CYAN}╭── Starting Beam Search Generation{Style.RESET_ALL}")
        print(f"{Fore.CYAN}├── Generating {num_candidates} candidates in parallel{Style.RESET_ALL}")
    
    with ThreadPoolExecutor(max_workers=num_candidates) as executor:
        futures = [
            executor.submit(
                generate_single_candidate,
                task,
                conversation_history,
                api_key,
                model,
                api_url,
                tools,
                verbose
            )
            for _ in range(num_candidates)
        ]
        
        # Handle failures for individual candidates
        for future in futures:
            try:
                result = future.result()
                candidates.append(result)
                if verbose:
                    print(f"{Fore.GREEN}├── Generated candidate successfully{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}├── Failed to generate candidate: {str(e)}{Style.RESET_ALL}")
                candidates.append({
                    'role': 'assistant',
                    'content': 'Failed to generate thought.',
                    'error': str(e)
                })
    
    if verbose:
        print(f"{Fore.CYAN}╰── Completed candidate generation{Style.RESET_ALL}")
    
    return candidates[:num_candidates]  # Ensure we return exactly num_candidates

def evaluate_candidates(
    candidates: List[Dict],
    task: str,
    conversation_history: List[Dict],
    api_key: str,
    model: str,
    api_url: str
) -> Tuple[int, str]:
    """Evaluate candidates and return best index and explanation."""
    evaluation_prompt = f"""Given the following task and conversation history, evaluate these candidate thoughts.
    
Task: {task}

Previous Steps:
{format_conversation_history(conversation_history)}

Candidate Thoughts:
{format_candidates(candidates)}

For each candidate, evaluate:
1. Relevance to the task (0-10)
2. Logical progression from previous steps (0-10)
3. Clarity and specificity (0-10)
4. Effectiveness of any tool usage (0-10, N/A if no tools used)

Return the index of the best candidate (0-based) and a brief explanation why.
Format: <index>|<explanation>
"""
    
    messages = [{'role': 'user', 'content': evaluation_prompt}]
    response = send_message_to_api(
        task=task,
        messages=messages,
        api_key=api_key,
        model=model,
        api_url=api_url,
        tools=[],  # No tools needed for evaluation
        temperature=0.3  # Lower temperature for more consistent evaluation
    )
    
    # Parse response format: "index|explanation"
    try:
        index_str, explanation = response['content'].split('|', 1)
        return int(index_str), explanation.strip()
    except (ValueError, TypeError) as e:
        # Fallback to first candidate if parsing fails
        return 0, f"Evaluation parsing failed: {str(e)}"

def requires_python_execution(candidate: Dict) -> bool:
    """Check if a candidate requires Python code execution."""
    tool_calls = candidate.get('tool_calls', [])
    return any(
        tool['function']['name'] == 'python' 
        for tool in tool_calls
    )

def get_python_code(candidate: Dict) -> Optional[str]:
    """Extract Python code from a candidate's tool calls."""
    tool_calls = candidate.get('tool_calls', [])
    for tool in tool_calls:
        if tool['function']['name'] == 'python':
            try:
                args = json.loads(tool['function']['arguments'])
                return args.get('code')
            except:
                pass
    return None

def process_beam_search_step(
    task: str,
    conversation_history: List[Dict],
    num_candidates: int,
    current_branch: str,
    api_key: str,
    model: str,
    api_url: str,
    tools: List[Dict],
    sandbox: Sandbox,
    verbose: bool = False
) -> Tuple[Dict, str]:
    """Process one step of beam search with resource management."""
    try:
        if verbose:
            print(f"\n{Fore.MAGENTA}╭── Starting Beam Search Step{Style.RESET_ALL}")
        
        candidates = generate_candidates(
            task=task,
            conversation_history=conversation_history,
            num_candidates=num_candidates,
            api_key=api_key,
            model=model,
            api_url=api_url,
            tools=tools,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n{Fore.MAGENTA}├── Processing {len(candidates)} candidates{Style.RESET_ALL}")
            for i, candidate in enumerate(candidates):
                print(f"{Fore.YELLOW}│   Candidate {i}:{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}│   Content: {Style.RESET_ALL}{candidate.get('content', '')}")
        
        # Track branches for candidates that need Python execution
        branch_mapping = {}
        
        # Process candidates that require Python execution
        for i, candidate in enumerate(candidates):
            if requires_python_execution(candidate):
                if verbose:
                    print(f"\n{Fore.BLUE}├── Processing Python execution for candidate {i}{Style.RESET_ALL}")
                
                new_branch = create_branch(current_branch)
                branch_mapping[i] = new_branch
                
                if verbose:
                    print(f"{Fore.BLUE}│   Created branch: {new_branch}{Style.RESET_ALL}")
                
                code = get_python_code(candidate)
                if code:
                    try:
                        if verbose:
                            print(f"{Fore.BLUE}│   Executing code in new sandbox{Style.RESET_ALL}")
                        
                        new_sandbox = Sandbox()
                        interpreter_branches[new_branch]['sandbox'] = new_sandbox
                        execute_branch_history(new_branch, new_sandbox)
                        
                        record_command(new_branch, code)
                        result = execute_tool(
                            tool_name="python",
                            parameters={"code": code},
                            task=task,
                            sandbox=new_sandbox
                        )
                        candidate['tool_result'] = result
                        
                        if verbose:
                            print(f"{Fore.GREEN}│   Code execution successful{Style.RESET_ALL}")
                    except Exception as e:
                        if verbose:
                            print(f"{Fore.RED}│   Code execution failed: {str(e)}{Style.RESET_ALL}")
                        candidate['tool_result'] = f"Error: {str(e)}"
                        cleanup_sandbox(new_sandbox)
        
        if verbose:
            print(f"\n{Fore.MAGENTA}├── Evaluating candidates{Style.RESET_ALL}")
        
        best_index, explanation = evaluate_candidates(
            candidates=candidates,
            task=task,
            conversation_history=conversation_history,
            api_key=api_key,
            model=model,
            api_url=api_url
        )
        
        if verbose:
            print(f"{Fore.GREEN}├── Selected candidate {best_index}: {explanation}{Style.RESET_ALL}")
        
        best_candidate = candidates[best_index]
        new_branch = branch_mapping.get(best_index, current_branch)
        
        if verbose:
            print(f"{Fore.MAGENTA}├── Cleaning up unused branches{Style.RESET_ALL}")
        
        cleanup_unused_branches(new_branch)
        
        if verbose:
            print(f"{Fore.MAGENTA}╰── Beam Search Step Complete{Style.RESET_ALL}")
        
        return best_candidate, new_branch
        
    except Exception as e:
        if verbose:
            print(f"{Fore.RED}╰── Beam Search Step Failed: {str(e)}{Style.RESET_ALL}")
        cleanup_unused_branches(current_branch)
        raise e