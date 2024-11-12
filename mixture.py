from typing import List, Dict, Optional, Tuple, Any, Union
from engine import complete_reasoning_task
import json
from colorama import init, Fore, Style
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize colorama for cross-platform colored output
init()

def run_agent(
    task: str,
    agent_config: Dict[str, str],
    verbose: bool = False,
    chain_store_api_key: Optional[str] = None
) -> Tuple[Dict[str, str], str, List[Dict], List[Dict]]:
    """
    Run a single agent with the given configuration.
    
    Args:
        task: The task to complete
        agent_config: Dictionary containing 'model', 'api_key', and 'api_url'
        verbose: Whether to show detailed output
        chain_store_api_key: API key for chain store if using
    
    Returns:
        Tuple of (agent_config, final_response, conversation_history, tools)
    """
    if verbose:
        print(f"\n{Fore.CYAN}Running agent with model: {Style.RESET_ALL}{agent_config['model']}")
    
    response, history, tools = complete_reasoning_task(
        task=task,
        api_key=agent_config['api_key'],
        model=agent_config['model'],
        api_url=agent_config['api_url'],
        verbose=verbose,
        chain_store_api_key=chain_store_api_key
    )

    # Remove example chains from conversation history by removing everything prior to the bottom-most system message
    bottom_system_message_index = next((i for i, msg in enumerate(reversed(history)) if msg.get('role') == 'system'), None)
    if bottom_system_message_index is not None:
        history = history[-bottom_system_message_index:]
    
    return agent_config, response, history, tools

def format_agent_results(
    agent_results: List[Tuple[Dict[str, str], str, List[Dict], List[Dict]]]
) -> str:
    """Format the results from multiple agents into a prompt for the coordinator."""
    formatted_results = "Here are the responses from different AI models:\n\n"
    
    for i, (agent_config, response, history, _) in enumerate(agent_results, 1):
        formatted_results += f"Model {i} ({agent_config['model']}):\n"
        formatted_results += "Reasoning steps:\n"
        
        # Extract reasoning steps from history
        for msg in history:
            if msg['role'] == 'assistant':
                if msg.get('content'):
                    formatted_results += f"- {msg['content']}\n"
            elif msg['role'] == 'tool':
                formatted_results += f"  Tool result: {msg['content']}\n"
        
        formatted_results += f"\nFinal response:\n{response}\n\n"
        formatted_results += "─" * 50 + "\n\n"
    
    return formatted_results

def run_agents_parallel(
    task: str,
    agents: List[Dict[str, str]],
    verbose: bool = False,
    chain_store_api_key: Optional[str] = None,
    max_workers: Optional[int] = None
) -> List[Tuple[Dict[str, str], str, List[Dict], List[Dict]]]:
    """Run multiple agents in parallel."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_agent = {
            executor.submit(
                run_agent, 
                task, 
                agent,
                verbose,
                chain_store_api_key
            ): agent for agent in agents
        }
        
        # Collect results as they complete
        results = []
        for future in as_completed(future_to_agent):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                if verbose:
                    agent = future_to_agent[future]
                    print(f"\n{Fore.RED}Error with model {agent['model']}: {str(e)}{Style.RESET_ALL}")
        
        return results

def ensemble(
    task: str,
    agents: List[Dict[str, str]],
    coordinator: Dict[str, str],
    verbose: bool = False,
    chain_store_api_key: Optional[str] = None,
    max_workers: Optional[int] = None,
    return_reasoning: bool = False
) -> Union[str, Tuple[str, List[Tuple[Dict[str, str], str, List[Dict], List[Dict]]]]]:
    """
    Run multiple agents in parallel and coordinate their responses.
    
    Args:
        task: The task to complete
        agents: List of dictionaries, each containing 'model', 'api_key', and 'api_url'
        coordinator: Dictionary containing 'model', 'api_key', and 'api_url' for the coordinating model
        verbose: Whether to show detailed output
        chain_store_api_key: API key for chain store if using
        max_workers: Maximum number of parallel workers
        return_reasoning: Whether to return the full reasoning chains
    
    Returns:
        If return_reasoning is False: Final coordinated response (str)
        If return_reasoning is True: Tuple of (final response, list of agent results)
    """
    if verbose:
        print(f"\n{Fore.MAGENTA}Starting Ensemble for task:{Style.RESET_ALL}")
        print(f"{task}\n")
        print(f"{Fore.MAGENTA}Using {len(agents)} agents in parallel{Style.RESET_ALL}")
    
    # Run all agents in parallel
    agent_results = run_agents_parallel(
        task,
        agents,
        verbose,
        chain_store_api_key,
        max_workers
    )
    
    # Format results for coordinator
    formatted_results = format_agent_results(agent_results)
    
    # Create coordinator prompt
    coordinator_task = f"""You are a coordinator model tasked with analyzing multiple AI responses to the following task:

Question: {task}

<Agent Responses>
{formatted_results}
</Agent Responses>

Please analyze all responses and their reasoning steps carefully. Consider:
1. The logical soundness of each approach
2. The thoroughness of the reasoning
3. The correctness of calculations and tool usage
4. The clarity and completeness of the final response

Based on your analysis, synthesize these responses into a single, high-quality response to the question. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the question. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability."""

    # Get coordinator's response
    if verbose:
        print(f"\n{Fore.CYAN}Running coordinator model: {Style.RESET_ALL}{coordinator['model']}")
    
    coordinator_response, _, _ = complete_reasoning_task(
        task=coordinator_task,
        api_key=coordinator['api_key'],
        model=coordinator['model'],
        api_url=coordinator['api_url'],
        verbose=verbose,
        chain_store_api_key=None  # Don't use chain store for coordinator
    )
    
    if return_reasoning:
        return coordinator_response, agent_results
    return coordinator_response

# Alias for backward compatibility
run_mixture_of_agents = ensemble