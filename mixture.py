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
    chain_store_api_key: Optional[str] = None,
    max_reasoning_steps: Optional[int] = None,
    wolfram_app_id: Optional[str] = None,
    default_temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 500,
    image: Optional[str] = None,
    output_tools: Optional[List[Dict]] = None,
    reflection_mode: bool = False
) -> Tuple[Dict[str, str], str, List[Dict], List[Dict]]:
    """
    Run a single agent with the given configuration.
    
    Args:
        task: The task to complete
        agent_config: Dictionary containing 'model', 'api_key', and 'api_url'
        verbose: Whether to show detailed output
        chain_store_api_key: API key for chain store if using
        max_reasoning_steps: Maximum number of reasoning steps for this agent
        wolfram_app_id: Wolfram Alpha app ID if using
        default_temperature: Default temperature for the model if using
        top_p: Top p for the model if using
        max_tokens: Maximum number of tokens for the model if using
        image: Optional image to pass to the model if using
        output_tools: Optional list of output tools for the model if using
        reflection_mode: Whether to enable reflection mode for this agent
    Returns:
        Tuple of (agent_config, final_response, conversation_history, thinking_tools, output_tools)
    """
    # Reinitialize colorama for this process
    init(autoreset=True)
    
    if verbose:
        print(f"\n{Fore.CYAN}Running agent with model: {Style.RESET_ALL}{agent_config['model']}")
        if max_reasoning_steps:
            print(f"{Fore.CYAN}Max steps: {Style.RESET_ALL}{max_reasoning_steps}")
        print(f"{Fore.CYAN}Temperature: {Style.RESET_ALL}{agent_config.get('temperature', default_temperature)}")
    
    if verbose and reflection_mode:
        print(f"{Fore.CYAN}Reflection mode: {Style.RESET_ALL}Enabled")
    
    response, history, thinking_tools, output_tools = complete_reasoning_task(
        task=task,
        api_key=agent_config['api_key'],
        model=agent_config['model'],
        api_url=agent_config['api_url'],
        verbose=verbose,
        chain_store_api_key=chain_store_api_key,    
        max_reasoning_steps=max_reasoning_steps,
        wolfram_app_id=wolfram_app_id,
        temperature=agent_config.get('temperature', default_temperature),
        top_p=top_p,
        max_tokens=max_tokens,
        image=image,
        output_tools=output_tools,
        reflection_mode=reflection_mode
    )

    # Remove example chains from conversation history
    bottom_system_message_index = next((i for i, msg in enumerate(reversed(history)) if msg.get('role') == 'system'), None)
    if bottom_system_message_index is not None:
        history = history[-bottom_system_message_index:]
    
    return agent_config, response, history, thinking_tools, output_tools

def format_agent_results(
    agent_results: List[Tuple[Dict[str, str], str, List[Dict], List[Dict]]]
) -> str:
    """Format the results from multiple agents into a prompt for the coordinator."""
    formatted_results = "Here are the responses from different AI models:\n\n"
    
    for i, (agent_config, response, history, thinking_tools, output_tools) in enumerate(agent_results, 1):
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
    max_workers: Optional[int] = None,
    max_reasoning_steps: Optional[int] = None,
    wolfram_app_id: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 500,
    image: Optional[str] = None,
    output_tools: Optional[List[Dict]] = None,
    reflection_mode: bool = False
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
                chain_store_api_key,
                max_reasoning_steps,
                wolfram_app_id,
                temperature,
                top_p,
                max_tokens,
                image,
                output_tools,
                reflection_mode
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
    return_reasoning: bool = False,
    max_reasoning_steps: Optional[int] = None,
    coordinator_max_steps: Optional[int] = None,
    wolfram_app_id: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 500,
    image: Optional[str] = None,
    output_tools: Optional[List[Dict]] = None,
    reflection_mode: bool = False
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
        max_reasoning_steps: Maximum steps for each agent
        coordinator_max_steps: Maximum steps for the coordinator (can be different from agents)
        wolfram_app_id: Wolfram Alpha app ID if using
        temperature: Default temperature for the model if using
        top_p: Top p for the model if using
        max_tokens: Maximum number of tokens for the model if using
        image: Optional image to pass to the model if using
        output_tools: Optional list of output tools for the model if using
        reflection_mode: Whether to enable reflection mode for all agents
    """
    # Reinitialize colorama for the main process
    init(autoreset=True)
    
    if verbose:
        print(f"\n{Fore.MAGENTA}Starting Ensemble for task:{Style.RESET_ALL}")
        print(f"{task}\n")
        print(f"{Fore.MAGENTA}Using {len(agents)} agents in parallel{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Default temperature: {temperature}{Style.RESET_ALL}")
        for agent in agents:
            if 'temperature' in agent:
                print(f"{Fore.MAGENTA}Temperature for {agent['model']}: {agent['temperature']}{Style.RESET_ALL}")
    
    if verbose and reflection_mode:
        print(f"{Fore.MAGENTA}Reflection mode: {Style.RESET_ALL}Enabled for all agents")
    
    # Run all agents in parallel with max steps
    agent_results = run_agents_parallel(
        task,
        agents,
        verbose,
        chain_store_api_key,
        max_workers,
        max_reasoning_steps,
        wolfram_app_id,
        temperature,
        top_p,
        max_tokens,
        image,
        output_tools,
        reflection_mode
    )
    
    # Format results for coordinator
    formatted_results = format_agent_results(agent_results)
    
    # Create coordinator prompt
    coordinator_task = f"""You are a coordinator model tasked with analyzing multiple AI responses to the following question:

Question: {task}

<Agent Responses>
{formatted_results}
</Agent Responses>

Please analyze all responses and their reasoning steps carefully. Consider:
1. The logical soundness of each approach
2. The thoroughness of the reasoning
3. The correctness of calculations and tool usage
4. The clarity and completeness of the final response

Based on your analysis, synthesize these responses into a single, high-quality response to the question. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the question. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Also remember that the user is only going to see your final answer, so make sure it's complete and self-contained, and actually answers the question."""

    # Get coordinator's response
    if verbose:
        print(f"\n{Fore.CYAN}Running coordinator model: {Style.RESET_ALL}{coordinator['model']}")
    
    coordinator_response, _, _, _ = complete_reasoning_task(
        task=coordinator_task,
        api_key=coordinator['api_key'],
        model=coordinator['model'],
        api_url=coordinator['api_url'],
        verbose=verbose,
        chain_store_api_key=None,
        max_reasoning_steps=coordinator_max_steps,
        wolfram_app_id=wolfram_app_id,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        image=image,
        output_tools=output_tools,
        reflection_mode=reflection_mode
    )
    
    if return_reasoning:
        return coordinator_response, agent_results
    return coordinator_response

# Alias for backward compatibility
run_mixture_of_agents = ensemble