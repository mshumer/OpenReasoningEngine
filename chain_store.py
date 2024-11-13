import json
import os
import requests
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

def init_store(store_file: str = "successful_chains.json") -> None:
    """Initialize the chain store if it doesn't exist."""
    if not os.path.exists(store_file):
        with open(store_file, 'w') as f:
            json.dump({"chains": []}, f)

def get_embedding(text: str, cohere_api_key: str, input_type: str = "search_document") -> Optional[List[float]]:
    """Get embeddings from Cohere API."""
    try:
        response = requests.post(
            "https://api.cohere.ai/v1/embed",
            headers={
                "Authorization": f"Bearer {cohere_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "texts": [text],
                "model": "embed-english-v3.0",
                "input_type": input_type,
                "embedding_type": "float"
            }
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def save_successful_chain(
    task: str,
    conversation_history: List[Dict],
    final_response: str,
    cohere_api_key: str,
    tools: List[Dict],
    store_file: str = "successful_chains.json"
) -> bool:
    """Save a successful chain to the store."""
    try:
        # Get embedding for the task
        embedding = get_embedding(task, cohere_api_key)
        if not embedding:
            return False
        
        # Initialize store if it doesn't exist
        if not os.path.exists(store_file):
            store = {"chains": []}
        else:
            try:
                with open(store_file, 'r') as f:
                    store = json.load(f)
            except json.JSONDecodeError:
                # If file exists but is invalid JSON, start fresh
                store = {"chains": []}
        
        # Process conversation history to redact long tool responses
        processed_history = []
        for msg in conversation_history:
            if msg['role'] == 'tool' and len(msg['content']) > 1500:
                msg = msg.copy()  # Create a copy to avoid modifying the original
                msg['content'] = "[redacted for token savings]"
            processed_history.append(msg)
        
        # Add new chain
        chain = {
            "task": task,
            "embedding": embedding,
            "conversation_history": processed_history,
            "final_response": final_response,
            "tools": tools,
            "timestamp": datetime.now().isoformat()
        }
        store["chains"].append(chain)
        
        # Save updated store
        with open(store_file, 'w') as f:
            json.dump(store, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving chain: {str(e)}")  # More detailed error message
        return False

def get_similar_chains(
    task: str,
    cohere_api_key: str,
    n: int = 3,
    store_file: str = "successful_chains.json"
) -> List[Dict]:
    """Get n most similar chains for a given task."""
    try:
        # Get embedding for the query task
        query_embedding = get_embedding(task, cohere_api_key, input_type="search_query")
        if not query_embedding:
            return []
        
        # Load chains
        with open(store_file, 'r') as f:
            store = json.load(f)
        
        # Calculate similarities
        similarities = []
        for chain in store["chains"]:
            similarity = cosine_similarity(query_embedding, chain["embedding"])
            similarities.append((similarity, chain))
        
        # Sort by similarity and get top n
        similarities.sort(reverse=True, key=lambda x: x[0])
        result = [chain for _, chain in similarities[:n]]
        return result
        
    except Exception as e:
        return []

def prepare_examples_messages(similar_chains: List[Dict], current_tools: List[Dict]) -> List[Dict]:
    """
    Prepare example chains as messages for the prompt.
    Now includes information about available tools.
    """
    if not similar_chains:
        return []
    
    messages = []
    for chain in similar_chains:
        # Get the tool names for both current and historical tools
        current_tool_names = {t['function']['name'] for t in current_tools}
        historical_tool_names = {t['function']['name'] for t in chain.get('tools', [])}
        
        # Create tool availability message
        tool_message = "Available tools in this example:"
        for tool_name in historical_tool_names:
            status = "✓" if tool_name in current_tool_names else "✗"
            tool_message += f"\n- {tool_name} {status}"
        
        # Add system message with the example task and tool information
        messages.append({
            "role": "system",
            "content": (
                "<TASK>\n"
                f"{chain['task']}\n\n"
                f"{tool_message}\n\n"
                "<INSTRUCTIONS>\n"
                "Slow down your thinking by breaking complex questions into multiple reasoning steps.\n"
                "Each individual reasoning step should be brief.\n"
                "Return <DONE> after the last step."
            )
        })
        
        # Add the conversation history
        messages.extend(chain["conversation_history"])
    
    # For each message, replace any instance of the substring TASK with EXAMPLE_TASK
    for i, msg in enumerate(messages):
        if 'TASK' in msg['content']:
            messages[i]['content'] = msg['content'].replace('CURRENT_TASK', 'EXAMPLE_TASK')
            messages[i]['content'] = msg['content'].replace('TASK', 'EXAMPLE_TASK')
            messages[i]['content'] = messages[i]['content'].replace('EXAMPLE_EXAMPLE_TASK', 'EXAMPLE_TASK')
    
    return messages