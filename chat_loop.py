import requests
import json
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def call_reason_api(
    task: str, 
    previous_chains: Optional[List[List[Dict]]] = None,
    api_key: Optional[str] = None,
    model: str = "openai/gpt-4o-mini",
    api_url: str = "https://openrouter.ai/api/v1/chat/completions",
    verbose: bool = True
) -> tuple[Dict, List[Dict]]:
    """Call the reasoning API and return the response and chain."""
    
    url = "http://localhost:5050/reason"
    
    payload = {
        "task": task,
        "api_key": "[redacted]",
        "model": "openai/gpt-4o-mini",
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "verbose": verbose
    }

    if previous_chains:
        payload["previous_chains"] = previous_chains

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["response"], data["reasoning_chain"]
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        raise

def main():
    print("Welcome to the reasoning chat loop!")
    print("Type 'exit' to quit, 'clear' to start a new conversation.")
    print("Enter your message:")

    conversation_chains = []
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.lower() == 'clear':
            conversation_chains = []
            print("\nConversation cleared. Starting fresh!")
            continue
            
        try:
            # Call API with previous conversation chains
            response, chain = call_reason_api(
                task=user_input,
                previous_chains=conversation_chains
            )
            
            # Print the response
            if isinstance(response, dict):
                if response.get('content'):
                    print("\nAssistant:", response['content'])
                if response.get('tool_calls'):
                    print("\nTool Calls:", json.dumps(response['tool_calls'], indent=2))
            else:
                print("\nAssistant:", response)
            
            # Add this chain to our conversation history
            conversation_chains.append(chain)
            
            # Print conversation stats
            print(f"\n(Conversation history: {len(conversation_chains)} chains)")
            
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main() 