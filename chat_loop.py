import requests
import json
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def call_chat_completions(
    messages: List[Dict],
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    model_url: str = "https://openrouter.ai/api/v1/chat/completions",
    verbose: bool = True
) -> Dict:
    """Call the chat completions API with OpenAI-compatible format."""
    
    url = "http://localhost:5050/v1/chat/completions"
    
    payload = {
        "messages": messages,
        "model": {
            "name": 'gpt-4o-mini',
            "api_key": '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',
            "url": 'https://openrouter.ai/api/v1/chat/completions'
        },
        "config": {
            "verbose": verbose,
            "tools": {
                "python": {"enabled": True},
                "web_search": {"enabled": True},
                "wolfram": {
                    "enabled": bool(os.getenv("WOLFRAM_APP_ID")),
                    "app_id": os.getenv("WOLFRAM_APP_ID")
                }
            },
            "chain_store": {
                "api_key": os.getenv("CHAIN_STORE_API_KEY")
            }
        }
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response text: {e.response.text}")
        raise

def format_reasoning_chain(chain_message: Dict) -> None:
    """Pretty print a reasoning chain message."""
    # First check if we have a valid chain message
    if not isinstance(chain_message.get('content'), dict):
        print("\nAssistant:", chain_message.get('content'))
        return
        
    if not chain_message.get('content', {}).get('chain'):
        print("\nAssistant:", chain_message.get('content', {}).get('response', ''))
        return
    
    print("\nReasoning Steps:")
    print("─" * 40)
    
    try:
        for step in chain_message['content']['chain']:
            if not isinstance(step, dict):
                continue
                
            if step.get('role') == 'assistant':
                print(f"\n🤔 Assistant: {step.get('content', '')}")
            elif step.get('role') == 'tool':
                print(f"🔧 Tool Output: {step.get('content', '')}")
            
            if step.get('tool_calls'):
                for tool_call in step['tool_calls']:
                    if isinstance(tool_call, dict):
                        print(f"  └─ Used: {tool_call.get('tool', 'unknown')}")
                        print(f"     Args: {tool_call.get('input', {})}")
                        print(f"     Result: {tool_call.get('output', '')}")
    
        print("─" * 40)
        print(f"\n📝 Final Response: {chain_message.get('content', {}).get('response', '')}\n")
    except Exception as e:
        print(f"\nError formatting reasoning chain: {e}")
        print("Raw message:", json.dumps(chain_message, indent=2))

def main():
    print("Welcome to the reasoning chat loop!")
    print("Type 'exit' to quit, 'clear' to start a new conversation.")
    print("Enter your message:")

    conversation_messages = []
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.lower() == 'clear':
            conversation_messages = []
            print("\nConversation cleared. Starting fresh!")
            continue
            
        try:
            # Add user message to conversation
            conversation_messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Call API with conversation history
            response = call_chat_completions(
                messages=conversation_messages,
                verbose=True
            )
            
            # Extract and format the assistant's message
            if response.get('choices'):
                assistant_message = response['choices'][0]['message']
                conversation_messages.append(assistant_message)
                
                # Pretty print the reasoning chain
                if assistant_message['role'] == 'reasoning_chain':
                    format_reasoning_chain(assistant_message)
                else:
                    print("\nAssistant:", assistant_message['content'])
                
                # Print conversation stats
                print(f"\n(Conversation history: {len(conversation_messages)} messages)")
            else:
                print("\nError: No response choices found")
            
        except Exception as e:
            print(f"\nError: {e}")
            # Optionally remove the failed message from history
            if conversation_messages:
                conversation_messages.pop()

if __name__ == "__main__":
    main() 