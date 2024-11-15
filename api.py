from flask import Flask, request, jsonify
from engine import complete_reasoning_task
from mixture import ensemble
import traceback
from typing import Dict, Any, Optional, List, Union
import json
import time

app = Flask(__name__)

class APIError(Exception):
    def __init__(self, message: str, type: str, code: str, param: Optional[str] = None, status_code: int = 400):
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        self.status_code = status_code
        super().__init__(self.message)

def validate_config(config: Optional[Dict[str, Any]]) -> None:
    """Validate the configuration object if present."""
    if not config:
        return
    
    # Validate tool configurations
    tools = config.get('tools', {})
    if tools.get('wolfram', {}).get('enabled', False):
        if not tools.get('wolfram', {}).get('app_id'):
            raise APIError(
                message="Wolfram app_id required when wolfram tool is enabled",
                type="invalid_request_error",
                code="tool_config_error",
                param="tools.wolfram.app_id"
            )

    # Validate reasoning configuration
    reasoning = config.get('reasoning', {})
    if 'max_steps' in reasoning and (
        not isinstance(reasoning['max_steps'], int) or 
        reasoning['max_steps'] <= 0
    ):
        raise APIError(
            message="max_steps must be a positive integer",
            type="invalid_request_error",
            code="invalid_config",
            param="reasoning.max_steps"
        )

def validate_model(model: Optional[Dict[str, Any]]) -> None:
    """Validate the model configuration."""
    if not model:
        raise APIError(
            message="Model configuration is required",
            type="invalid_request_error",
            code="invalid_request",
            param="model"
        )
    
    required_fields = ['name', 'api_key', 'url']
    missing_fields = [field for field in required_fields if not model.get(field)]
    
    if missing_fields:
        raise APIError(
            message=f"Model configuration missing required fields: {', '.join(missing_fields)}",
            type="invalid_request_error",
            code="invalid_request",
            param="model"
        )

def convert_messages_to_previous_chains(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Convert API message format to engine's previous_chains format."""
    previous_chains = []
    current_chain = []
    
    for msg in messages:
        if msg['role'] == 'reasoning_chain':
            # Add the chain from the reasoning_chain message
            if 'chain' in msg.get('content', {}):
                chain = []
                tool_call_id = None  # Track the current tool call ID
                
                for step in msg['content']['chain']:
                    step_msg = {
                        'role': step['role'],
                        'content': step['content']
                    }
                    
                    # Handle tool calls in assistant messages
                    if step['role'] == 'assistant' and step.get('tool_calls'):
                        step_msg['tool_calls'] = step['tool_calls']
                        # Get the tool call ID for the next tool response
                        if step['tool_calls']:
                            tool_call_id = step['tool_calls'][0]['id']
                    
                    # Add tool_call_id for tool messages
                    if step['role'] == 'tool' and tool_call_id:
                        step_msg['tool_call_id'] = tool_call_id
                        tool_call_id = None  # Reset for next tool call
                    
                    chain.append(step_msg)
                
                previous_chains.append(chain)
        else:
            # Regular message
            current_chain.append({
                'role': msg['role'],
                'content': msg['content']
            })
    
    # Add the current chain if not empty
    if current_chain:
        previous_chains.append(current_chain)
    
    return previous_chains

def convert_response_to_message(
    response: Dict[str, Any],
    history: List[Dict[str, Any]],
    thinking_tools: List[Dict[str, Any]],
    output_tools: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Convert engine response to API message format."""
    chain = []
    tool_call_id = None
    
    for idx, msg in enumerate(history):
        if msg['role'] not in ['assistant', 'tool']:
            continue
            
        step = {
            'step': idx + 1,
            'role': msg['role'],
            'content': msg['content']
        }
        
        # Handle tool calls in assistant messages
        if msg['role'] == 'assistant' and msg.get('tool_calls'):
            step['tool_calls'] = msg['tool_calls']
            if msg['tool_calls']:
                tool_call_id = msg['tool_calls'][0]['id']
        
        # Add tool_call_id for tool messages
        if msg['role'] == 'tool' and tool_call_id:
            step['tool_call_id'] = tool_call_id
            tool_call_id = None
            
        chain.append(step)
    
    return {
        'role': 'reasoning_chain',
        'content': {
            'response': response.get('content'),
            'chain': chain
        }
    }

@app.errorhandler(APIError)
def handle_api_error(error):
    response = jsonify({
        'error': {
            'message': error.message,
            'type': error.type,
            'code': error.code
        }
    })
    if error.param:
        response.json['error']['param'] = error.param
    response.status_code = error.status_code
    return response

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    OpenAI-compatible chat completions endpoint with reasoning capabilities.
    """
    try:
        data = request.get_json()
        
        # Validate basic request structure
        if not isinstance(data.get('messages', []), list):
            raise APIError(
                message="messages must be an array",
                type="invalid_request_error",
                code="invalid_request"
            )
        
        if not data.get('messages'):
            raise APIError(
                message="messages array cannot be empty",
                type="invalid_request_error",
                code="invalid_request"
            )

        # Validate model configuration first
        model = data.get('model')
        validate_model(model)
        
        # Then validate other config
        config = data.get('config', {})
        validate_config(config)
        
        # Convert messages to previous_chains format
        previous_chains = convert_messages_to_previous_chains(data['messages'])
        
        # Get the current task from the last user message
        task = next(
            (msg['content'] if isinstance(msg['content'], str)
             else msg['content'][0]['content'] if isinstance(msg['content'], list) 
             else None
             for msg in reversed(data['messages'])
             if msg['role'] == 'user'),
            None
        )

        # If there's an image url in the last user message, save it as image_url
        final_user_message = data['messages'][-1]['content']
        # Image is the item where 'type' is 'image_url'
        try:
            image = next((item for item in final_user_message if item.get('type') == 'image_url'), None)
        except:
            image = None
        if image:
            image_url = image['image_url']['url']
        else:
            image_url = None

        # Now remove the final user message from the previous_chains array, as we have the task and image_url
        previous_chains = previous_chains[:-1]

        if not task:
            raise APIError(
                message="No user message found",
                type="invalid_request_error",
                code="invalid_request"
            )

        # Extract parameters
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 1.0)
        max_tokens = data.get('max_tokens', 500)
        
        # Extract config parameters
        reasoning_config = config.get('reasoning', {})
        tools_config = config.get('tools', {})
        
        print(f"Calling complete_reasoning_task with model config: {model}")  # Debug print
        
        # Run reasoning task
        response, history, thinking_tools, output_tools = complete_reasoning_task(
            task=task,
            api_key=model['api_key'],
            model=model['name'],
            api_url=model['url'],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            verbose=config.get('verbose', False),
            chain_store_api_key=config.get('chain_store', {}).get('api_key'),
            wolfram_app_id=tools_config.get('wolfram', {}).get('app_id'),
            max_reasoning_steps=reasoning_config.get('max_steps'),
            image=image_url,
            output_tools=config.get('output_tools'),
            reflection_mode=reasoning_config.get('reflection_mode', False),
            previous_chains=previous_chains,
            use_planning=reasoning_config.get('use_planning', True)
        )
        
        # Convert response to API format
        message = convert_response_to_message(response, history, thinking_tools, output_tools)
        
        # Return OpenAI-compatible response
        return jsonify({
            'id': f'chatcmpl-{id(response)}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model['name'],
            'choices': [{
                'index': 0,
                'message': message,
                'finish_reason': 'stop'  # We'll need to implement proper finish reason logic
            }],
            'usage': {
                'prompt_tokens': 0,  # We'll need to implement token counting
                'completion_tokens': 0,
                'total_tokens': 0
            }
        })

    except APIError as e:
        raise e
    except Exception as e:
        raise APIError(
            message=str(e),
            type="api_error",
            code="internal_error",
            status_code=500
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050) 