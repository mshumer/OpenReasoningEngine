from flask import Flask, request, jsonify
from engine import complete_reasoning_task
from mixture import ensemble
import traceback

app = Flask(__name__)

@app.route('/reason', methods=['POST'])
def reason():
    """
    Single model reasoning endpoint.
    
    Expected JSON payload:
    {
        "task": "The task description",
        "model_config": {
            "api_key": "your-api-key",
            "model": "model-name",
            "api_url": "api-endpoint",
            "temperature": 0.7,            # optional
            "top_p": 1.0,                 # optional
            "max_tokens": 500             # optional
        },
        "verbose": false,             # optional
        "chain_store_api_key": "key", # optional
        "wolfram_app_id": "key",      # optional
        "max_reasoning_steps": 10,    # optional
        "image": "image-url or base64" # optional
        "output_tools": [
            {
                "type": "tool-type",
                "name": "tool-name",
                "description": "tool-description"
            }
        ] # optional
        "reflection_mode": false,    # optional: enable reflection mode
    }
    """
    try:
        data = request.get_json()
        
        # Required parameters
        task = data.get('task')
        model_config = data.get('model_config')
    
        if not all([task, model_config, model_config['api_key'], model_config['model'], model_config['api_url']]):
            return jsonify({
                'error': 'Missing required parameters. Need: task, model_config, model_config["api_key"], model_config["model"], model_config["api_url"]'
            }), 400
                
        # Optional parameters
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 1.0)
        max_tokens = data.get('max_tokens', 500)
        verbose = data.get('verbose', False)
        chain_store_api_key = data.get('chain_store_api_key')
        wolfram_app_id = data.get('wolfram_app_id')
        max_reasoning_steps = data.get('max_reasoning_steps')
        image = data.get('image')
        output_tools = data.get('output_tools')
        reflection_mode = data.get('reflection_mode', False)

        # Run reasoning
        response, history, thinking_tools, output_tools = complete_reasoning_task(
            task=task,
            model_config=model_config,
            max_tokens=max_tokens,
            verbose=verbose,
            chain_store_api_key=chain_store_api_key,
            wolfram_app_id=wolfram_app_id,
            max_reasoning_steps=max_reasoning_steps,
            image=image,
            output_tools=output_tools,
            reflection_mode=reflection_mode
        )
                
        return jsonify({
            'response': response,
            'conversation_history': history,
            'thinking_tools': thinking_tools,
            'output_tools': output_tools
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/ensemble', methods=['POST'])
def run_ensemble():
    """
    Ensemble reasoning endpoint.
    
    Expected JSON payload:
    {
        "task": "The task description",
        "agents": [
            {
                "model": "model-name-1",
                "api_key": "key-1",
                "api_url": "url-1",
                "temperature": "temperature-1",
            },
            {
                "model": "model-name-2",
                "api_key": "key-2",
                "api_url": "url-2",
                "temperature": "temperature-2"
            }
        ],
        "coordinator": {
            "model": "model-name",
            "api_key": "key",
            "api_url": "url",
            "temperature": "temperature"
        },
        "verbose": false,             # optional
        "chain_store_api_key": "key", # optional
        "max_workers": 3,             # optional
        "return_reasoning": false,    # optional
        "max_reasoning_steps": 10,    # optional: max steps per agent
        "coordinator_max_steps": 5,   # optional: max steps for coordinator
        "wolfram_app_id": "key",      # optional
        "temperature": 0.7,           # optional
        "top_p": 1.0,                # optional
        "max_tokens": 500            # optional
        "reflection_mode": false,    # optional: enable reflection mode for all agents
    }
    """
    try:
        data = request.get_json()
        
        # Required parameters
        task = data.get('task')
        agents = data.get('agents')
        coordinator = data.get('coordinator')
        
        if not all([task, agents, coordinator]):
            return jsonify({
                'error': 'Missing required parameters. Need: task, agents, coordinator'
            }), 400
        
        # Optional parameters
        verbose = data.get('verbose', False)
        chain_store_api_key = data.get('chain_store_api_key')
        max_workers = data.get('max_workers')
        return_reasoning = data.get('return_reasoning', False)
        max_reasoning_steps = data.get('max_reasoning_steps')
        coordinator_max_steps = data.get('coordinator_max_steps')
        wolfram_app_id = data.get('wolfram_app_id')
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 1.0)
        max_tokens = data.get('max_tokens', 500)
        image = data.get('image', None)
        output_tools = data.get('output_tools')
        reflection_mode = data.get('reflection_mode', False)

        # Run ensemble
        result = ensemble(
            task=task,
            agents=agents,
            coordinator=coordinator,
            verbose=verbose,
            chain_store_api_key=chain_store_api_key,
            max_workers=max_workers,
            return_reasoning=return_reasoning,
            max_reasoning_steps=max_reasoning_steps,
            coordinator_max_steps=coordinator_max_steps,
            wolfram_app_id=wolfram_app_id,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            image=image,
            output_tools=output_tools,
            reflection_mode=reflection_mode
        )
        
        if return_reasoning:
            coordinator_response, agent_results = result
            return jsonify({
                'response': coordinator_response,
                'agent_results': [
                    {
                        'model': config['model'],
                        'response': response,
                        'reasoning_chain': history,
                        'thinking_tools': thinking_tools,
                        'output_tools': output_tools
                    }
                    for config, response, history, thinking_tools, output_tools in agent_results
                ]
            })
        
        return jsonify({
            'response': result
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050) 