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
        "api_key": "your-api-key",
        "model": "model-name",
        "api_url": "api-endpoint",
        "temperature": 0.7,            # optional
        "top_p": 1.0,                 # optional
        "max_tokens": 500,            # optional
        "verbose": false,             # optional
        "chain_store_api_key": "key"  # optional
    }
    """
    try:
        data = request.get_json()
        
        # Required parameters
        task = data.get('task')
        api_key = data.get('api_key')
        model = data.get('model')
        api_url = data.get('api_url')
        
        if not all([task, api_key, model, api_url]):
            return jsonify({
                'error': 'Missing required parameters. Need: task, api_key, model, api_url'
            }), 400
        
        # Optional parameters
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 1.0)
        max_tokens = data.get('max_tokens', 500)
        verbose = data.get('verbose', False)
        chain_store_api_key = data.get('chain_store_api_key')
        
        # Run reasoning
        response, history, tools = complete_reasoning_task(
            task=task,
            api_key=api_key,
            model=model,
            api_url=api_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            verbose=verbose,
            chain_store_api_key=chain_store_api_key
        )
        
        return jsonify({
            'response': response,
            'conversation_history': history,
            'tools': tools
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
                "api_url": "url-1"
            },
            {
                "model": "model-name-2",
                "api_key": "key-2",
                "api_url": "url-2"
            }
        ],
        "coordinator": {
            "model": "model-name",
            "api_key": "key",
            "api_url": "url"
        },
        "verbose": false,             # optional
        "chain_store_api_key": "key", # optional
        "max_workers": 3,             # optional
        "return_reasoning": false     # optional
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
        
        # Run ensemble
        result = ensemble(
            task=task,
            agents=agents,
            coordinator=coordinator,
            verbose=verbose,
            chain_store_api_key=chain_store_api_key,
            max_workers=max_workers,
            return_reasoning=return_reasoning
        )
        
        if return_reasoning:
            coordinator_response, agent_results = result
            return jsonify({
                'response': coordinator_response,
                'agent_results': [
                    {
                        'model': config['model'],
                        'response': response,
                        'reasoning_chain': history
                    }
                    for config, response, history, _ in agent_results
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
    app.run(host='0.0.0.0', port=5000) 