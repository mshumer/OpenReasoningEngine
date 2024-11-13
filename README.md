# OpenReasoningEngine

This repo will serve as a modular, open-source test-time compute engine — anyone in the community with a useful idea to improve model capabilities through is encouraged to add their approach to the system. As many approaches are added, this system will enable users to compose them to drastically increase capabilities.

## Initial Features

- Step-by-step reasoning with integrated tools (calculator, Python interpreter, web access, Wolfram Alpha)
- Semantic chain search powered by Cohere embeddings so the models within the system can continually learn to better use this engine's rails
- Parallel ensemble processing with multiple models (MoA)
- Stateful Python interpreter sessions, enabling multiple threads at once
- Model-agnostic API supporting multiple providers (OpenAI, Anthropic, etc.)

## Installation

```bash
git clone https://github.com/mshumer/OpenReasoningEngine.git
cd OpenReasoningEngine
pip install -r requirements.txt
```

## Usage

### Basic Mode

Use a single model for reasoning tasks:

```python
from engine import complete_reasoning_task

response, history, tools = complete_reasoning_task(
    task="Calculate compound interest on $1000 for 5 years at 5% annual interest",
    api_key="your-api-key",
    model="anthropic/claude-3-opus",
    api_url="https://openrouter.ai/api/v1/chat/completions",
    verbose=True
)
```

### Ensemble Mode

Leverage multiple models with a coordinator:

```python
from mixture import ensemble

ensemble_response = ensemble(
    task=task,
    agents=[
        {
            "model": "openai/gpt-4",
            "api_key": "key1",
            "api_url": "url1"
        },
        {
            "model": "anthropic/claude-3",
            "api_key": "key2",
            "api_url": "url2"
        }
    ],
    coordinator={
        "model": "anthropic/claude-3-opus",
        "api_key": "key3",
        "api_url": "url3"
    },
    verbose=True
)
```

## Built-in Tools

### Calculator

Evaluate mathematical expressions:

```python
{
    "tool": "calculator",
    "parameters": {
        "operation": "2 + 2"
    }
}
```

### Python Interpreter

Execute Python code with optional state persistence:

```python
{
    "tool": "python",
    "parameters": {
        "code": "x = 5",
        "thread_id": "my_session"  # Optional: maintains state across calls
    }
}
```

## Chain Management

### Storing Successful Chains

```python
chain_store.save_successful_chain(
    task="your task",
    conversation_history=history,
    final_response=response,
    cohere_api_key="your-key",
    tools=tools
)
```

### Automatic Chain Retrieval

Similar chains are automatically retrieved for new tasks:

```python
response, history, tools = complete_reasoning_task(
    task="your new task",
    chain_store_api_key="your-cohere-key",
    # ... other parameters
)
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `task` | The reasoning task to complete | Required |
| `api_key` | Provider API key | Required |
| `model` | Model identifier | Required |
| `api_url` | API endpoint | Required |
| `verbose` | Show detailed progress | `False` |
| `log_conversation` | Save conversation to file | `False` |
| `chain_store_api_key` | Cohere API key for semantic search | Optional |

## Logging

### Verbose Mode
When `verbose=True`, the engine displays:
- API interactions
- Tool usage and results
- Step-by-step reasoning progress

### Conversation Logging
When `log_conversation=True`, conversations are saved to:
```
logs/conversation_[timestamp].json
```

## Examples

See `main.py` for complete examples including:
- Mathematical calculations
- Python code execution
- Ensemble reasoning
- Chain storage and retrieval

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
