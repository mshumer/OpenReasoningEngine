<div align="center">

# OpenReasoningEngine

**While AI labs are quietly building closed reasoning systems,  
we can create something more powerful together in the open.**

</div>

---

This repo serves as a modular, open-source test-time compute engine — anyone in the community with a useful idea to improve model capabilities is encouraged to add their approach to the system. As approaches are added, this system will enable users to compose them to drastically increase capabilities.

And over time, as users save successful reasoning chains, we will be able to train models designed to take full advantage of this system.

> ### ⚠️ Important Note
> **We are going to be very selective about what we add to this system. If an approach doesn't have a clear path to increasing the capabilities of the system, we will not add it.**

---

## 🚀 Initial System

### Core Features

🔹 **Step-by-Step Reasoning**  
   &nbsp;&nbsp;&nbsp;&nbsp;Executes reasoning one step per turn with integrated tools:
   - Python interpreter
   - Web search (via Perplexity)
   -  Wolfram Alpha integration

🔹 **Memory-Based Planning**  
   &nbsp;&nbsp;&nbsp;&nbsp;Continually learns and adapts from past experiences

🔹 **MoA**  
   &nbsp;&nbsp;&nbsp;&nbsp;Implements mixture-of-agents for ensemble decision making

🔹 **Beam Search**  
   &nbsp;&nbsp;&nbsp;&nbsp;Sample multiple next reasoning step candidates at each turn, and choose the best (soon to be updated with forking Python interpreters to significantly improve the system)

🔹 **Self-Reflection**  
   &nbsp;&nbsp;&nbsp;&nbsp;Force the AI to validate reasoning steps as it thinks

🔹 **Flexible Model Support**  
   &nbsp;&nbsp;&nbsp;&nbsp;Model-agnostic API supporting any OpenAI-compatible provider (OpenAI, Anthropic, etc.)

🔹 **Rich Input/Output**  
   &nbsp;&nbsp;&nbsp;&nbsp;Handles image input, **function calling**, and multi-turn conversations

---

## ⚙️ Installation

### 1. Clone and Install
```bash
git clone https://github.com/mshumer/OpenReasoningEngine.git
cd OpenReasoningEngine
pip install -r requirements.txt
```

### 2. API Setup
Get API keys from [OpenRouter](https://openrouter.ai/) and [E2B](https://e2b.dev/)

Create a `.env` file:
```env
E2B_API_KEY="your_e2b_key_here"
OPENROUTER_API_KEY="your_openrouter_key_here"
```

### 3. Load Environment
```bash
source .env
```

---

## 🛠️ Usage

### Running the Engine
Two options available:
- Direct execution: `python main.py`
- API server: `python api.py` (starts a Flask API endpoint)

## Config Options
Running the code as-is will work — I've chosen reasonable default settings. If you'd like to customize the way the system reasons, here are all of the settings you can configure: [LINK HERE!]

### Tool System

#### 1. Internal Tools
- Used during the reasoning process
- Default setup includes:
  - Python interpreter
  - Web search (Perplexity API)
  - Wolfram Alpha (optional)
- Customizable based on your needs

#### 2. Output Tools
- Standard AI API output tools
- Called after reasoning completion
- Configurable based on use-case

---

## 🧮 Learning System

### Memory Management

A major goal of OpenReasoningEngine is to enable learning from experience. The initial implementation is simple, and will continue to be iterated on as I (and others) come up with smarter approaches.

#### Steps to Enable Continual Learning:

1. Obtain an API key from [Cohere](https://cohere.ai/)

2. Save successful reasoning chains:
```python
chain_store.save_successful_chain(
    task=task,
    conversation_history=history,
    final_response=response,
    cohere_api_key=cohere_api_key,
    thinking_tools=thinking_tools,
    output_tools=output_tools,
    metadata={"model": model, "api_url": api_url}
)
```

The system includes starter chains in `successful_chains.json`. Community contributions to this database are welcome, subject to validation. If you have ideas to make this process more seamless and scalable, please reach out!

### 📊 Performance Notes

- AIME tasks show improved performance with memory-based planning
- HumanEval tasks perform better without memory planning
- Performance may vary based on the specific chains in your memory store (the above cases were tested using the starter chains in `successful_chains.json`, so performance may be dramatically different with different chains)

---

## 📝 Logging

### Verbose Mode
When `verbose=True`, the engine displays:
- 🔄 API interactions
- 🛠️ Tool usage and results
- 📋 Step-by-step reasoning progress

This makes it easy to see what's going on under the hood and diagnose issues.

---

## 🤝 Contributing

Contributions are welcome if they:
- ✨ Demonstrably improve system capabilities
- 📈 Include clear performance metrics

Quality-of-life improvements are also appreciated.

---

## Acknowledgements
Thank you to the following folks who provided advice, feedback, ideas, and helped me implement and test the initial versions of OpenReasoningEngine:
- [Steve Ickman](https://x.com/stevenic)
- [Vasek Mlejnsky](https://x.com/mlejva)
- [Josh Bickett](https://x.com/josh_bickett)
- [Aidan Gomez](https://x.com/aidangomez)
- [Alec Velikanov](https://x.com/alecvxyz) (Alex, imo)

[Follow me on X](https://x.com/mattshumer_) for updates on this and other AI things I'm working on.
