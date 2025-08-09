# SimulationSDK

A Python framework for simulating tool calls and tracking agent performance in AI workflows. Test your AI agents safely without affecting production systems.

## 🚀 Quick Start

The SDK uses decorators to make your functions "simulation-aware". Here's how it works:

**1. Import the SDK and enable simulation mode:**
```python
import os
from simulation_sdk import simulation_tool, ToolCategory, SimulationResponse

# Turn on simulation mode for testing
os.environ["SIMULATION_MODE"] = "true"
```

**2. Decorate your function to make it simulatable:**
```python
@simulation_tool(
    category=ToolCategory.PRODUCTION_AFFECTING,  # This is a dangerous operation, we simulate the response
    success_template=SimulationResponse(         # What to return in simulation mode
        success=True,
        response_data={"email_id": "sim_123", "status": "sent"}
    ),
    failure_template=SimulationResponse(         # Error response for testing
        success=False,
        error_message="Failed to send email"
    )
)
def send_email(to: str, subject: str) -> dict:
    # This code only runs in production mode
    return {"email_id": "real_123", "status": "sent"}
```

**3. Call your function normally:**
```python
result = send_email("test@example.com", "Hello")
# In simulation mode: returns {"email_id": "sim_123", "status": "sent"}
# In production mode: actually sends email and returns real result
```

The decorator automatically checks if you're in simulation mode and returns fake data instead of executing the real function. This lets you test your AI agents without sending real emails, creating real files, or calling real APIs!

## 📦 Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/Scarlet23333/SimulationSDK.git
cd SimulationSDK
pip install -e .
```

Or add to your project's requirements:

```python
# In your Python files
import sys
sys.path.append('/path/to/SimulationSDK')
from simulation_sdk import simulation_tool, ToolCategory
```

## 🎯 Core Concepts

### Tool Categories of @simulation_tool

```python
ToolCategory.READ_ONLY           # Always executes (safe operations)
ToolCategory.PRODUCTION_AFFECTING # Simulated in test mode  
ToolCategory.AGENT_TOOL          # Uses cached performance data (decorate llm agent functions to save and reuse results and performance)
```

### Read Only Tools

Read only tools (functions that you want to track it performance) execute normally in simulation mode, but the performance (duration) would be record for evaluation.

```python
@simulation_tool(
    category=ToolCategory.READ_ONLY,
    description="Search for documents"  # optional
)
def search_documents(query: str) -> list:
    """Search for documents matching the query."""
    return [
        {"id": 1, "title": "Document 1", "relevance": 0.95},
        {"id": 2, "title": "Document 2", "relevance": 0.87}
    ]
```

### Production Affecting Tools

Simulate production affecting tool call (external api call) to skip real operation and return fake response

```python
@simulation_tool(
    category=ToolCategory.PRODUCTION_AFFECTING,  # This is a dangerous operation, we simulate the response
    success_template=SimulationResponse(         # What to return in simulation mode
        success=True,
        response_data={"email_id": "sim_123", "status": "sent"}
    ),
    failure_template=SimulationResponse(         # Error response for testing
        success=False,
        error_message="Failed to send email"
    ),
    delay_ms=500,   # optional, prior than delay_range
    delay_range=(200, 300),  # optional
    description=""  # optional
)
def send_email(to: str, subject: str) -> dict:
    # This code only runs in production mode
    return {"email_id": "real_123", "status": "sent"}
```

### Smart Agent Tools

Create AI agent tools that automatically cache their performance:

```python
@simulation_tool(
    category=ToolCategory.AGENT_TOOL,
    agent_name="email_writer", # required
    delay_ms=500,   # optional, prior than delay_range
    delay_range=(200, 300),  # optional
    description=""  # optional
)
def email_writer_agent(topic: str) -> dict:
    # Your agent logic here
    result = send_email("user@example.com", f"About {topic}")
    return {"status": "completed", "email_id": result["email_id"]}

# First call: executes and saves performance
# Later calls: uses cached performance data
```

### Simulate task

Always executes, evaluate and saves performance. Use when need to update and save an agent's performance or need to call a agent many times with different prompts. If only need one time execute and cached performance, just use @simulation_tool(category=ToolCategory.AGENT_TOOL, ...).

```python
@simulation_agent(
    name="research_assistant",
    delay_ms=1000,   # optional, prior than delay_range
    delay_range=(500, 900),  # optional
)
def research_topic(topic: str) -> dict:
    """Research a topic and provide summary."""
    # Search for relevant documents
    docs = search_documents(f"query: {topic}")
    
    # Extract information from top documents
    summaries = []
    for doc in docs[:2]:
        info = extract_info(doc["id"])
        summaries.append(info)
    
    # Compile research results
    return {
        "topic": topic,
        "documents_reviewed": len(summaries),
        "key_findings": [s["summary"] for s in summaries],
        "confidence": 0.85
    }
```

### Automatic Token Tracking

Track LLM tokens with a simple decorator:

```python
from simulation_sdk import track_llm_tokens

@track_llm_tokens
def call_openai(prompt: str):
    return openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
```

### Workflow Tracking

Track complete workflows with metrics:

```python
from simulation_sdk import get_current_context, save_workflow_metrics

context = get_current_context()
context.start_workflow("wf_001", "Customer Support")

# Run your agents...
result = email_writer_agent("Product inquiry")

# End workflow and get metrics
metrics = context.end_workflow()
save_workflow_metrics(metrics)
```

## 📊 Automatic Evaluation

The SDK automatically evaluates agent performance using:
- **OpenAI GPT** (if OPENAI_API_KEY is set)
- **MockEvaluator** (fallback option)

Evaluation considers:
1. **Correctness** - Did the agent achieve its goal?
2. **Efficiency** - Minimal tool calls and reasonable duration?
3. **Token Usage** - Appropriate token consumption?

## 🔧 Configuration

```bash
# Enable simulation mode (default: false)
export SIMULATION_MODE=true

# Set OpenAI key for intelligent evaluation
export OPENAI_API_KEY=your_key_here

# Error injection for testing (0.0-1.0)
export SIMULATION_ERROR_RATE=0.1

# Configure performance data location (default: ./simulation_data)
export SIMULATION_STORAGE_PATH=/path/to/your/data
```

## 📚 Examples

- `basic_tool_example.py` - Simple tool creation and usage
- `smart_agent_workflow_example.py` - Agent coordination patterns
- `token_tracking_example.py` - Automatic LLM token tracking
- `automatic_evaluation_example.py` - Performance evaluation

## 🔄 Development Workflow

1. **Create tools** with appropriate categories use `@simulation_tool()`
2. **Track tokens** with `@track_llm_tokens` decorator
3. **Build agents** using `@simulation_agent(name="researcher")`
4. **Test in simulation** mode to save performance
5. **Switch to production** when ready

## 📝 License

MIT License - see LICENSE file for details