# droidflow

**droidflow** is a lightweight multi-agent orchestration library designed to coordinate reasoning, planning, and tool-using agents. It is built for financial and data-driven applications, but is flexible enough for general AI agent workflows.

---

## Features

- **PlannerAgent** – Breaks down a user query into minimal actionable steps  
- **RouterAgent** – Routes sub-tasks to the appropriate specialized agent  
- **DomainAgent** – Wraps tools (functions/APIs) into callable interfaces  
- **ReasonerAgent** – Executes reasoning with tool calls when needed  
- Simple `State` & `History` management for contextual workflows  
- Easy integration with LLMs (Google Generative AI, OpenAI, etc.)  

---

## Installation

```bash
pip install droidflow

## Quick Example

```python
from droidflow.planer import PlannerAgent
from droidflow.router import RouterAgent
from droidflow.domain import DomainAgent, ToolFunction
from droidflow.model import State

# Example tool function
def get_stock_price(symbol: str) -> str:
    return f"Price of {symbol} is 100"

# Register tool
tool_fn = ToolFunction("get_stock_price", get_stock_price, state_enabled=False)

# Dummy LLM (replace with Google Generative AI, OpenAI, etc.)
class DummyLLM:
    def generate_content(self, prompt):
        print(f"[LLM called] {prompt}")
        class Response: text = "SINGLE_CALL"
        return Response()

llm = DummyLLM()

# Create agents
stock_agent = DomainAgent(llm, [tool_fn], name="stock_agent", mode=True)
planner = PlannerAgent(llm, "You are a planner for stock tasks.")
router = RouterAgent(llm, agents=[stock_agent])

# Run
plans = planner.plan("Get the price of AAPL stock")
print("Plans:", plans)

result = router.route(plans)
print("Final result:", result)

