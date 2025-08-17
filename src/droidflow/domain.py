import dataclasses
import logging
from typing import Dict, Callable, Optional, List

from src.droidflow.model import State

@dataclasses.dataclass
class ToolFunction:
    name: str
    callable: Callable
    state_enabled: bool

class DomainAgent(object):
    def __init__(self, llm, tool_functions: list[ToolFunction], name: str, mode: bool, debug: bool = False):
        self.llm = llm
        self.tool_functions = tool_functions
        self.mode = mode
        self.name = name

        self.debug = debug

        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def execute(self, query: str, state: State):
        if self.mode:
            sub_plans = self._plan(query)
            if sub_plans is None:
                return self.execute_query(query, state)
            else:
                chat_history = []
                for sub_plan in sub_plans:
                    response, state = self.execute_query(sub_plan, state, chat_history)
                    chat_history.append(
                        {
                            "query": query,
                            "response": response,
                        }
                    )
        else:
            return self.execute_query(query, state)

    def _plan(self, query: str) -> Optional[List[str]]:
        prompt = (
        f"""
        Analyze the following user query:

        "{query}"

        Determine whether a **single tool/API call** is sufficient to answer it.

        - If a single tool/API call is enough, respond with:
        SINGLE_CALL

        - Otherwise, break the query down into a **step-by-step plan** using a numbered list like:
        Step 1: ...
        Step 2: ...
        Step 3: ...

        Only return the required output. Do not explain your reasoning."""
        )

        response = self.llm.generate_content(prompt)
        if response.text.strip() == "SINGLE_CALL":
            return None

        steps = [line.strip() for line in response.text.split("\n") if line.strip()]
        return steps

    def execute_query(self, query: str, state: State, chat_history: Optional[List[Dict]] = None):
        history_prompt = ""
        if chat_history:
            history_prompt += "Previous steps and their outputs:\n"
            for prev_id, (prev_step, prev_response) in enumerate(chat_history):
                history_prompt += f"- Step {prev_id}: {prev_step}\n"
                history_prompt += f"  → Output: {prev_response}\n"

        prompt = (
            f"{history_prompt}\n"
            f"Task: '{query}'. Generate only the appropriate function_call to complete this task. "
            "Do not return any explanation or other text. Only produce a function_call."
        )

        response = self.llm.generate_content(prompt)
        function_call = response.candidates[0].content.parts[0].function_call
        function_name = function_call.name

        self.logger.debug(f"Agent wants to execute '{function_name}'.")

        function_to_call = self._find_function(function_name)
        if not function_to_call:
            raise ValueError(f"Unknown tool call: {function_name}")

        args = {key: value for key, value in function_call.args.items()}
        if function_to_call.state_enabled:
            args['state'] = state
            function_response_data, state = function_to_call.callable(**args)
        else:
            function_response_data = function_to_call.callable(**args)

        self.logger.debug(f"Tool executed. Result: {function_response_data}")

        return function_response_data, state

    def _find_function(self, name):
        for func in self.tool_functions:
            if func.name == name:
                return func
        return None
