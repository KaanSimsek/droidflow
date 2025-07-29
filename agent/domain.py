import logging
from typing import Dict, Callable, Optional, List


class DomainAgent(object):
    def __init__(self, llm, tool_functions: Dict[str, Callable], mode: bool, debug: bool = False):
        self.llm = llm
        self.tool_functions = tool_functions
        self.mode = mode

        self.debug = debug

        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def execute(self, query: str):
        if self.mode:
            sub_plans = self._plan(query)
            self.llm.execute()

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

    def execute_query(self, query: str, chat_history: Optional[List[Dict]] = None):
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

        function_to_call = self.tool_functions.get(function_name)
        if not function_to_call:
            raise ValueError(f"Unknown tool call: {function_name}")

        args = {key: value for key, value in function_call.args.items()}
        function_response_data = function_to_call(**args)

        self.logger.debug(f"Tool executed. Result: {function_response_data}")

        return function_response_data
