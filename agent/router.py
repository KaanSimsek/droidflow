from typing import Dict, Callable


import logging
from typing import Dict, Callable

class RouterAgent:
    def __init__(self, llm, agent: Dict[str, Callable], history_enabled: bool = False, debug: bool = False):
        self.llm = llm
        self.agents = agent
        self.history_enabled = history_enabled
        self.debug = debug

        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def route(self, plans: list[str]):
        history = []
        for step_id, step in enumerate(plans):
            if self.debug:
                self.logger.debug(f"\n--- Step {step_id}: {step} ---")
            history_prompt = ""
            if history:
                history_prompt += "Previous steps and their outputs:\n"
                for prev_id, prev in enumerate(history):
                    history_prompt += f"- Step {prev_id}: {prev['question']}\n"
                    history_prompt += f"  → Output: {prev['answer']}\n"

            prompt = (
                f"{history_prompt}\n"
                f"Task: '{step}'. Generate only the appropriate agent's function_call to complete this task. "
                "Do not return any explanation or additional text. Only produce the function_call."
            )

            response = self.llm.generate_content(prompt)
            tool_call = response.candidates[0].content.parts[0].function_call

            if tool_call:
                if self.debug:
                    self.logger.debug(f"Function call: {tool_call}")
                agent_name_to_find = tool_call.name.replace("trigger_", "").replace("_agent", "")
                agent_to_run = self.agents.get(agent_name_to_find)

                if agent_to_run:
                    args = {
                        'task': step + ". Chat history:\n" + "\n".join(map(str, history))
                    }
                    result = agent_to_run(**args)
                    history.append({"question": step, "answer": result})
                else:
                    error_msg = f"Error: Agent '{tool_call.name}' not found."
                    history.append({"question": step, "answer": error_msg})
                    self.logger.warning(error_msg)
            else:
                error_msg = "Error: No tool was selected."
                history.append({"question": step, "answer": error_msg})
                self.logger.warning(error_msg)

        if self.debug:
            self.logger.debug(f"Final history: {history}")



