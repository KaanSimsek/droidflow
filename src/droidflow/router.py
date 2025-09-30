import logging
from .domain import DomainAgent
from .model import History, State, RequestAndReply, SkipStep


class RouterAgent:
    def __init__(self, llm, agents: list[DomainAgent], route_prompt: str, final_answer_prompt: str, history_enabled: bool = False, debug: bool = False):
        self.llm = llm
        self.agents = None
        self.set_agents(agents)
        self.state = State()
        self.history_enabled = history_enabled
        if history_enabled:
            self.history = History()

        self.route_prompt = route_prompt
        self.final_answer_prompt = final_answer_prompt

        self.debug = debug
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def set_agents(self, agents: list[DomainAgent]):
        tools = {}
        for agent in agents:
            function_name = f"{agent.name}_execute"
            function_to_call = agent.execute
            tools[function_name] = function_to_call
        self.agents = tools
        return tools

    def route(self, plans: list[str]) -> str:
        for step_id, step in enumerate(plans):
            self._route_step(step, step_id)
        if self.debug:
            self.logger.debug(f"Final history: {self.history.history}")
        history_text = "\n".join(
            f"Step {i}: {h.step}\n→ {h.response}"
            for i, h in enumerate(self.history.history)
        )

        prompt = f"{self.final_answer_prompt}:\n{history_text}"
        resp = self.llm.generate_content(prompt)

        final_text = ""
        if hasattr(resp, "text"):
            final_text = resp.text
        elif hasattr(resp, "candidates"):
            parts = resp.candidates[0].content.parts
            final_text = "".join(getattr(p, "text", "") for p in parts)

        return final_text

    def _route_step(self, step: str, step_id: int):
        if self.debug:
            self.logger.debug(f"\n--- Step {step_id}: {step} ---")

        history_prompt = self._set_history_prompt()
        prompt = (
            f"{history_prompt}\n"
            f"Task: '{step}'. {self.route_prompt}\n"
        )

        response = self.llm.generate_content(prompt)
        tool_call = response.candidates[0].content.parts[0].function_call

        if tool_call:
            if self.debug:
                self.logger.debug(f"Function call: {tool_call}")
            agent_to_run = self.agents.get(tool_call.name)

            if agent_to_run:
                result, self.state = agent_to_run(step + ". Chat history:\n" + "\n".join(map(str, self.history.history)), self.state)
                if isinstance(result, SkipStep):
                    return
                self.history.append(RequestAndReply(step, result))
            else:
                error_msg = f"Error: Agent '{tool_call.name}' not found."
                self.history.append(RequestAndReply(step, error_msg))
                self.logger.warning(error_msg)
        else:
            error_msg = "Error: No tool was selected."
            self.history.append(RequestAndReply(step, error_msg))
            self.logger.warning(error_msg)

    def _set_history_prompt(self):
        history_prompt = ""
        if self.history_enabled:
            history_prompt += "Previous steps and their outputs:\n"
            for prev_id, prev in enumerate(self.history.history):
                history_prompt += f"- Step {prev_id}: {prev.step}\n"
                history_prompt += f"  → Output: {prev.response}\n"
        return history_prompt


