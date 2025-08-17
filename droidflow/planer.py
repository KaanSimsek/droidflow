def _default_planner_prompt():
    return """
        You are a planner droidflow. Your goal is to break down the user's task into minimal, essential steps.
        Respond with numbered steps or return `MISSING_INFO: [field]` if critical data is missing.
                """.strip()


def _parse_steps(plan_text: str):
    steps = []
    for line in plan_text.strip().split('\n'):
        if line.strip().startswith("MISSING_INFO:"):
            return line.strip()
        if line.strip() and line[0].isdigit():
            step = line.split('.', 1)[1].strip()
            steps.append(step)
    return steps


class PlannerAgent(object):
    def __init__(self, llm, domain_prompt: str):
        self.llm = llm
        self.domain_prompt = domain_prompt

    def build_prompt(self, user_prompt: str) -> str:
        return f"""{self.domain_prompt}
                {_default_planner_prompt()}
                Task: {user_prompt}
                Begin."""

    def plan(self, user_prompt: str):
        full_prompt = self.build_prompt(user_prompt)
        response = self.llm.generate_content(full_prompt)
        return _parse_steps(response.text)



