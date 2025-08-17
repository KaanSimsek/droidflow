import logging

from .domain import ToolFunction

SYSTEM_PROMPT = """
        You are a reasoning AI agent designed to answer financial questions by calling functions (tools) when necessary. 

        You can access:
        - A vector database for semantic search
        - A knowledge graph for entity relationships (e.g., companies ↔ disclosures)
        - Backend APIs for real-time or structured financial data (e.g., company sector, market info, disclosures)

        Your goal is to:
        1. Understand the user's intent
        2. Decide if any tool is required to answer the query
        3. Call one or more tools in order to gather relevant information
        4. Combine and interpret results
        5. Produce a final answer in plain, informative language

        Do not assume or hallucinate data. Always use the available tools when real data is needed.

        Examples of useful tool usage:
        - Searching related disclosures or news
        - Fetching sector or financial data
        - Looking up relationships between entities (e.g., companies ↔ topics)

        If you can answer without a tool, you may do so.
        If tool results are required, call them first and wait for results before continuing.

        Respond in a helpful, concise, and accurate manner.
        """

class ReasonerAgent(object):
    def __init__(self, llm, tool_functions: list[ToolFunction], name: str, debug: bool = False):
        self.llm = llm
        self.tool_functions = tool_functions
        self.name = name

        self.debug = debug

        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def execute(self, query: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        response = self.llm.generate_content(messages)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                function_to_call = self._find_function(tool_call.function.name)

                if not function_to_call:
                    raise ValueError(f"Unknown tool call: {tool_call.function.name}")

                args = {key: value for key, value in tool_call.args.items()}
                function_response_data = function_to_call.callable(**args)

                messages.append({
                    "role": "function",
                    "name": tool_call.function.name,
                    "content": function_response_data
                })
            response = self.llm.chat(messages, tools=self.tool_functions, tool_choice="auto")

        return response.text

    def _find_function(self, name):
        for func in self.tool_functions:
            if func.name == name:
                return func
        return None
