from dataclasses import dataclass
from typing import Any

@dataclass
class RequestAndReply:
    step: str
    response: str

class History(object):
    def __init__(self):
        self.history: list[RequestAndReply] = []
    def append(self, request: RequestAndReply):
        self.history.append(request)


class State(object):
    def __init__(self):
        self.state = dict()
    def append(self, key: str, value: Any) -> None:
        self.state[key] = value

class SkipStep:
    pass
