from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class ConversationManager:
    """
    Manages multi-turn conversation memory with a sliding window buffer.
    """
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_ai_message(self, content: str):
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def get_messages(self) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        for msg in self.history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def get_raw_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    def _trim(self):
        # 1 turn = 1 user + 1 assistant message (2 messages)
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def clear(self):
        self.history = []

