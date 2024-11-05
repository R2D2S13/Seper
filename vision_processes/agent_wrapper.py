from LLM.chatgpt_function_model import TemplateChatGpt
class AgentWrapper():
    def __init__(self,agent_llm:TemplateChatGpt) -> None:
        self.agent_llm = agent_llm
        