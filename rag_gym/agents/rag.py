"""
    RAG (Retrieval-Augmented Generation) Agent 的实现。
    RAG Agent 遵循两阶段流程：
    1. 检索阶段：如果历史记录为空，则生成检索查询（默认使用原始问题）。
    2. 生成阶段：如果历史记录中已有检索到的文档，则结合文档和问题生成最终答案。
"""
import re
import json
import torch
from liquid import Template
from liquid.template import BoundTemplate
from rag_gym import State, LLMEngine, Action, BaseAgent

# RAG 的系统提示词
rag_system_prompt = '''You are a helpful assistant. Your task is to think step-by-step and answer a given question using the provided relevant documents.'''

# RAG 的用户模板，包含检索到的文档和原始问题
rag_user_template = Template('''### Relevant Documents
{{documents}}

### Question
{{question}}

Your output must include two sections:
1. **### Step-by-step Reasoning**:
  - Think step-by-step and then answer the question using the provided relevant documents.

2. **### Structured Output**:
  - Present your predicted answer in the following JSON format:
  ```json
  {
    "predicted_answer": "Provide a single letter (for multiple-choice questions), digit, word, or short phrase here."
  }
  ```''')

class RAGAgent(BaseAgent):
    """
    RAG Agent 类，封装了检索增强生成的逻辑。
    """
    def __init__(
        self, 
        llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        system_prompt: str | None = None, 
        user_template: BoundTemplate | None = None, 
        cache_dir: str | None = None, 
        api: bool = False, 
        model_dtype: torch.dtype = torch.bfloat16,
        reward_llm_name: str | None = None,
        train_mode: bool = False,
        **kwargs
    ):
        super().__init__(llm_name, cache_dir, api, model_dtype, reward_llm_name, train_mode)
        self.agent_type = "rag"
        self.system_prompt = system_prompt if system_prompt is not None else rag_system_prompt
        self.user_template = user_template if user_template is not None else rag_user_template

    def generate_action(
        self, 
        state: State, 
        max_new_tokens: int = 2048, 
        temperature: float = 0.0, 
        num_actions: int = 1,
        **kwargs
    ):
        """
        根据当前状态生成动作。
        
        Args:
            state: 当前的环境状态，包含问题和历史记录（检索到的文档）。
            max_new_tokens: 生成的最大 token 数。
            temperature: 采样温度。
            num_actions: 生成的候选动作数量。
            
        Returns:
            list[Action]: 包含检索动作 (Action(query=...)) 或回答动作 (Action(answer=...)) 的列表。
        """
        # 如果历史记录为空，说明尚未进行检索，生成检索动作
        if len(state.history) == 0:
            return [Action(query=state.question)]
        
        # 如果已有历史记录，则根据文档生成回答
        action_strings = []
        try:
            action_strings = self.llm.generate(
                messages = self.apply_template(state), 
                max_new_tokens = max_new_tokens, 
                temperature = temperature, 
                num_return_sequences = num_actions, 
                **kwargs
            )
            actions = []
            for action_str in action_strings:
                action = self.post_process(action_str)
                actions.append(action)
        except Exception as E:
            error_class = E.__class__.__name__
            actions = [Action(answer=f"{error_class}: {str(E)}", action_string=f"{error_class}: {str(E)}")]
        return actions
    
    def post_process(self, action_str):
        """
        对 LLM 生成的字符串进行后处理，提取 JSON 格式的预测答案。
        """
        match = []
        try:
            # 尝试从 "### Structured Output" 之后的部分提取 JSON
            match = re.findall(r'```json\s*({(?:[^`]|\`(?!``))*})', action_str.split("### Structured Output")[-1], re.DOTALL)
            match = match if len(match) > 0 else re.findall(r'{.*?}', action_str.split("### Structured Output")[-1], re.DOTALL)
            output = eval(re.sub(r' //.*', '', match[-1].replace("null", "None"))) # 移除注释并解析
            answer = output["predicted_answer"]
            action = Action(answer=str(answer), action_string=action_str)
        except:
            # 解析失败时的兜底处理
            action = Action(answer=action_str.split("predicted_answer")[-1], action_string=action_str)
        return action

    def apply_template(self, state):
        """
        将当前状态（问题 + 检索到的文档）应用到 Prompt 模板中。
        """
        question = state.question
        # 格式化检索到的文档列表
        documents = '\n'.join(["Document [{:d}] (Title: {:s}) {:s}".format(idx, state.history[0]["documents"][idx]["title"], state.history[0]["documents"][idx]["content"]) for idx in range(len(state.history[0]["documents"]))])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.render(question=question, documents=documents)}
        ]
        return messages