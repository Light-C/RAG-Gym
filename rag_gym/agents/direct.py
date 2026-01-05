"""
    Direct Agent 的实现。
    该 Agent 采用最简单直接的方式回答问题，不进行检索或复杂的思维链推理。
"""
import re
import json
import torch
from liquid import Template
from liquid.template import BoundTemplate
from rag_gym import State, LLMEngine, Action, BaseAgent

# Direct 模式的系统提示词
direct_system_prompt = '''You are a helpful assistant. Your task is to answer a given question with no additional text or explanations.'''

# Direct 模式的用户模板，仅包含问题
direct_user_template = Template('''### Question
{{question}}

Directly present your predicted answer in the following JSON format:
```json
{
"predicted_answer": "Provide a single letter (for multiple-choice questions), digit, word, or short phrase here."
}
```''')

class DirectAgent(BaseAgent):
    """
    直接回答问题的 Agent。
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
        self.agent_type = "direct"
        self.system_prompt = system_prompt if system_prompt is not None else direct_system_prompt
        self.user_template = user_template if user_template is not None else direct_user_template

    def generate_action(
        self, 
        state: State, 
        max_new_tokens: int = 2048, 
        temperature: float = 0.0, 
        num_actions: int = 1,
        **kwargs
    ):
        """
        生成回答动作。
        
        Args:
            state: 当前状态，包含问题。
            max_new_tokens: 生成的最大 token 数。
            temperature: 采样温度。
            num_actions: 生成的候选动作数量。
            
        Returns:
            list[Action]: 包含回答动作的列表。
        """
        action_strings = []
        try:
            # 调用 LLM 生成原始字符串
            action_strings = self.llm.generate(
                messages = self.apply_template(state), 
                max_new_tokens = max_new_tokens, 
                temperature = temperature, 
                num_return_sequences = num_actions, 
                **kwargs
            )
            actions = []
            for action_str in action_strings:
                # 对每个生成的字符串进行后处理提取答案
                action = self.post_process(action_str)
                actions.append(action)
        except Exception as E:
            error_class = E.__class__.__name__
            actions = [Action(answer=f"{error_class}: {str(E)}", action_string=f"{error_class}: {str(E)}")]
        return actions
    
    def post_process(self, action_str):
        """
        从 LLM 生成的字符串中提取 JSON 格式的答案。
        """
        match = []
        try:
            # 使用正则匹配 JSON 代码块或普通 JSON 对象
            match = re.findall(r'```json\s*({(?:[^`]|\`(?!``))*})', action_str, re.DOTALL)
            match = match if len(match) > 0 else re.findall(r'{.*?}', action_str, re.DOTALL)
            # 解析并提取 predicted_answer 字段
            output = eval(re.sub(r' //.*', '', match[-1].replace("null", "None"))) # 移除注释并解析
            answer = output["predicted_answer"]
            action = Action(answer=str(answer), action_string=action_str)
        except:
            # 解析失败时的兜底提取逻辑
            action = Action(answer=action_str.split("predicted_answer")[-1], action_string=action_str)
        return action
    
    def apply_template(self, state):
        """
        构建 LLM 的输入消息。
        """
        question = state.question
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.render(question=question)}
        ]
        return messages