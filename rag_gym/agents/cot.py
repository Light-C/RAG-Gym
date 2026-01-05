"""
CoTAgent (Chain of Thought Agent) 实现。
该 Agent 引导 LLM 进行逐步推理（Step-by-step Reasoning）后再给出最终答案。
"""
import re
import json
import torch
from liquid import Template
from liquid.template import BoundTemplate
from rag_gym import State, LLMEngine, Action, BaseAgent

# CoT 默认系统提示词，要求模型逐步思考
cot_system_prompt = '''You are a helpful assistant. Your task is to think step-by-step and answer a given question.'''

# CoT 默认用户模板，定义了输出格式要求
cot_user_template = Template('''### Question
{{question}}

Your output must include two sections:
1. **### Step-by-step Reasoning**:
  - Think step-by-step and then answer the question.

2. **### Structured Output**:
  - Present your predicted answer in the following JSON format:
  ```json
  {
    "predicted_answer": "Provide a single letter (for multiple-choice questions), digit, word, or short phrase here."
  }
  ```''')

class CoTAgent(BaseAgent):
    """
    CoTAgent 类，继承自 BaseAgent。
    通过 Chain of Thought (思维链) 策略来增强 LLM 的推理能力。
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
        """
        初始化 CoTAgent。
        Args:
            llm_name: 模型名称或路径。
            system_prompt: 自定义系统提示词。
            user_template: 自定义用户模板。
            cache_dir: 模型缓存目录。
            api: 是否使用 API 调用。
            model_dtype: 模型权重类型。
            reward_llm_name: 奖励模型名称（可选）。
            train_mode: 是否处于训练模式。
        """
        super().__init__(llm_name, cache_dir, api, model_dtype, reward_llm_name, train_mode)
        self.agent_type = "cot"
        self.system_prompt = system_prompt if system_prompt is not None else cot_system_prompt
        self.user_template = user_template if user_template is not None else cot_user_template

    def generate_action(
        self, 
        state: State, 
        max_new_tokens: int = 2048, 
        temperature: float = 0.0, 
        num_actions: int = 1,
        **kwargs
    ):
        """
        生成 CoT 推理动作。
        Args:
            state: 当前环境状态（包含问题）。
            max_new_tokens: 最大生成 token 数。
            temperature: 采样温度。
            num_actions: 生成候选动作的数量。
        Returns:
            list[Action]: 包含推理过程和预测答案的动作列表。
        """
        action_strings = []
        try:
            question = state.question
            # 调用 LLM 生成回复
            action_strings = self.llm.generate(
                messages = self.apply_template(state), 
                max_new_tokens = max_new_tokens, 
                temperature = temperature, 
                num_return_sequences = num_actions, 
                **kwargs
            )
            actions = []
            for action_str in action_strings:
                # 对生成的字符串进行后处理，提取答案
                action = self.post_process(action_str)
                actions.append(action)
        except Exception as E:
            # 异常处理，返回错误信息
            error_class = E.__class__.__name__
            actions = [Action(answer=f"{error_class}: {str(E)}", action_string=f"{error_class}: {str(E)}")]
        return actions
    
    def post_process(self, action_str):
        """
        从模型生成的 CoT 文本中提取结构化答案。
        主要查找 "### Structured Output" 之后的 JSON 块。
        """
        match = []
        try:
            match = re.findall(r'```json\s*({(?:[^`]|\`(?!``))*})', action_str.split("### Structured Output")[-1], re.DOTALL)
            match = match if len(match) > 0 else re.findall(r'{.*?}', action_str.split("### Structured Output")[-1], re.DOTALL)
            output = eval(re.sub(r' //.*', '', match[-1].replace("null", "None"))) # remove comments
            answer = output["predicted_answer"]
            action = Action(answer=str(answer), action_string=action_str)
        except:
            # 兜底逻辑：如果正则提取失败，尝试根据关键字切割
            action = Action(answer=action_str.split("predicted_answer")[-1], action_string=action_str)
        return action

    def apply_template(self, state):
        """
        将当前状态应用到模板中，构造对话消息。
        """
        question = state.question
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.render(question=question)}
        ]
        return messages