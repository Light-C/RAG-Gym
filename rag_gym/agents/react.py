"""
ReActAgent (Reasoning and Acting Agent) 实现。
该 Agent 结合了推理（Reasoning）和行动（Acting），可以根据历史信息决定是继续检索（Search）还是结束并回答（Finish）。
"""
import re
import json
import torch
from liquid import Template
from liquid.template import BoundTemplate
from rag_gym import State, LLMEngine, Action, BaseAgent

# ReAct 默认系统提示词
react_system_prompt = '''You are a helpful assistant. Your task is to think step-by-step and take an action to help solve a given question.'''

# ReAct 默认用户模板，支持多轮对话历史
react_user_template = Template('''### Information-seeking History
{{history}}

### Original Question
{{question}}

Your output must include two sections:
1. **### Step-by-step Reasoning**:
  - Think step-by-step and reason about the current situation.
  - Take an action for the next step which can be two types:
    - Search[query], which searches the exact query in an external knowledge base. Avoid duplicating queries already asked in the history.
    - Finish[answer], which returns the answer and finishes the task.

2. **### Structured Output**:
  - If the next action is `Search`, present your generated query in the following JSON format:
    ```json
    {
        "generated_query": "Provide an entity, question, or statement to be searched in an external knowledge base.",
    }
    ```
  - If the next action is `Finish`, present your predicted answer in the following JSON format:
    ```json
    {
        "predicted_answer": "Provide a single letter (for multiple-choice questions), digit, word, or short phrase here."
    }
    ```''')

# 截断后的用户模板，当达到最大轮数时强制模型结束任务
react_truncated_user_template = Template('''### Information-seeking History
{{history}}

### Original Question
{{question}}

Your output must include two sections:
1. **### Step-by-step Reasoning**:
  - Think step-by-step and reason about the current situation.
  - Take an action for the next step which can be one type:
    - Finish[answer], which returns the answer and finishes the task.

2. **### Structured Output**:
  - Present your predicted answer in the following JSON format:
    ```json
    {
        "predicted_answer": "Provide a single letter (for multiple-choice questions), digit, word, or short phrase here."
    }
    ```''')

class ReActAgent(BaseAgent):
    """
    ReActAgent 类，实现推理与行动的交替。
    """
    def __init__(
        self, 
        llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        system_prompt: str | None = None, 
        user_template: BoundTemplate | None = None, 
        truncated_user_template: BoundTemplate | None = None, 
        cache_dir: str | None = None, 
        api: bool = False, 
        model_dtype: torch.dtype = torch.bfloat16,
        reward_llm_name: str | None = None,
        train_mode: bool = False,
        **kwargs
    ):
        """
        初始化 ReActAgent。
        Args:
            llm_name: LLM 模型路径。
            system_prompt: 自定义系统提示词。
            user_template: 标准用户模板。
            truncated_user_template: 强制结束时的模板。
            cache_dir: 模型缓存。
            api: 是否使用 API。
            model_dtype: 精度。
            reward_llm_name: 奖励模型。
            train_mode: 是否训练。
        """
        super().__init__(llm_name, cache_dir, api, model_dtype, reward_llm_name, train_mode)
        self.agent_type = "react"
        self.system_prompt = system_prompt if system_prompt is not None else react_system_prompt
        self.user_template = user_template if user_template is not None else react_user_template
        self.truncated_user_template = truncated_user_template if truncated_user_template is not None else react_truncated_user_template

    def generate_action(
        self, 
        state: State, 
        max_new_tokens: int = 2048, 
        temperature: float = 0.0, 
        num_actions: int = 1,
        **kwargs
    ):
        """
        生成推理与行动动作。
        Args:
            state: 当前状态，包含已有的检索历史。
            max_new_tokens: 最大生成长度。
            temperature: 温度。
            num_actions: 生成动作数量。
        Returns:
            list[Action]: 解析后的动作列表。
        """
        # 将历史检索记录及其文档格式化为字符串
        history = "\n\n".join([f"Query: {item['query']}\n" + "\n".join([f"Document [{idx}] (Title: {doc['title']}) {doc['content']}" for idx, doc in enumerate(item["documents"])]) for item in state.history])
        action_strings = []
        try:
            question = state.question
            # 根据 state.truncated 决定使用哪种模板
            action_strings = self.llm.generate(
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_template.render(question=question, history=history) if not state.truncated else self.truncated_user_template.render(question=question, history=history)}
                ], 
                max_new_tokens = max_new_tokens, 
                temperature = temperature, 
                num_return_sequences = num_actions, 
                **kwargs
            )
            actions = []
            for action_str in action_strings:
                assert type(action_str) is str
                # 对 LLM 生成的字符串进行后处理，判断是 Search 还是 Finish
                action = self.post_process(action_str)
                actions.append(action)
        except Exception as E:
            error_class = E.__class__.__name__
            actions = [Action(action_string=f"{error_class}: {str(E)}")]
        return actions
    
    def post_process(self, action_str):
        """
        后处理 LLM 生成的 ReAct 响应。
        解析 JSON 内容，区分 'generated_query' (继续搜索) 和 'predicted_answer' (完成回答)。
        """
        match = []
        try:
            match = re.findall(r'```json\s*({(?:[^`]|\`(?!``))*})', action_str.split("### Structured Output")[-1], re.DOTALL)
            match = match if len(match) > 0 else re.findall(r'{.*?}', action_str.split("### Structured Output")[-1], re.DOTALL)
            output = eval(re.sub(r' //.*', '', match[-1].replace("null", "None"))) # remove comments
            query = output.get("query", output.get("generated_query", output.get("generated_queries", None)))
            answer = output.get("predicted_answer", None)
            
            if type(query) == list:
                query = query[0]
            query = str(query) if query is not None else query
            
            # 如果存在 query，则该动作是 Search
            if query:
                action = Action(query=str(query), action_string=action_str)
            else:
                # 否则该动作是 Finish
                action = Action(answer=str(answer), action_string=action_str)
        except:
            # 兜底：如果解析失败，将整个结构化部分作为 query（可能是 Search[query] 这种直接格式）
            action = Action(query=action_str.split("### Structured Output")[-1].strip(), action_string=action_str)
        return action

    def apply_template(self, state):
        """
        构造 ReAct 风格的消息。
        """
        question = state.question
        history = "\n\n".join([f"Query: {item['query']}\n" + "\n".join([f"Document [{idx}] (Title: {doc['title']}) {doc['content']}" for idx, doc in enumerate(item["documents"])]) for item in state.history])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.render(question=question, history=history)}
        ]
        return messages