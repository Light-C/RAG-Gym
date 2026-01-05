"""
Searcho1Agent 实现。
该 Agent 模仿 o1 的思考方式，在每一轮决策前，会先利用 RAGModule 对历史检索到的文档进行摘要/回答，
然后再将“问题-回答”对作为历史背景，决定下一步是继续 Search 还是 Finish。
"""
import re
import json
import torch
from liquid import Template
from liquid.template import BoundTemplate
from rag_gym import State, LLMEngine, Action, BaseAgent

# Searcho1 默认系统提示词
search_o1_system_prompt = '''You are a helpful assistant. Your task is to think step-by-step and take an action to help solve a given question.'''

# Searcho1 默认用户模板，强调基于历史回答（history）进行推理
search_o1_user_template = Template('''### Information-seeking History
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

# 截断后的模板，强制结束
search_o1_truncated_user_template = Template('''### Information-seeking History
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

# RAGModule 内部使用的系统提示词
rag_system_prompt = "You are a helpful assistant tasked with answering a follow-up query using the relevant documents provided."

# RAGModule 内部使用的用户模板
rag_user_template = Template('''### Relevant Documents
{{documents}}

### Context
{{context}}

### Follow-up Query
{{query}}

Answer the follow-up query succinctly, using only the information from the documents. When the documents do not provide sufficient information, explicitly point this out instead of making up facts. Do not include unrelated or excessive details in the response.''')

class Searcho1Agent(BaseAgent):
    """
    Searcho1Agent 类。
    与普通 ReAct 不同，它在内部维护了一个 RAGModule 来预处理检索到的文档。
    """
    def __init__(
        self, 
        llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        rag_llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        system_prompt: str | None = None, 
        user_template: BoundTemplate | None = None, 
        truncated_user_template: BoundTemplate | None = None, 
        cache_dir: str | None = None, 
        api: bool = False, 
        model_dtype: torch.dtype = torch.bfloat16,
        rag: bool = True,
        reward_llm_name: str | None = None,
        train_mode: bool = False,
        **kwargs
    ):
        """
        初始化 Searcho1Agent。
        Args:
            rag: 是否启用 RAGModule。
            rag_llm_name: RAGModule 使用的模型名称。
        """
        super().__init__(llm_name, cache_dir, api, model_dtype, reward_llm_name, train_mode)
        self.agent_type = "search_o1"
        self.system_prompt = system_prompt if system_prompt is not None else search_o1_system_prompt
        self.user_template = user_template if user_template is not None else search_o1_user_template
        self.truncated_user_template = truncated_user_template if truncated_user_template is not None else search_o1_truncated_user_template
        
        # 如果启用 RAG，则实例化 RAGModule
        if rag:
            self.rag_llm_name = rag_llm_name
            self.rag_module = RAGModule(
                llm_name = rag_llm_name, 
                system_prompt = kwargs.get("rag_system_prompt", None),
                user_template = kwargs.get("rag_user_template", None),
                cache_dir = cache_dir,
                api = api,
                model_dtype = model_dtype,
                shared_checkpoint = False if self.train_mode else True
            )
        else:
            self.rag_llm_name = None
            self.rag_module = None

    def generate_action(
        self, 
        state: State, 
        max_new_tokens: int = 2048, 
        temperature: float = 0.0, 
        num_actions: int = 1,
        **kwargs
    ):
        """
        生成决策动作。
        核心逻辑：
        1. 遍历检索历史中的每一轮。
        2. 调用 RAGModule 对该轮检索到的文档进行总结回答。
        3. 将总结后的 "Query-Answer" 历史喂给决策模型。
        """
        action_strings = []
        try:
            question = state.question
            # 获取检索历史
            history = state.history.return_as_json(return_documents=True).copy()
            
            if self.rag_module is None:
                # 如果没有 RAGModule，退化为普通文档拼接格式
                history = "\n\n".join([f"Query: {item['query']}\n" + "\n".join([f"Document [{idx}] (Title: {doc['title']}) {doc['content']}" for idx, doc in enumerate(item["documents"])]) for item in history])
            else:
                # 对历史中的每一轮进行 RAG 总结
                for i in range(len(history)):
                    query = history[i]["query"]
                    documents = history[i]["documents"]
                    # import pdb; pdb.set_trace()
                    answer = self.rag_module(
                        query = query, 
                        documents = documents,
                        context = f"Original question:\n{question}",
                        max_new_tokens = max_new_tokens,
                        temperature = 0.0,
                        num_return_sequences = 1,
                    )
                    history[i]["answer"] = answer
                # 构造简化后的历史字符串
                history = "\n\n".join([f"Query: {item['query']}\nAnswer: {item['answer']}" for item in history])
            
            # 决策模型生成下一步动作
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
                action = self.post_process(action_str)
                actions.append(action)
        except Exception as E:
            error_class = E.__class__.__name__
            actions = [Action(action_string=f"{error_class}: {str(E)}")]
        return actions
    
    def post_process(self, action_str):
        """
        解析决策模型的输出。
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
            
            if query:
                action = Action(query=str(query), action_string=action_str)
            else:
                action = Action(answer=str(answer), action_string=action_str)
        except:
            action = Action(query=action_str.split("### Structured Output")[-1].strip(), action_string=action_str)
        return action

    def apply_template(self, state, qa_cache):
        """
        应用模板，主要用于打分或重新推理。
        """
        question = state.question
        # 利用外部提供的 qa_cache (Query-Answer 缓存) 构造历史
        history = "\n\n".join([f"Query: {item['query']}\nAnswer: {qa_cache[item['query']]}" for item in state.history])
        # history = "\n\n".join([f"Query: {item['query']}\nAnswer: {qa_cache[item['query']][-1]}" for item in state.history])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.render(question=question, history=history)}
        ]
        return messages
    
class RAGModule:
    """
    RAGModule 类。
    专门用于接收 Query 和一组 Documents，然后生成针对该 Query 的回答。
    """
    def __init__(
        self, 
        llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        system_prompt: str | None = None, 
        user_template: BoundTemplate | None = None,
        cache_dir: str | None = None, 
        api: bool = False, 
        model_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ):
        """
        初始化 RAG 模块。
        """
        self.llm_name = llm_name
        self.cache_dir = cache_dir
        self.api = api
        self.lora = True if "lora" in llm_name.lower() else False
        self.model_dtype = model_dtype
        self.llm = LLMEngine(llm_name=self.llm_name, cache_dir=self.cache_dir, api=self.api, lora=self.lora, model_dtype=self.model_dtype, **kwargs)
        self.system_prompt = system_prompt if system_prompt is not None else rag_system_prompt
        self.user_template = user_template if user_template is not None else rag_user_template
        self.qa_cache = {}
        self.context_cache = ""

    def __call__(
        self, 
        query: str, 
        documents: list,
        context: str = "", 
        max_new_tokens: int = 2048, 
        temperature: float = 0.0, 
        num_return_sequences: int = 1, 
        messages: list[dict[str, str]] | None = None,
        **kwargs
    ):
        if context != self.context_cache:
            self.qa_cache = {}
            self.context_cache = context
        if query in self.qa_cache and "Error:" not in self.qa_cache[query]:
            answer = self.qa_cache[query]
        else:
            if messages is None:
                documents = '\n'.join(["(Title: {:s}) {:s}".format(documents[idx]["title"], documents[idx]["content"]) for idx in range(len(documents))])
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_template.render(documents=documents, context=context, query=query)}
                ]
            try:
                answer = self.llm.generate(
                    messages = messages, 
                    max_new_tokens = max_new_tokens, 
                    temperature = temperature, 
                    num_return_sequences = num_return_sequences,
                    **kwargs
                )
            except Exception as E:
                error_class = E.__class__.__name__
                answer = [f"{error_class}: {str(E)}"]
            self.qa_cache[query] = answer
        if num_return_sequences == 1:
            answer = answer[-1]
        return answer