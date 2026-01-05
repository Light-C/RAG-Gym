"""
ReSearchAgent 实现。
该 Agent 模仿 Self-RAG 或类似的研究型工作流。它在生成回答的过程中会识别“未验证的陈述”（Unverified Claims），
并针对这些陈述发起检索（Research），直到回答被充分证实或达到轮数上限。
"""
import re
import json
import torch
from liquid import Template
from liquid.template import BoundTemplate
from rag_gym import State, LLMEngine, Action, BaseAgent

# ReSearch 默认系统提示词
research_system_prompt = '''You are a helpful assistant. Your task is to answer a question following user instructions.'''

# ReSearch 默认用户模板，增加了对“未验证陈述”的识别步骤
research_user_template = Template('''### Information-seeking History
{{history}}

### Original Question
{{question}}

Your output must include three sections:
1. **### Step-by-step Reasoning**:
  - Think step-by-step and then answer the question.

2. **### Unverified Claim Identification**:
  - Identify if there are claims in the step-by-step reasoning section that are not grounded in the information-seeking history section.
  - If yes, summarize the first piece of missing information as a atomic query to search in an external knowledge.
  - If no, clearly state that no further query is needed.

3. **### Structured Output**:
  - Present your predicted answer and generated query in the following JSON format:
    ```json
    {
        "predicted_answer": "Provide a single letter (for multiple-choice questions), digit, word, or short phrase here.",
        "generated_query": "Provide an entity, question, or statement to search in an external knowledge base. Output \"None\" if no query is generated.",
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

class ReSearchAgent(BaseAgent):
    """
    ReSearchAgent 类。
    核心特点是它在生成回答的同时会自发地寻找证据缺口。
    """
    def __init__(
        self, 
        llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        rag_llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        system_prompt: str | None = None, 
        user_template: BoundTemplate | None = None, 
        cache_dir: str | None = None, 
        api: bool = False, 
        model_dtype: torch.dtype = torch.bfloat16,
        rag: bool = True,
        reward_llm_name: str | None = None,
        train_mode: bool = False,
        **kwargs
    ):
        """
        初始化 ReSearchAgent。
        """
        super().__init__(llm_name, cache_dir, api, model_dtype, reward_llm_name, train_mode)
        self.agent_type = "research"
        self.system_prompt = system_prompt if system_prompt is not None else research_system_prompt
        self.user_template = user_template if user_template is not None else research_user_template
        
        # 实例化内部 RAG 模块用于预处理检索文档
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
        生成 Research 动作。
        逻辑流程与 Searcho1 类似，但模板要求模型识别 Unverified Claim。
        """
        action_strings = []
        try:
            question = state.question
            history = state.history.return_as_json(return_documents=True).copy()
            if self.rag_module is None:
                history = "\n\n".join([f"Query: {item['query']}\n" + "\n".join([f"Document [{idx}] (Title: {doc['title']}) {doc['content']}" for idx, doc in enumerate(item["documents"])]) for item in history])
            else:
                # 使用 RAGModule 总结每一轮检索
                for i in range(len(history)):
                    query = history[i]["query"]
                    documents = history[i]["documents"]
                    answer = self.rag_module(
                        query = query, 
                        documents = documents,
                        context = f"Original question:\n{question}",
                        max_new_tokens = max_new_tokens,
                        temperature = 0.0,
                        num_return_sequences = 1,
                    )
                    history[i]["answer"] = answer
                history = "\n\n".join([f"Query: {item['query']}\nAnswer: {item['answer']}" for item in history])
            
            # 生成包含推理、未验证陈述识别和最终 JSON 的响应
            action_strings = self.llm.generate(
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_template.render(question=question, history=history)}
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
        后处理。从 JSON 中提取 'predicted_answer' 和 'generated_query'。
        """
        match = []
        try:
            match = re.findall(r'```json\s*({(?:[^`]|\`(?!``))*})', action_str.split("### Structured Output")[-1], re.DOTALL)
            match = match if len(match) > 0 else re.findall(r'{.*?}', action_str.split("### Structured Output")[-1], re.DOTALL)
            output = eval(re.sub(r' //.*', '', match[-1].replace("null", "None").replace("\'None\'", "None").replace("\"None\"", "None"))) # remove comments
            query = output.get("query", output.get("generated_query", output.get("generated_queries", None)))
            answer = output.get("predicted_answer", None)
            
            if type(query) == list:
                query = query[0] if len(query) > 0 else None
            if query:
                action = Action(query=str(query), answer=answer, action_string=action_str)
            else:
                # 否则认为任务结束
                action = Action(answer=str(answer), action_string=action_str)
        except:
            action = Action(query=action_str.split("### Structured Output")[-1].strip(), action_string=action_str)
        return action

    def apply_template(self, state, qa_cache):
        """
        构造对话历史消息。
        """
        question = state.question
        history = "\n\n".join([f"Query: {item['query']}\nAnswer: {qa_cache[item['query']]}" for item in state.history])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.render(question=question, history=history)}
        ]
        return messages
    
class RAGModule:
    """
    RAGModule 类。
    封装了 LLM 调用，专门用于基于文档回答子查询。
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
        初始化 RAGModule。
        """
        self.llm_name = llm_name
        self.cache_dir = cache_dir
        self.api = api
        self.lora = True if "lora" in llm_name.lower() else False
        self.model_dtype = model_dtype
        # 实例化 LLM 引擎
        self.llm = LLMEngine(llm_name=self.llm_name, cache_dir=self.cache_dir, api=self.api, lora=self.lora, model_dtype=self.model_dtype, **kwargs)
        self.system_prompt = system_prompt if system_prompt is not None else rag_system_prompt
        self.user_template = user_template if user_template is not None else rag_user_template
        # 缓存机制，避免对相同的 query 进行重复 RAG
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
        """
        执行 RAG 过程。
        Args:
            query: 需要回答的问题。
            documents: 检索到的文档列表。
            context: 额外背景（如原始大问题）。
        """
        # 如果背景变了，清空缓存
        if context != self.context_cache:
            self.qa_cache = {}
            self.context_cache = context
            
        # 检查缓存
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
                # 调用 LLM 生成针对子问题的回答
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