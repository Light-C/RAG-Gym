import json
import torch
from rag_gym import State, LLMEngine, Action
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BaseAgent:
    """
    Agent 的基类。
    提供了 LLM 引擎的初始化、奖励模型 (Reward Model) 的加载以及对生成动作进行打分的功能。
    """
    def __init__(
        self, 
        llm_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        cache_dir: str | None = None, 
        api: bool = False, 
        model_dtype: torch.dtype = torch.bfloat16,
        reward_llm_name: str | None = None,
        train_mode: bool = False
    ):
        """
        初始化基类 Agent。

        Args:
            llm_name: 主语言模型名称或路径。
            cache_dir: 模型缓存目录。
            api: 是否通过 API 调用模型。
            model_dtype: 模型权重数据类型。
            reward_llm_name: 奖励模型名称或路径（可选）。
            train_mode: 是否处于训练模式。
        """
        self.llm_name = llm_name
        self.cache_dir = cache_dir
        self.api = api
        self.model_dtype = model_dtype
        # 初始化主语言模型引擎
        self.llm = LLMEngine(llm_name=self.llm_name, cache_dir=self.cache_dir, api=self.api, model_dtype=self.model_dtype)
        self.reward_llm_name = reward_llm_name
        self.reward_model = None
        self.reward_tokenizer = None
        self.train_mode = train_mode
        
        # 如果指定了奖励模型，则进行加载和初始化
        if self.reward_llm_name:
            from peft import PeftModel
            # 加载分类模型作为奖励模型的基础
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_llm_name, 
                device_map = "auto", 
                cache_dir = cache_dir, 
                torch_dtype = torch.bfloat16,
                num_labels = 1
            )
            # 加载 LoRA 权重并合并
            self.reward_model = PeftModel.from_pretrained(self.reward_model, reward_llm_name)
            self.reward_model = self.reward_model.merge_and_unload()
            
            # 初始化奖励模型的分词器
            self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_llm_name, cache_dir=cache_dir)
            self.reward_tokenizer.padding_side = 'left'
            # self.reward_tokenizer.padding_side = 'right'
            self.reward_tokenizer.truncation_side = 'left'
            
            # 处理 pad_token
            if self.reward_tokenizer.pad_token is None:
                self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
                # self.reward_tokenizer.pad_token = "<|end_of_text|>" # 128001
            if self.reward_model.config.pad_token_id is None:
                # self.reward_model.config.pad_token_id = self.reward_tokenizer.pad_token_id
                self.reward_model.config.pad_token_id = -1

    def generate_action(self, state: State, max_new_tokens: int, temperature: float, num_actions: int) -> list[Action]:
        """
        根据当前状态生成动作。子类必须实现此方法。
        """
        raise NotImplementedError
    
    def post_process(self, action_str: str) -> Action:
        """
        对 LLM 生成的字符串进行后处理。子类必须实现此方法。
        """
        raise NotImplementedError
    
    def apply_template(self, state: State) -> list[dict[str, str]]:
        """
        将当前状态应用到对话模板中。子类必须实现此方法。
        """
        raise NotImplementedError
    
    def score(self, state: State, actions: list[Action], max_length=4096, **kwargs):
        """
        使用奖励模型为生成的候选动作进行打分。
        
        Args:
            state: 当前状态。
            actions: 候选动作列表。
            max_length: 最大输入长度。
        
        Returns:
            list[float]: 对应每个动作的奖励分值列表。
        """
        assert self.reward_model is not None
        # 获取当前状态的消息模板
        messages = self.apply_template(state, **kwargs)
        # 将每个动作拼接到对话历史中
        all_messages = [messages + [{"role": "assistant", "content": act.action_string}] for act in actions]
        # 应用聊天模板并分词
        inputs = [self.reward_tokenizer.apply_chat_template(m, tokenize=False) for m in all_messages]
        inputs = self.reward_tokenizer(inputs, add_special_tokens=False, return_tensors="pt", padding="longest", truncation=True, max_length=max_length)
        # inputs = ["\n\n".join([f"{m['role']}: {m['content']}" for m in cm]) for cm in all_messages]
        # inputs = self.reward_tokenizer(inputs, return_tensors="pt", padding="longest", truncation=True, max_length=max_length)
        rewards = self.reward_model(
            input_ids=inputs["input_ids"].to(self.reward_model.device),
            attention_mask=inputs["attention_mask"].to(self.reward_model.device),
            return_dict=True,
        )["logits"]
        # 返回第一列（通常是标量奖励值）的列表
        return rewards[:,0].tolist()