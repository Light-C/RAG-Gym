"""
    RAG-Gym 环境类实现。
    模拟了一个交互式 RAG 环境，Agent 可以通过执行 Action（如检索或回答）来改变状态。
"""
from rag_gym import State, Action, RetrievalSystemCached, EMReward

class Env:
    """
    RAG 交互环境类。
    """
    def __init__(
        self, 
        retriever_name: str = None, 
        corpus_name: str = None, 
        max_iter: int = 5, 
        k: int = 32, 
        rrf_k: int = 60, 
        cache = True,
        **kwargs,
    ):
        """
        初始化环境。
        
        Args:
            retriever_name: 检索器模型名称。
            corpus_name: 语料库名称。
            max_iter: 最大交互轮数。
            k: 检索返回的文档数。
            rrf_k: RRF 融合参数。
            cache: 是否启用检索缓存。
        """
        self.max_iter = max_iter
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.k = k
        self.rrf_k = rrf_k
        self.state = None
        self.curr_iter = 0
        # 初始化检索系统
        if self.retriever_name is None or self.corpus_name is None:
            self.retrieval_system = None
        else:
            self.retrieval_system = RetrievalSystemCached(retriever_name=self.retriever_name, corpus_name=self.corpus_name, cache=cache, **kwargs)
        # 默认使用 Exact Match (EM) 奖励
        self.reward_model = EMReward()

    def info(self):
        """返回当前环境的配置信息"""
        return {
            "curr_iter": self.curr_iter,
            "max_iter": self.max_iter,
            "retriever_name": self.retriever_name,
            "corpus_name": self.corpus_name,
            "k": self.k,
            "rrf_k": self.rrf_k,
        }

    def reset(self, question: str, truth: str | None = None):
        """
        重置环境状态。
        
        Args:
            question: 新的问题文本。
            truth: 真实答案（用于计算奖励）。
        """
        self.curr_iter = 0
        self.state = State(question=question, truth=truth, terminated=False, truncated=self.curr_iter>=self.max_iter)        
        return self.state, self.info()

    def close(self):
        """清理环境，清除检索器缓存"""
        import rag_gym
        rag_gym.envs.utils.cached_retrievers.clear()

    def step(self, action: Action):
        """
        执行一个动作，推进环境状态。
        
        Args:
            action: Agent 生成的动作（检索查询或回答）。
            
        Returns:
            tuple: (next_state, reward, terminated, truncated, info)
        """
        # 状态转移
        next_state = self.transition(self.state, action)
        self.curr_iter += 1
        
        # 判断是否终止
        terminated = self.is_terminal(next_state)
        truncated = self.curr_iter >= self.max_iter
        next_state.terminated = terminated
        next_state.truncated = truncated
        
        # 计算奖励
        reward = self.get_reward(self.state, action, next_state)
        self.state = next_state
        return next_state, reward, terminated, truncated, self.info()

    def transition(self, state: State, action: Action):
        """
        实现状态转移逻辑。
        1. 如果 action 包含 query，则执行检索并将结果存入 history。
        2. 如果 action 包含 answer，则将其设为当前状态的预测答案。
        """
        history = state.history.copy()
        
        # 如果是回答动作，或者达到了最大迭代次数
        if action.query is None or self.curr_iter + 1 == self.max_iter:
            next_state = State(question=state.question, history=history, truth=state.truth, answer=action.answer)
        else:
            # 执行检索动作
            query = action.query
            assert self.retrieval_system is not None
            # 调用检索系统获取文档
            documents = self.retrieval_system.retrieve(query, k=self.k, rrf_k=self.rrf_k)[0]
            # 更新历史记录
            history.add_qd(query=query, documents=documents)
            next_state = State(question=state.question, history=history, truth=state.truth)
        return next_state

    def is_terminal(self, state):
        """判断状态是否达到终止条件（即已产出答案）"""
        if state.is_terminal():
            return True
        return False

    def get_reward(self, prev_state, action, curr_state):
        """计算当前步的奖励"""
        if curr_state.truth is None:
            return None
        return self.reward_model(prev_state, action, curr_state)

def make(**kwargs) -> Env:
    """创建环境的便捷函数"""
    return Env(**kwargs)