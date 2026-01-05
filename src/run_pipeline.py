"""
    RAG-Gym 流水线运行脚本。
    该脚本负责初始化 Agent 和环境，遍历数据集，并记录 Agent 在环境中的交互过程。
    支持多种 Agent 类型（Direct, CoT, RAG, ReAct, Search-o1, ReSearch）和不同的检索配置。
"""
import os
import re
import time
import tqdm
import json
import argparse
from liquid import Template
import sys
sys.path.append(".")
import rag_gym
from src.data_loader import KIQA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 RAG-Gym 评估流水线")
    # Agent 相关参数
    parser.add_argument("--agent_type", type=str, default="direct", help="Agent 类型: direct, cot, rag, react, search_o1, research")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="主语言模型名称")
    parser.add_argument("--rag_llm_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="RAG 模块使用的辅助 LLM 名称")
    parser.add_argument("--reward_llm_name", type=str, default=None, help="奖励模型 (RM) 名称，若不提供则不进行重排序/打分")
    
    # 检索与语料库参数
    parser.add_argument("--retriever_name", type=str, default=None, help="检索器名称")
    parser.add_argument("--corpus_name", type=str, default=None, help="语料库名称")
    parser.add_argument("--k", type=int, default=32, help="每次检索返回的文档数量")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF 融合时的参数 k")
    
    # 任务与数据参数
    parser.add_argument("--data", type=str, default="medqa", help="数据集名称 (如 medqa, hotpotqa)")
    parser.add_argument("--save_dir", type=str, default="./predictions", help="结果保存目录")
    parser.add_argument("--max_iterations", type=int, default=10, help="最大交互轮数")
    
    # 采样与生成参数
    parser.add_argument("--n_actions", type=int, default=10, help="每轮生成的候选动作数量 (配合 reward_llm 使用)")
    parser.add_argument("--temperature", type=float, default=1.0, help="生成动作时的采样温度")
    parser.add_argument("--cache_dir", type=str, default="../huggingface/hub", help="模型缓存目录")
    
    # 并行/切片运行参数
    parser.add_argument("--n", type=int, default=1, help="总分片数，用于并行运行")
    parser.add_argument("--i", type=int, default=0, help="当前分片索引")
    parser.add_argument("--api", action=argparse.BooleanOptionalAction, help="是否使用 API 调用模型")
    args = parser.parse_args()

    # --- 参数初始化与目录设置 ---
    agent_type = args.agent_type
    llm_name = args.llm_name
    rag_llm_name = args.rag_llm_name
    reward_llm_name = args.reward_llm_name
    retriever_name = args.retriever_name
    corpus_name = args.corpus_name
    data = args.data
    save_dir = args.save_dir
    k = args.k
    rrf_k = args.rrf_k
    max_iterations = args.max_iterations
    # 如果没有 reward_llm，则只生成 1 个动作，温度设为 0
    n_actions = args.n_actions if reward_llm_name else 1
    temperature = args.temperature if reward_llm_name else 0.0
    cache_dir = args.cache_dir
    api = False if args.api is None else True
    
    # 构建保存目录路径，包含模型名、Agent类型、迭代次数等信息
    save_dir = os.path.join(save_dir, data, agent_type, '_'.join([name.replace('/', '_') for name in [llm_name, reward_llm_name] if name is not None]), f"{max_iterations}" if n_actions == 1 else f"{max_iterations}_{n_actions}_{temperature}")

    # 加载数据集并进行切片（如果需要并行运行）
    dataset = KIQA(data)
    curr_range = [j for j in range(len(dataset)) if j % args.n == args.i]

    # --- 检索器与语料库默认配置 ---
    if agent_type in ["rag", "react", "search_o1", "research"]:
        # 根据数据集设置默认的检索器和语料库
        retriever_name = retriever_name if retriever_name else "RRF-2" if data == "medqa" else "RRF-BGE"
        corpus_name = corpus_name if corpus_name else "MedText" if data == "medqa" else "Wikipedia_HotpotQA" if data == "hotpotqa" else "Wikipedia"
        
        # 构建检索相关的保存路径后缀
        if "rrf" in retriever_name.lower():
            retrieval_suffix = f"{retriever_name}_{corpus_name}_{k}_{rrf_k}"
        else:
            retrieval_suffix = f"{retriever_name}_{corpus_name}_{k}"
        save_dir = os.path.join(save_dir, retrieval_suffix)
    os.makedirs(save_dir, exist_ok=True)

    # --- Agent 初始化 ---
    type2class = {
        "direct": rag_gym.DirectAgent,
        "cot": rag_gym.CoTAgent,
        "rag": rag_gym.RAGAgent,
        "react": rag_gym.ReActAgent,
        "search_o1": rag_gym.Searcho1Agent,
        "research": rag_gym.ReSearchAgent,
    }
    kwargs = {}
    if agent_type in ["search_o1", "research"]:
        kwargs = {"rag_llm_name": rag_llm_name}

    agent = type2class[agent_type](
        llm_name = llm_name,
        api = api,
        cache_dir = cache_dir,
        reward_llm_name = reward_llm_name,
        **kwargs
    )

    # 部分 Agent 类型的最大迭代次数需要微调
    max_iterations = max_iterations - 1 if agent_type in ["react", "search_o1"] else max_iterations

    # --- 环境 (Environment) 初始化 ---
    env = rag_gym.make(
        retriever_name = retriever_name,
        corpus_name = corpus_name,
        max_iter = max_iterations,
        k = k,
        rrf_k = rrf_k,
        cache = True,   # 启用缓存以加速加载
        HNSW = True     # 启用 HNSW 索引以加速检索
    )

    # --- 主循环：遍历数据集 ---
    for idx in tqdm.tqdm(curr_range):
        item = dataset[idx]
        # 获取样本 ID
        index = item.get('_id', item.get('index', item.get('id', str(idx))))
        
        # 检查是否已存在预测结果，支持断点续传
        try:
            existing_pred = json.load(open(os.path.join(save_dir, f"{index}.json")))['prediction']
            if "Error:" not in open(os.path.join(save_dir, f"{index}.json")).read():
                # 如果历史记录大于 1 轮，则认为已完成
                if len(json.load(open(os.path.join(save_dir, f"{index}.json")))["history"]) > 1:
                    continue
        except:
            pass

        # 准备问题文本（包含选项）
        question = f"{item['question']} {item.get('options', '')}".strip()
        answer = item.get("answer", None)

        history = []
        action_cache = []
        
        # 重置环境，开始新的交互
        observation, info = env.reset(
            question = question,
        )
        history.append({"type": "state", "content": observation.return_as_json()})
        
        # 如果是搜索类 Agent，重置其内部的 QA 缓存
        if agent_type in ["search_o1", "research"]:
            agent.rag_module.qa_cache = {}

        # 迭代交互过程
        for _ in range(max_iterations):
            
            # --- 动作生成 (Action Generation) ---
            if n_actions == 1 or (agent_type == "rag" and env.curr_iter == 0):
                # 单动作模式：直接生成贪婪预测的动作
                action = agent.generate_action(
                    state = observation,
                    temperature = 0.0,
                )[0]
            else:
                # 多动作采样与打分模式
                actions = agent.generate_action(
                    state = observation,
                    temperature = temperature,
                    num_actions = n_actions
                )
                # 使用奖励模型对动作进行评分
                if agent_type in ["search_o1", "research"]:
                    rewards = agent.score(observation, actions, qa_cache=agent.rag_module.qa_cache)
                else:
                    rewards = agent.score(observation, actions)
                assert len(rewards) == len(actions)
                
                # 记录所有候选动作及其得分
                action_cache.append(
                    {
                        "curr_iter": env.curr_iter,
                        "actions": [
                            {
                                "content": actions[act_idx].return_as_json(),
                                "string": actions[act_idx].return_as_string(),
                                "reward": reward
                            } for act_idx, reward in enumerate(rewards)
                        ]
                    }
                )
                # 选择得分最高的动作
                action = actions[rewards.index(max(rewards))]

            # --- 执行动作与状态更新 ---
            history.append({"type": "action", "content": action.return_as_json(), "string": action.return_as_string()})
            observation, reward, terminated, truncated, info = env.step(action)
            history.append({"type": "state", "content": observation.return_as_json()})
            
            # 实时保存中间结果
            with open(os.path.join(save_dir, f"{index}.json"), "w") as f:
                json.dump({**item, "history": history, "action_cache": action_cache}, f, indent=4)
            
            # 保存检索缓存 (QD Cache)
            if agent_type in ["rag", "react", "search_o1", "research"]:
                with open(os.path.join(save_dir, f"{index}_qd_cache.json"), "w") as f:
                    json.dump({qd["query"]:qd["documents"] for qd in observation.history.qd_list}, f, indent=4)
            
            # 保存 QA 缓存 (QA Cache)
            if agent_type in ["search_o1", "research"]:
                with open(os.path.join(save_dir, f"{index}_qa_cache.json"), "w") as f:
                    json.dump(agent.rag_module.qa_cache, f, indent=4)
            
            if terminated:
                break
            
            # 处理截断情况（达到最大迭代次数前的收尾工作）
            if truncated:
                if agent_type in ["react", "search_o1"]:
                    # 生成最后的回答动作
                    if n_actions == 1:
                        action = agent.generate_action(
                            state = observation,
                            temperature = 0.0,
                        )[0]
                    else:
                        actions = agent.generate_action(
                            state = observation,
                            temperature = temperature,
                            num_actions = n_actions
                        )
                        if agent_type == "search_o1":
                            rewards = agent.score(observation, actions, qa_cache=agent.rag_module.qa_cache)
                        else:
                            rewards = agent.score(observation, actions)
                        assert len(rewards) == len(actions)
                        action_cache.append(
                            {
                                "curr_iter": env.curr_iter,
                                "actions": [
                                    {
                                        "content": actions[act_idx].return_as_json(),
                                        "string": actions[act_idx].return_as_string(),
                                        "reward": reward
                                    } for act_idx, reward in enumerate(rewards)
                                ]
                            }
                        )
                        action = actions[rewards.index(max(rewards))]
                    history.append({"type": "action", "content": action.return_as_json(), "string": action.return_as_string()})
                    observation, reward, terminated, truncated, info = env.step(action)
                    history.append({"type": "state", "content": observation.return_as_json()})
                break
        
        # 保存最终预测结果
        item["prediction"] = observation.answer
        item["history"] = history
        item["action_cache"] = action_cache

        with open(os.path.join(save_dir, f"{index}.json"), "w") as f:
            json.dump(item, f, indent=4)