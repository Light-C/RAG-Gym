"""
    检索模块核心实现。
    主要包含：
    1. 自定义 SentenceTransformer 以支持特定的池化策略。
    2. 语料库向量化与索引构建功能。
    3. Retriever 类：封装单个语料库的检索逻辑。
    4. RetrievalSystem 类：管理多个检索器和语料库，支持 RRF (Reciprocal Rank Fusion) 融合检索。
    5. DocExtracter 类：根据 ID 快速提取文档内容。
    
    参考自: https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/utils.py
"""

from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
import os
import faiss
import json
import torch
import tqdm
import numpy as np
from openai import OpenAI
from rag_gym.config import config_local_embedding

class APIEmbeddingFunction:
    """
    通过 API 调用实现向量化的类，模拟 SentenceTransformer 的接口。
    """
    def __init__(self, model_name, base_url, api_key):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def encode(self, texts, batch_size=128, **kwargs):
        """
        将文本列表转换为向量列表，支持自动分批处理。
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # 调用 OpenAI 兼容接口
            response = self.client.embeddings.create(
                input=batch_texts,
                model=self.model_name
            )
            # 提取向量
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
        return np.array(all_embeddings)

    def eval(self):
        pass

# 语料库名称及其包含的子语料库映射
corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "MedText": ["textbooks", "statpearls"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
    "Wikipedia_HotpotQA": ["wiki_hotpotqa"],
}

# 检索模型名称及其对应的模型标识
retriever_names = {
    "BM25": ["bm25"],
    "Contriever": ["facebook/contriever"],
    "SPECTER": ["allenai/specter"],
    "MedCPT": ["ncbi/MedCPT-Query-Encoder"],
    "BGE": ["BAAI/bge-base-en-v1.5"],
    "BGE-M3": ["bge-m3"],
    "RRF-BGE": ["bm25", "BAAI/bge-base-en-v1.5"],
    "RRF-2": ["bm25", "ncbi/MedCPT-Query-Encoder"],
    "RRF-4": ["bm25", "facebook/contriever", "allenai/specter", "ncbi/MedCPT-Query-Encoder"]
}

def ends_with_ending_punctuation(s):
    """检查字符串是否以结束标点符号结尾"""
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    """将标题和内容拼接，确保标点符号正确"""
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

class CustomizeSentenceTransformer(SentenceTransformer): 
    """
    自定义 SentenceTransformer，将默认的均值池化 (MEAN) 改为 CLS 池化。
    """

    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        """
        创建一个简单的 Transformer + CLS 池化模型并返回模块。
        """
        print("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(model_name_or_path))
        token = kwargs.get('token', None)
        cache_folder = kwargs.get('cache_folder', None)
        revision = kwargs.get('revision', None)
        trust_remote_code = kwargs.get('trust_remote_code', False)
        if 'token' in kwargs or 'cache_folder' in kwargs or 'revision' in kwargs or 'trust_remote_code' in kwargs:
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
                tokenizer_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
            )
        else:
            transformer_model = Transformer(model_name_or_path)
        # 使用 CLS 池化
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]


def embed(chunk_dir, index_dir, model_name, **kwarg):
    """
    对语料库中的分块进行向量化。
    
    Args:
        chunk_dir: 语料库分块所在目录。
        index_dir: 索引存储目录。
        model_name: 使用的模型名称。
        **kwarg: 传递给 model.encode 的其他参数。
    
    Returns:
        int: 向量维度。
    """
    save_dir = os.path.join(index_dir, "embedding")
    
    if model_name == config_local_embedding.get("model_name"):
        model = APIEmbeddingFunction(
            model_name=config_local_embedding["model_name"],
            base_url=config_local_embedding["base_url"],
            api_key=config_local_embedding["api_key"]
        )
    elif "contriever" in model_name:
        model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = CustomizeSentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    fnames = sorted([fname for fname in os.listdir(chunk_dir) if fname.endswith(".jsonl")])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames):
            fpath = os.path.join(chunk_dir, fname)
            save_path = os.path.join(save_dir, fname.replace(".jsonl", ".npy"))
            if os.path.exists(save_path):
                continue
            if open(fpath).read().strip() == "":
                continue
            texts = [json.loads(item) for item in open(fpath).read().strip().split('\n')]
            # 根据模型要求处理输入文本
            if "specter" in model_name.lower():
                texts = [model.tokenizer.sep_token.join([item["title"], item["content"]]) for item in texts]
            elif "contriever" in model_name.lower():
                texts = [". ".join([item["title"], item["content"]]).replace('..', '.').replace("?.", "?") for item in texts]
            elif "medcpt" in model_name.lower():
                texts = [[item["title"], item["content"]] for item in texts]
            else:
                texts = [concat(item["title"], item["content"]) for item in texts]
            # 生成并保存向量
            embed_chunks = model.encode(texts, **kwarg)
            np.save(save_path, embed_chunks)
        # 获取向量维度
        embed_chunks = model.encode([""], **kwarg)
    return embed_chunks.shape[-1]

def construct_index(index_dir, model_name, h_dim=768, HNSW=False, M=32, efConstruction=64, efSearch=32):
    """
    构建 FAISS 索引。
    
    Args:
        index_dir: 索引存储目录。
        model_name: 模型名称（用于决定度量方式）。
        h_dim: 向量维度。
        HNSW: 是否使用 HNSW 索引。
        M, efConstruction, efSearch: HNSW 索引的相关参数。
    
    Returns:
        faiss.Index: 构建好的 FAISS 索引。
    """
    with open(os.path.join(index_dir, "metadatas.jsonl"), 'w') as f:
        f.write("")
    
    if HNSW:
        # 使用 HNSW 索引
        M = M
        if "specter" in model_name.lower():
            index = faiss.IndexHNSWFlat(h_dim, M)
            index.metric_type = faiss.METRIC_L2
        else:
            index = faiss.IndexHNSWFlat(h_dim, M)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = efSearch        
    else:
        # 使用线性索引（暴力搜索）
        if "specter" in model_name.lower():
            index = faiss.IndexFlatL2(h_dim)
        else:
            index = faiss.IndexFlatIP(h_dim)

    # 遍历所有保存的向量文件并添加到索引中
    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(index_dir, "embedding")))):
        curr_embed = np.load(os.path.join(index_dir, "embedding", fname))
        index.add(curr_embed)
        # 记录元数据，以便后续根据索引 ID 找回原始文档
        with open(os.path.join(index_dir, "metadatas.jsonl"), 'a+') as f:
            f.write("\n".join([json.dumps({'index': i, 'source': fname.replace(".npy", "")}) for i in range(len(curr_embed))]) + '\n')

    # 保存索引文件
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    return index


class Retriever: 
    """
    封装单个检索器对单个语料库的检索逻辑。
    支持向量检索和 BM25 检索。
    """

    def __init__(self, retriever_name="ncbi/MedCPT-Query-Encoder", corpus_name="textbooks", db_dir="./corpus", HNSW=False, **kwarg):
        """
        初始化检索器。
        如果本地没有对应的语料库或索引，会自动尝试下载。
        """
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name

        self.db_dir = db_dir
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, "chunk")
        
        # 1. 检查并准备语料库分块数据
        if not os.path.exists(self.chunk_dir):
            print("Cloning the {:s} corpus from Huggingface...".format(self.corpus_name))
            if self.corpus_name == "wiki_hotpotqa":
                os.system("git clone https://huggingface.co/datasets/RAG-Gym/{:s} {:s}".format(corpus_name, os.path.join(self.db_dir, self.corpus_name)))
            else:
                os.system("git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(corpus_name, os.path.join(self.db_dir, self.corpus_name)))
            if self.corpus_name == "statpearls":
                print("Downloading the statpearls corpus from NCBI bookshelf...")
                os.system("wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {:s}".format(os.path.join(self.db_dir, self.corpus_name)))
                os.system("tar -xzvf {:s} -C {:s}".format(os.path.join(db_dir, self.corpus_name, "statpearls_NBK430685.tar.gz"), os.path.join(self.db_dir, self.corpus_name)))
                print("Chunking the statpearls corpus...")
                os.system("python src/data/statpearls.py")
        
        # 2. 确定索引目录
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index", self.retriever_name.replace("Query-Encoder", "Article-Encoder"))
        
        # 3. 初始化 BM25 检索或向量检索
        if "bm25" in self.retriever_name.lower():
            from pyserini.search.lucene import LuceneSearcher
            self.metadatas = None
            self.embedding_function = None
            if os.path.exists(self.index_dir):
                self.index = LuceneSearcher(os.path.join(self.index_dir))
            else:
                # 自动构建 BM25 索引
                os.system("python -m pyserini.index.lucene --collection JsonCollection --input {:s} --index {:s} --generator DefaultLuceneDocumentGenerator --threads 16".format(self.chunk_dir, self.index_dir))
                self.index = LuceneSearcher(os.path.join(self.index_dir))
        else:
            # 初始化向量检索索引
            if os.path.exists(os.path.join(self.index_dir, "faiss.index")):
                self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
                self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]
            else:
                print("[In progress] Embedding the {:s} corpus with the {:s} retriever...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))
                # 尝试从网上下载预先计算好的向量（针对特定模型和语料库）
                if self.corpus_name in ["textbooks", "pubmed", "wikipedia"] and self.retriever_name in ["allenai/specter", "facebook/contriever", "ncbi/MedCPT-Query-Encoder"] and not os.path.exists(os.path.join(self.index_dir, "embedding")):
                    print("[In progress] Downloading the {:s} embeddings given by the {:s} model...".format(self.corpus_name, self.retriever_name.replace("Query-Encoder", "Article-Encoder")))
                    os.makedirs(self.index_dir, exist_ok=True)
                    if self.corpus_name == "textbooks":
                        if self.retriever_name == "allenai/specter":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EYRRpJbNDyBOmfzCOqfQzrsBwUX0_UT8-j_geDPcVXFnig?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "facebook/contriever":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQqzldVMCCVIpiFV4goC7qEBSkl8kj5lQHtNq8DvHJdAfw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQ8uXe4RiqJJm0Tmnx7fUUkBKKvTwhu9AqecPA3ULUxUqQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                    elif self.corpus_name == "pubmed":
                        if self.retriever_name == "allenai/specter":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/Ebz8ySXt815FotxC1KkDbuABNycudBCoirTWkKfl8SEswA?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "facebook/contriever":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EWecRNfTxbRMnM0ByGMdiAsBJbGJOX_bpnUoyXY9Bj4_jQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EVCuryzOqy5Am5xzRu6KJz4B6dho7Tv7OuTeHSh3zyrOAw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                    elif self.corpus_name == "wikipedia":
                        if self.retriever_name == "allenai/specter":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/Ed7zG3_ce-JOmGTbgof3IK0BdD40XcuZ7AGZRcV_5D2jkA?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "facebook/contriever":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/ETKHGV9_KNBPmDM60MWjEdsBXR4P4c7zZk1HLLc0KVaTJw?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                        elif self.retriever_name == "ncbi/MedCPT-Query-Encoder":
                            os.system("wget -O {:s} https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EXoxEANb_xBFm6fa2VLRmAcBIfCuTL-5VH6vl4GxJ06oCQ?download=1".format(os.path.join(self.index_dir, "embedding.zip")))
                    os.system("unzip {:s} -d {:s}".format(os.path.join(self.index_dir, "embedding.zip"), self.index_dir))
                    os.system("rm {:s}".format(os.path.join(self.index_dir, "embedding.zip")))
                    h_dim = 768
                else:
                    # 如果无法下载，则现场计算向量并构建索引
                    h_dim = embed(chunk_dir=self.chunk_dir, index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), **kwarg)

                print("[In progress] Embedding finished! The dimension of the embeddings is {:d}.".format(h_dim))
                self.index = construct_index(index_dir=self.index_dir, model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"), h_dim=h_dim, HNSW=HNSW)
                print("[Finished] Corpus indexing finished!")
                self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]            
            
            # 加载向量检索模型（Query Encoder）
            if self.retriever_name == config_local_embedding.get("model_name"):
                self.embedding_function = APIEmbeddingFunction(
                    model_name=config_local_embedding["model_name"],
                    base_url=config_local_embedding["base_url"],
                    api_key=config_local_embedding["api_key"]
                )
            elif "contriever" in self.retriever_name.lower():
                self.embedding_function = SentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.embedding_function = CustomizeSentenceTransformer(self.retriever_name, device="cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_function.eval()

    def get_relevant_documents(self, question, k=32, id_only=False, **kwarg):
        """
        根据问题检索相关文档。
        
        Args:
            question: 检索问题字符串。
            k: 返回的相关文档数量。
            id_only: 是否只返回文档 ID 和得分。
        
        Returns:
            tuple: (文档内容列表/ID列表, 检索得分列表)
        """
        assert type(question) == str
        question = [question]

        if "bm25" in self.retriever_name.lower():
            # BM25 检索逻辑
            res_ = [[]]
            hits = self.index.search(question[0], k=k)
            res_[0].append(np.array([h.score for h in hits]))
            ids = [h.docid for h in hits]
            indices = [{"source": '_'.join(h.docid.split('_')[:-1]), "index": eval(h.docid.split('_')[-1])} for h in hits]
        else:
            # 向量检索逻辑
            with torch.no_grad():
                query_embed = self.embedding_function.encode(question, **kwarg)
            res_ = self.index.search(query_embed, k=k)
            ids = ['_'.join([self.metadatas[i]["source"], str(self.metadatas[i]["index"])]) for i in res_[1][0]]
            indices = [self.metadatas[i] for i in res_[1][0]]

        scores = res_[0][0].tolist()
        
        if id_only:
            return [{"id":i} for i in ids], scores
        else:
            return self.idx2txt(indices), scores

    def idx2txt(self, indices):
        """
        根据索引元数据加载原始文档内容。
        """
        return [json.loads(open(os.path.join(self.chunk_dir, i["source"]+".jsonl")).read().strip().split('\n')[i["index"]]) for i in indices]

class RetrievalSystem:
    """
    检索系统集成类，管理多个 Retriever 实例，并提供统一的检索和 RRF 融合接口。
    """

    def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", HNSW=False, cache=False):
        """
        初始化检索系统。
        
        Args:
            retriever_name: 检索器配置名称（见 retriever_names）。
            corpus_name: 语料库配置名称（见 corpus_names）。
            db_dir: 数据库存储根目录。
            HNSW: 是否使用 HNSW 索引。
            cache: 是否使用文档内容提取缓存。
        """
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names
        
        # 初始化内部所有的检索器实例
        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                self.retrievers[-1].append(Retriever(retriever, corpus, db_dir, HNSW=HNSW))
        
        self.cache = cache
        if self.cache:
            self.docExt = DocExtracter(cache=True, corpus_name=self.corpus_name, db_dir=db_dir)
        else:
            self.docExt = None
    
    def retrieve(self, question, k=32, rrf_k=100, id_only=False):
        """
        综合所有检索器和语料库的结果，进行统一检索。
        
        Args:
            question: 检索问题。
            k: 最终返回的文档数量。
            rrf_k: RRF 融合算法中的常数参数。
            id_only: 是否只返回 ID。
        """
        assert type(question) == str

        output_id_only = id_only
        if self.cache:
            # 如果开启了缓存，先获取 ID，后续再通过 DocExtracter 提取内容
            id_only = True

        texts = []
        scores = []

        # 确定每个检索器需要获取的初始结果数量
        if "RRF" in self.retriever_name:
            k_ = max(k * 2, 100)
        else:
            k_ = k
        
        # 并行/循环遍历所有检索器和语料库
        for i in range(len(retriever_names[self.retriever_name])):
            texts.append([])
            scores.append([])
            for j in range(len(corpus_names[self.corpus_name])):
                t, s = self.retrievers[i][j].get_relevant_documents(question, k=k_, id_only=id_only)
                texts[-1].append(t)
                scores[-1].append(s)
        
        # 融合检索结果
        texts, scores = self.merge(texts, scores, k=k, rrf_k=rrf_k)
        
        # 如果有缓存，从缓存中提取完整内容
        if self.cache:
            texts = self.docExt.extract(texts)
        return texts, scores

    def merge(self, texts, scores, k=32, rrf_k=100):
        """
        使用 RRF (Reciprocal Rank Fusion) 算法合并来自不同检索器的结果。
        """
        RRF_dict = {}
        for i in range(len(retriever_names[self.retriever_name])):
            texts_all, scores_all = None, None
            # 首先合并同一个检索器在不同语料库上的结果
            for j in range(len(corpus_names[self.corpus_name])):
                if texts_all is None:
                    texts_all = texts[i][j]
                    scores_all = scores[i][j]
                else:
                    texts_all = texts_all + texts[i][j]
                    scores_all = scores_all + scores[i][j]
            
            # 对当前检索器的所有结果按得分排序
            if "specter" in retriever_names[self.retriever_name][i].lower():
                sorted_index = np.array(scores_all).argsort() # SPECTER 使用 L2 距离，越小越好
            else:
                sorted_index = np.array(scores_all).argsort()[::-1] # 其他模型通常使用内积或相似度，越大越好
            
            texts[i] = [texts_all[i] for i in sorted_index]
            scores[i] = [scores_all[i] for i in sorted_index]
            
            # 计算 RRF 得分
            for j, item in enumerate(texts[i]):
                if item["id"] in RRF_dict:
                    RRF_dict[item["id"]]["score"] += 1 / (rrf_k + j + 1)
                    RRF_dict[item["id"]]["count"] += 1
                else:
                    RRF_dict[item["id"]] = {
                        "id": item["id"],
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "score": 1 / (rrf_k + j + 1),
                        "count": 1
                        }
        
        # 按 RRF 得分降序排列
        RRF_list = sorted(RRF_dict.items(), key=lambda x: x[1]["score"], reverse=True)
        
        if len(texts) == 1:
            # 如果只有一个检索器，直接返回其 Top-K 结果
            texts = texts[0][:k]
            scores = scores[0][:k]
        else:
            # 返回融合后的 Top-K 结果
            texts = [dict((key, item[1][key]) for key in ("id", "title", "content")) for item in RRF_list[:k]]
            scores = [item[1]["score"] for item in RRF_list[:k]]
        return texts, scores
    

class DocExtracter:
    """
    文档提取器，支持根据文档 ID 快速获取其内容。
    可以通过将语料库 ID 到内容的映射加载到内存（cache=True）来加速。
    """
    
    def __init__(self, db_dir="./corpus", cache=False, corpus_name="MedCorp"):
        """
        初始化提取器。
        """
        self.db_dir = db_dir
        self.cache = cache
        print("Initializing the document extracter...")
        
        # 确保语料库分块数据存在
        for corpus in corpus_names[corpus_name]:
            if not os.path.exists(os.path.join(self.db_dir, corpus, "chunk")):
                print("Cloning the {:s} corpus from Huggingface...".format(corpus))
                os.system("git clone https://huggingface.co/datasets/MedRAG/{:s} {:s}".format(corpus, os.path.join(self.db_dir, corpus)))
                if corpus == "statpearls":
                    print("Downloading the statpearls corpus from NCBI bookshelf...")
                    os.system("wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz -P {:s}".format(os.path.join(self.db_dir, corpus)))
                    os.system("tar -xzvf {:s} -C {:s}".format(os.path.join(self.db_dir, corpus, "statpearls_NBK430685.tar.gz"), os.path.join(self.db_dir, corpus)))
                    print("Chunking the statpearls corpus...")
                    os.system("python src/data/statpearls.py")
        
        # 如果开启缓存，则加载整个映射表到内存
        if self.cache:
            if os.path.exists(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"]))):
                self.dict = json.load(open(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"]))))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(self.db_dir, corpus, "chunk")))):
                        if open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip() == "":
                            continue
                        for i, line in enumerate(open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip().split('\n')):
                            item = json.loads(line)
                            _ = item.pop("contents", None)
                            self.dict[item["id"]] = item
                with open(os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"])), 'w') as f:
                    json.dump(self.dict, f)
    
        else:
            if os.path.exists(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"]))):
                self.dict = json.load(open(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"]))))
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    for fname in tqdm.tqdm(sorted(os.listdir(os.path.join(self.db_dir, corpus, "chunk")))):
                        if open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip() == "":
                            continue
                        for i, line in enumerate(open(os.path.join(self.db_dir, corpus, "chunk", fname)).read().strip().split('\n')):
                            item = json.loads(line)
                            # assert item["id"] not in self.dict
                            self.dict[item["id"]] = {"fpath": os.path.join(corpus, "chunk", fname), "index": i}
                with open(os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"])), 'w') as f:
                    json.dump(self.dict, f, indent=4)
        print("Initialization finished!")
    
    def extract(self, ids):
        if self.cache:
            output = []
            for i in ids:
                item = self.dict[i] if type(i) == str else self.dict[i["id"]]
                output.append(item)
        else:
            output = []
            for i in ids:
                item = self.dict[i] if type(i) == str else self.dict[i["id"]]
                output.append(json.loads(open(os.path.join(self.db_dir, item["fpath"])).read().strip().split('\n')[item["index"]]))
        return output