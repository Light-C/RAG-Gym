from rag_gym.envs.IR import RetrievalSystem, Retriever, DocExtracter, corpus_names, retriever_names

cached_retrievers = dict()
cached_corpora = dict()

class RetrievalSystemCached(RetrievalSystem):
    """
    带有缓存机制的检索系统。
    继承自 RetrievalSystem，通过全局变量缓存已初始化的 Retriever 和 DocExtracter，
    避免在多次实例化时重复加载模型和语料库索引，从而节省显存和初始化时间。
    """
    def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", HNSW=False, cache=False):
        """
        初始化带有缓存的检索系统。
        """
        global cached_retrievers
        global cached_corpora
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names
        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                # 检查全局缓存中是否已存在该检索器和语料库的组合
                if f"{retriever}_{corpus}" not in cached_retrievers:
                    cached_retrievers[f"{retriever}_{corpus}"] = Retriever(retriever, corpus, db_dir, HNSW=HNSW)
                self.retrievers[-1].append(cached_retrievers[f"{retriever}_{corpus}"])
        self.cache = cache
        if self.cache:
            # 检查全局缓存中是否已存在该语料库的文档提取器
            if self.corpus_name not in cached_corpora:
                cached_corpora[self.corpus_name] = DocExtracter(cache=True, corpus_name=self.corpus_name, db_dir=db_dir)
            self.docExt = cached_corpora[self.corpus_name]
        else:
            self.docExt = None