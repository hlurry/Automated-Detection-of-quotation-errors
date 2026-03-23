"""
检索模块 - 包含两种检索方案
方案A: 关键词提取 + BM25 + 语义混合检索
方案B: 假设生成检索 (HyDE变体)
"""
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import re

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, 
    BM25_WEIGHT, SEMANTIC_WEIGHT, EMBEDDING_MODEL
)
from utils import chunk_text

class BaseRetriever:
    """检索器基类"""
    
    def __init__(self):
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """加载embedding模型"""
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding model loaded.")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取文本的embedding"""
        return self.embedding_model.encode(texts, show_progress_bar=False)


class HybridRetriever(BaseRetriever):
    """
    方案A: 混合检索器
    结合BM25关键词检索和语义检索
    """
    
    def __init__(self):
        super().__init__()
    
    def extract_keywords(self, claim: str) -> List[str]:
        """从claim中提取关键词（简单方法：提取名词短语和关键术语）"""
        # 移除常见停用词
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'that',
            'which', 'who', 'whom', 'this', 'these', 'those', 'what', 'it', 'its'
        }
        
        # 分词并过滤
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9-]*[a-zA-Z0-9]\b|\b[a-zA-Z]\b', claim.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # 保留原始大小写的版本用于匹配
        return list(set(keywords))
    
    def bm25_search(self, query_keywords: List[str], chunks: List[str], top_k: int) -> List[Tuple[int, float]]:
        """BM25关键词检索"""
        # 对chunks进行分词
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        
        # 使用关键词作为查询
        scores = bm25.get_scores(query_keywords)
        
        # 获取top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def semantic_search(self, query: str, chunks: List[str], chunk_embeddings: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """语义检索"""
        query_embedding = self.get_embeddings([query])[0]
        
        # 计算余弦相似度
        similarities = np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def reciprocal_rank_fusion(self, 
                                bm25_results: List[Tuple[int, float]], 
                                semantic_results: List[Tuple[int, float]], 
                                k: int = 60) -> List[Tuple[int, float]]:
        """RRF融合两种检索结果"""
        scores = {}
        
        # BM25结果
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + BM25_WEIGHT / (k + rank + 1)
        
        # 语义检索结果
        for rank, (idx, _) in enumerate(semantic_results):
            scores[idx] = scores.get(idx, 0) + SEMANTIC_WEIGHT / (k + rank + 1)
        
        # 排序
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def retrieve(self, claim: str, full_text: str, top_k: int = TOP_K) -> List[str]:
        """
        混合检索主函数
        """
        # 1. 文本分块
        chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            return []
        
        # 2. 提取关键词
        keywords = self.extract_keywords(claim)
        
        # 3. BM25检索
        bm25_results = self.bm25_search(keywords, chunks, top_k * 2)
        
        # 4. 语义检索
        chunk_embeddings = self.get_embeddings(chunks)
        semantic_results = self.semantic_search(claim, chunks, chunk_embeddings, top_k * 2)
        
        # 5. RRF融合
        fused_results = self.reciprocal_rank_fusion(bm25_results, semantic_results)
        
        # 6. 返回top-k个chunk
        top_indices = [idx for idx, _ in fused_results[:top_k]]
        return [chunks[idx] for idx in top_indices]


class HyDERetriever(BaseRetriever):
    """
    方案B: 假设生成检索 (HyDE变体)
    生成支持和反驳两个假设，分别检索后合并
    """
    
    def __init__(self, llm_client=None):
        super().__init__()
        self.llm_client = llm_client
    
    def set_llm_client(self, llm_client):
        """设置LLM客户端"""
        self.llm_client = llm_client
    
    def generate_hypotheses(self, claim: str) -> Tuple[str, str]:
        """
        用LLM生成支持假设和反驳假设
        """
        if self.llm_client is None:
            # 如果没有LLM客户端，使用简单的模板生成
            support_hypo = f"Research evidence supports that {claim}"
            contradict_hypo = f"Research evidence contradicts that {claim}. Studies show the opposite."
            return support_hypo, contradict_hypo
        
        prompt = f"""Given the following scientific claim, generate two hypothetical passages:
1. A passage that would SUPPORT this claim
2. A passage that would CONTRADICT this claim

Claim: {claim}

Generate brief, realistic scientific text (2-3 sentences each) that might appear in a research paper.

Output format:
SUPPORT: [supporting passage]
CONTRADICT: [contradicting passage]"""

        try:
            response = self.llm_client.chat(prompt, max_tokens=300)
            
            # 解析响应
            support_match = re.search(r'SUPPORT:\s*(.+?)(?=CONTRADICT:|$)', response, re.DOTALL | re.IGNORECASE)
            contradict_match = re.search(r'CONTRADICT:\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)
            
            support_hypo = support_match.group(1).strip() if support_match else f"Evidence supports that {claim}"
            contradict_hypo = contradict_match.group(1).strip() if contradict_match else f"Evidence contradicts that {claim}"
            
            return support_hypo, contradict_hypo
        except Exception as e:
            print(f"Error generating hypotheses: {e}")
            return f"Evidence supports that {claim}", f"Evidence contradicts that {claim}"
    
    def semantic_search(self, query: str, chunks: List[str], chunk_embeddings: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """语义检索"""
        query_embedding = self.get_embeddings([query])[0]
        
        similarities = np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def retrieve(self, claim: str, full_text: str, top_k: int = TOP_K) -> List[str]:
        """
        HyDE检索主函数
        """
        # 1. 文本分块
        chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            return []
        
        # 2. 生成假设
        support_hypo, contradict_hypo = self.generate_hypotheses(claim)
        
        # 3. 计算chunk embeddings
        chunk_embeddings = self.get_embeddings(chunks)
        
        # 4. 分别用两个假设检索
        support_results = self.semantic_search(support_hypo, chunks, chunk_embeddings, top_k)
        contradict_results = self.semantic_search(contradict_hypo, chunks, chunk_embeddings, top_k)
        
        # 5. 也用原始claim检索
        claim_results = self.semantic_search(claim, chunks, chunk_embeddings, top_k)
        
        # 6. 合并结果（去重，保持顺序）
        seen_indices = set()
        merged_indices = []
        
        # 交替添加三种结果
        all_results = [claim_results, support_results, contradict_results]
        max_len = max(len(r) for r in all_results)
        
        for i in range(max_len):
            for results in all_results:
                if i < len(results):
                    idx = results[i][0]
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        merged_indices.append(idx)
                        if len(merged_indices) >= top_k:
                            break
            if len(merged_indices) >= top_k:
                break
        
        return [chunks[idx] for idx in merged_indices[:top_k]]


def get_retriever(method: str, llm_client=None):
    """获取检索器实例"""
    if method == "hybrid":
        return HybridRetriever()
    elif method == "hyde":
        retriever = HyDERetriever(llm_client)
        return retriever
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
