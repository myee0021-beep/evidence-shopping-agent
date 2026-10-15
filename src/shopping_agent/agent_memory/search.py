"""记忆检索：BM25 + 向量两路召回，用 RRF 融合。

- BM25 负责精确词：品牌、型号、“45 码”；
- 向量负责模糊语义：“适合跑步的”“给长辈用的”；
- RRF 只看名次不看分数，两路分数量纲不同也能直接融合，不用调权重。

默认向量是“字符 n-gram 哈希”，零依赖、结果确定，适合单测和离线评测；
接真实 embedding 模型时，传一个 embed(texts) -> list[list[float]] 进来即可。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Callable, Sequence

Embedder = Callable[[Sequence[str]], list[list[float]]]

CJK = r"㐀-䶿一-鿿"
TOKEN_RE = re.compile(rf"[{CJK}]+|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文切成单字 + 相邻二元组，英文和数字按词。不依赖分词器。"""
    out: list[str] = []
    for chunk in TOKEN_RE.findall(text.lower()):
        if re.match(rf"[{CJK}]", chunk):
            out += list(chunk)
            out += [chunk[i : i + 2] for i in range(len(chunk) - 1)]
        else:
            out.append(chunk)
    return out


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / len(self.docs) if self.docs else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / (self.avgdl or 1)))
            out.append(s)
        return out


def hashing_embed(texts: Sequence[str], dim: int = 512) -> list[list[float]]:
    """字符 1~3-gram 哈希向量，L2 归一化。"""
    vecs = []
    for text in texts:
        v = [0.0] * dim
        s = re.sub(r"\s+", " ", text.lower())
        for n in (1, 2, 3):
            for i in range(len(s) - n + 1):
                h = int(hashlib.md5(s[i : i + n].encode()).hexdigest()[:8], 16)
                v[h % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：score(d) = Σ 1 / (k + rank)。"""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: (-x[1], x[0]))


def hybrid_search(query: str, docs: Sequence[str], k: int = 5, embed: Embedder = hashing_embed) -> list[tuple[int, float]]:
    """返回 [(文档下标, RRF 分数)]。只有至少一路认为相关（分数 > 0）的文档才参与排名。"""
    if not docs:
        return []
    bm = BM25(docs).scores(query)
    bm_rank = [i for i in sorted(range(len(docs)), key=lambda i: -bm[i]) if bm[i] > 0]
    vecs = embed([query, *docs])
    sims = [cosine(vecs[0], v) for v in vecs[1:]]
    vec_rank = [i for i in sorted(range(len(docs)), key=lambda i: -sims[i]) if sims[i] > 0.1]
    return rrf([bm_rank, vec_rank])[:k]
