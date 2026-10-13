"""
向量存储与检索模块（轻量版，无需 ChromaDB）

使用 numpy + JSON 实现：
- 向量存为 .npy 文件
- 条款数据存为 .json 文件
- 检索时计算余弦相似度

对于几千条法规条款的规模完全够用，且零额外依赖。
"""
import os
import json
import numpy as np

from src.utils.config import TOP_K_RETRIEVAL
from src.rag.embedding import get_embeddings

# bge-large-zh-v1.5 最大支持 512 token，中文约 1 字符/token
# 实测 520 字成功、540 字失败；截断到 450 字 + 法条前缀后仍 <512 token
MAX_CHARS_FOR_EMBEDDING = 450


def _get_store_paths():
    """获取向量库文件路径（兼容 processed/ 与 data/processed/ 两种目录结构）"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(project_root, "processed"),
        os.path.join(project_root, "data", "processed"),
    ]
    # 优先返回同时包含 vectors.npy 和 articles.json 的目录
    for base in candidates:
        vectors_path = os.path.join(base, "vectors.npy")
        articles_path = os.path.join(base, "articles.json")
        if os.path.exists(vectors_path) and os.path.exists(articles_path):
            return {"vectors": vectors_path, "articles": articles_path}
    # 兜底：返回 processed/，并确保目录存在（供 build_index 写入）
    base = candidates[0]
    os.makedirs(base, exist_ok=True)
    return {
        "vectors": os.path.join(base, "vectors.npy"),       # 向量矩阵
        "articles": os.path.join(base, "articles.json"),    # 条款数据
    }


def build_index(articles_json_path: str):
    """
    构建向量索引。
    1. 读取解析好的条款JSON
    2. 逐条调用SiliconFlow API生成向量
    3. 保存为 .npy + .json

    这个操作只需执行一次（或法规更新后重新执行）。
    """
    with open(articles_json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"共 {len(articles)} 条法规条款待索引")

    embeddings = get_embeddings()
    paths = _get_store_paths()

    # 生成向量（SiliconFlow免费额度每天2000条）
    all_vectors = []
    skipped = []   # 记录被跳过的条款
    vec_dim = None  # 向量维度（从第一条成功获取）
    BATCH_SIZE = 50
    total = len(articles)

    for i in range(0, total, BATCH_SIZE):
        batch = articles[i:i + BATCH_SIZE]
        texts = []
        for a in batch:
            full_text = a["text"]
            if len(full_text) > MAX_CHARS_FOR_EMBEDDING:
                full_text = full_text[:MAX_CHARS_FOR_EMBEDDING]
            texts.append(f"【{a['law_name']}】{a['article_id']}\n{full_text}")

        # 先尝试批量处理
        try:
            vecs_list = embeddings.embed_documents(texts)
            all_vectors.extend(vecs_list)
            if vec_dim is None and vecs_list:
                vec_dim = len(vecs_list[0])
        except Exception:
            # 批量失败 → 逐条重试，跳过问题条款
            print(f"  批量 {i//BATCH_SIZE + 1} 失败，逐条重试中...")
            for j, (a, t) in enumerate(zip(batch, texts)):
                try:
                    vec = embeddings.embed_query(t)
                    all_vectors.append(vec)
                    if vec_dim is None:
                        vec_dim = len(vec)
                except Exception:
                    # 该条款API拒绝，用零向量占位
                    skipped.append(f"{a['law_name']} {a['article_id']}")
                    if vec_dim:
                        all_vectors.append([0.0] * vec_dim)
                    print(f"    跳过: {a['law_name']} {a['article_id']}")

        done = min(i + BATCH_SIZE, total)
        print(f"  生成向量: {done}/{total}")

    # 最终为跳过的条款补零向量（如果之前不知道维度）
    if vec_dim is None:
        raise RuntimeError("无法获取向量维度，请检查API连接")
    for _ in range(len(skipped) - sum(1 for v in all_vectors if len(v) == vec_dim)):
        pass  # 已在循环中处理

    if skipped:
        print(f"\n警告: {len(skipped)} 条条款被跳过（API拒绝）:")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... 等共 {len(skipped)} 条")

    # 保存
    vectors_array = np.array(all_vectors, dtype=np.float32)
    np.save(paths["vectors"], vectors_array)
    with open(paths["articles"], "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"向量索引构建完成！共 {len(articles)} 条")
    print(f"向量文件: {paths['vectors']} ({vectors_array.shape})")
    return vectors_array, articles


def _load_index():
    """加载向量索引到内存"""
    paths = _get_store_paths()
    vectors_exists = os.path.exists(paths["vectors"])
    articles_exists = os.path.exists(paths["articles"])
    if not vectors_exists or not articles_exists:
        raise RuntimeError(
            "向量索引未初始化！找不到文件：\n"
            f"  vectors.npy 存在={vectors_exists} -> {paths['vectors']}\n"
            f"  articles.json 存在={articles_exists} -> {paths['articles']}"
        )

    vectors = np.load(paths["vectors"])
    with open(paths["articles"], "r", encoding="utf-8") as f:
        articles = json.load(f)
    return vectors, articles


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """计算余弦相似度（query_vec 与 matrix 中每一行的相似度）"""
    # 归一化
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return np.dot(matrix_norm, query_norm)


def search(query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    """
    语义检索：输入用户描述，返回最相关的K条法规条款。

    返回格式:
    [
        {
            "law_name": "中华人民共和国个人信息保护法",
            "article_id": "第十三条",
            "chapter": "第二章",
            "text": "条款全文...",
            "relevance_score": 0.85
        },
        ...
    ]
    """
    embeddings = get_embeddings()
    vectors, articles = _load_index()

    # 生成查询向量
    query_vec = embeddings.embed_query(query)
    query_vec = np.array(query_vec, dtype=np.float32)

    # 计算相似度
    scores = _cosine_similarity(query_vec, vectors)

    # 取 Top-K
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        art = articles[idx]
        results.append({
            "law_name": art["law_name"],
            "article_id": art["article_id"],
            "chapter": art.get("chapter", ""),
            "text": art["text"],
            "relevance_score": round(score, 4),
        })

    return results


def get_articles_by_keys(pairs: list) -> dict:
    """
    按 (法规全名, 条款号) 精确取回条款原文，供"规则 + 语义"混合召回使用。

    返回 {(law_name, article_id): article_record}
    """
    if not pairs:
        return {}
    _, articles = _load_index()
    want = {(str(law), str(art)) for law, art in pairs}
    out = {}
    for a in articles:
        key = (a["law_name"], a["article_id"])
        if key in want:
            out[key] = {
                "law_name": a["law_name"],
                "article_id": a["article_id"],
                "chapter": a.get("chapter", ""),
                "text": a["text"],
                "relevance_score": None,
            }
    return out


def get_collection_stats() -> dict:
    """获取向量库统计信息"""
    try:
        vectors, articles = _load_index()
        return {
            "total_articles": len(articles),
            "vector_dimension": vectors.shape[1],
            "status": "已就绪",
        }
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"索引加载失败（{type(e).__name__}）：{e}"}


if __name__ == "__main__":
    import sys

    if "--build" in sys.argv:
        from src.utils.config import BASE_DIR
        articles_path = os.path.join(BASE_DIR, "data", "processed", "articles.json")
        if not os.path.exists(articles_path):
            print(f"错误: 找不到 {articles_path}")
            print("请先运行解析: python main.py --build")
            sys.exit(1)
        build_index(articles_path)
    else:
        stats = get_collection_stats()
        print(f"数据库状态: {stats}")

        test_query = "用户上传身份证照片后，App自动识别并保存用户身份信息"
        print(f"\n测试查询: {test_query}")
        try:
            results = search(test_query, top_k=3)
            for i, r in enumerate(results):
                print(f"\n--- 第{i+1}条 (相关性: {r['relevance_score']}) ---")
                print(f"{r['law_name']} {r['article_id']}")
                print(r['text'][:200])
        except RuntimeError as e:
            print(f"错误: {e}")
