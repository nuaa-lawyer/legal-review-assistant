"""
Embedding生成模块：将文本转为向量，供语义检索使用。

使用 SiliconFlow 云端 API 调用 bge-large-zh-v1.5 模型。
无需本地 GPU，无需下载模型，有免费额度（2000条/天）。

依赖：openai（已安装）
"""
from openai import OpenAI
from src.utils.config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL


class SimpleEmbeddings:
    """
    轻量Embedding封装，提供和 langchain 一样的接口，
    但只依赖 openai 库，无需安装 langchain。
    """

    def __init__(self):
        if not SILICONFLOW_API_KEY:
            raise ValueError(
                "请先在 .env 文件中配置 SILICONFLOW_API_KEY\n"
                "注册地址: https://siliconflow.cn"
            )
        self.client = OpenAI(
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL,
        )
        self.model = "BAAI/bge-large-zh-v1.5"

    def embed_query(self, text: str) -> list[float]:
        """将单条文本转为向量（用于用户输入的查询）"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """将多条文本批量转为向量（用于构建法规索引）"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # 按输入顺序返回
        embeddings = [item.embedding for item in response.data]
        return embeddings


# 全局单例，避免重复创建客户端
_embeddings_instance = None


def get_embeddings() -> SimpleEmbeddings:
    """获取Embedding实例（单例模式）"""
    global _embeddings_instance
    if _embeddings_instance is None:
        print(f"连接 SiliconFlow Embedding API...")
        _embeddings_instance = SimpleEmbeddings()
        print("Embedding 服务就绪")
    return _embeddings_instance


if __name__ == "__main__":
    emb = get_embeddings()
    test_text = "个人信息处理者应当取得个人的同意"
    vec = emb.embed_query(test_text)
    print(f"测试文本: {test_text}")
    print(f"向量维度: {len(vec)}")
    print(f"前5个值: {vec[:5]}")
    print("Embedding模块工作正常！")
