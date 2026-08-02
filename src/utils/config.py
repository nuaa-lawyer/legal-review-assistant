"""
集中管理所有配置：API Key、模型选择、路径等。
所有敏感信息从 .env 文件读取，不硬编码在代码中。
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")         # 原始Word文件
# 向量数据现在存储在 data/processed/ 目录下 (vectors.npy + articles.json)

# ============ LLM配置 ============
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = "deepseek-chat"             # DeepSeek-V3
LLM_TEMPERATURE = 0.3                    # 低温度 = 更严谨的法律分析

# ============ Embedding配置 ============
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local")  # "local" 或 "api"
LOCAL_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"       # 本地免费模型
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

# ============ 检索配置 ============
TOP_K_RETRIEVAL = 5       # 每次检索返回的法条数量

# ============ 核心法规列表（优先处理） ============
PRIORITY_LAWS = [
    "中华人民共和国个人信息保护法",
    "中华人民共和国数据安全法",
    "中华人民共和国网络安全法",
    "中华人民共和国未成年人保护法",
    "中华人民共和国消费者权益保护法",
    "中华人民共和国电子商务法",
    "中华人民共和国反电信网络诈骗法",
    "中华人民共和国生物安全法",
]
