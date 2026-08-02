"""
法审助手 - Hugging Face Spaces 部署入口

基于 Streamlit，兼容 HF Spaces 的 Python 3.10+ 环境。
本地运行: streamlit run app.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.agents.orchestrator import multi_agent_review
from src.rag.vector_store import get_collection_stats

st.set_page_config(
    page_title="法审助手 - AI合规审查",
    page_icon="⚖️",
    layout="wide",
)

# ===== 预设示例 =====
EXAMPLES = [
    ("📍 社交App-附近的人 (位置隐私)", "我们的社交App允许用户查看附近100米内的其他用户的实时位置信息"),
    ("🪪 身份认证-人脸识别 (生物识别)", "用户上传身份证照片和人脸照片后，系统自动进行实名认证并保存信息"),
    ("🧒 在线教育-未成年人 (未成年人保护)", "面向中小学生的在线教育平台，收集学生姓名、年龄、成绩、位置信息"),
    ("🌐 跨境电商-数据出境 (数据跨境)", "跨境电商App将中国用户的订单数据和收货地址传输到海外服务器处理"),
    ("🏥 健康App-敏感信息 (健康数据)", "健康管理App收集用户心率、睡眠、步数等健康数据，用于生成健康报告"),
]


def main():
    # ===== 侧边栏 =====
    with st.sidebar:
        st.title("⚖️ 法审助手")
        st.caption("AI驱动的产品合规审查系统 | 多Agent协作架构")

        st.divider()
        st.subheader("📊 系统状态")
        try:
            stats = get_collection_stats()
            if "error" not in stats:
                st.success(f"法规库已就绪: {stats['total_articles']} 条条款")
            else:
                st.warning("法规库未初始化")
        except Exception:
            st.warning("数据库状态未知")

        st.divider()
        st.subheader("📋 预设示例")
        for title, desc in EXAMPLES:
            if st.button(title, key=title[:20], use_container_width=True):
                st.session_state.user_input = desc
                st.rerun()

        st.divider()
        st.subheader("🏗️ 技术架构")
        st.markdown("""
        **多Agent协作模式**:
        1. 🔍 **分类Agent** - 识别数据类型与风险领域
        2. 📚 **检索Agent** - 多路召回+语义匹配法条
        3. 📝 **报告Agent** - 生成结构化合规报告

        **技术栈**:
        - Embedding: bge-large-zh-v1.5
        - 向量检索: NumPy + 余弦相似度
        - LLM: DeepSeek-V3
        - 前端: Streamlit
        """)

        st.divider()
        st.caption("法学 + AI + 产品 交叉能力展示项目")

    # ===== 主区域 =====
    st.title("AI合规审查")
    st.caption("输入产品功能描述 → 三个Agent协作 → 结构化合规风险报告")

    # 输入区
    st.subheader("📝 产品功能描述")
    user_input = st.text_area(
        "请描述你要审查的产品功能",
        value=st.session_state.get("user_input", ""),
        placeholder="例如：我们的社交App允许用户查看附近100米内的其他用户位置信息...",
        height=120,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([5, 1])
    with col1:
        pass
    with col2:
        analyze_btn = st.button("🔍 开始审查", type="primary", use_container_width=True)
        clear_btn = st.button("🗑️ 清空", use_container_width=True)

    if clear_btn:
        st.session_state.user_input = ""
        st.rerun()

    # 执行审查
    if analyze_btn and user_input.strip():
        st.session_state.user_input = user_input.strip()

        with st.spinner("多Agent协作中...\n\n🔍 Agent 1/3: 风险分类器正在分析数据类型...\n📚 Agent 2/3: 法规检索器正在匹配法条...\n📝 Agent 3/3: 报告生成器正在撰写报告..."):
            try:
                result = multi_agent_review(user_input.strip(), verbose=False)
            except Exception as e:
                st.error(f"审查失败: {e}")
                st.info("请检查 .env 中的 API Key 配置")
                return

        # 显示Agent执行过程
        st.divider()
        st.subheader("🤖 多Agent协作过程")

        cols = st.columns(3)
        icons = ["🔍", "📚", "📝"]
        for i, (step, col) in enumerate(zip(result["trace"], cols)):
            with col:
                with st.container(border=True):
                    st.metric(
                        label=f"{icons[i]} {step['agent']}",
                        value=f"{step['elapsed']}秒",
                    )
                    st.caption(step["summary"])

        st.caption(f"总耗时: {result['total_elapsed']} 秒")

        # 显示报告
        st.divider()
        st.subheader("📄 合规审查报告")
        with st.container(border=True):
            st.markdown(result["report"])

        # 追问区
        st.divider()
        st.subheader("💬 追问修改建议")
        followup = st.text_input(
            "如果对方案有修改想法，输入你的追问",
            placeholder="例如：如果去掉实时位置，只显示距离，风险会变化吗？",
        )
        if followup:
            st.info("追问功能将在下一版本中支持多Agent模式。当前版本请使用本地服务 (python src/ui/server.py)。")


if __name__ == "__main__":
    main()
