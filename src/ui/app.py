"""
法审助手 - Web界面 (Streamlit)

启动方式:
    python -m streamlit run src/ui/app.py
    或在项目根目录: streamlit run src/ui/app.py
"""
import sys
import os

# 确保项目根目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from src.rag.generator import generate_report, generate_followup
from src.rag.vector_store import get_collection_stats

# ====== 页面配置 ======
st.set_page_config(
    page_title="法审助手",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== 预设示例 ======
PRESET_EXAMPLES = [
    {
        "title": "📍 社交App - 附近的人",
        "description": "社交App允许用户查看附近100米内其他用户的实时位置",
        "category": "位置隐私",
    },
    {
        "title": "🪪 身份认证 - 人脸识别",
        "description": "用户上传身份证照片和人脸照片，系统自动实名认证并保存信息",
        "category": "生物识别",
    },
    {
        "title": "🧒 教育App - 未成年人",
        "description": "面向中小学生的在线教育平台，收集学生姓名、年龄、成绩、位置信息",
        "category": "未成年人保护",
    },
    {
        "title": "💬 客服系统 - AI对话记录",
        "description": "电商平台使用AI客服机器人，自动记录和分析用户聊天内容用于优化服务",
        "category": "数据使用",
    },
    {
        "title": "🌐 跨境电商 - 数据出境",
        "description": "跨境电商App将中国用户的订单数据和收货地址传输到海外服务器处理",
        "category": "数据跨境",
    },
    {
        "title": "🏥 健康App - 敏感信息",
        "description": "健康管理App收集用户心率、睡眠、步数等健康数据，用于生成健康报告",
        "category": "敏感个人信息",
    },
]


def init_session_state():
    """初始化Streamlit会话状态（保存对话历史）"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_input" not in st.session_state:
        st.session_state.current_input = ""
    if "current_report" not in st.session_state:
        st.session_state.current_report = ""


def render_sidebar():
    """侧边栏：显示状态、预设示例"""
    with st.sidebar:
        st.title("⚖️ 法审助手")
        st.caption("AI驱动的产品合规审查系统")

        # 数据库状态
        st.divider()
        st.subheader("📊 系统状态")
        try:
            stats = get_collection_stats()
            if "error" not in stats:
                st.success(f"法规库已就绪: {stats['total_articles']} 条条款")
            else:
                st.warning("法规库未初始化，请先在命令行运行: python main.py --build")
        except Exception as e:
            st.warning(f"数据库状态异常: {e}")

        # 预设示例
        st.divider()
        st.subheader("📋 预设示例")
        for i, example in enumerate(PRESET_EXAMPLES):
            if st.button(
                f"{example['title']}",
                key=f"example_{i}",
                use_container_width=True,
                help=example["description"],
            ):
                st.session_state.current_input = example["description"]
                st.rerun()

        # 使用说明
        st.divider()
        st.subheader("❓ 使用说明")
        st.markdown("""
        1. 在输入框中描述产品功能
        2. 点击"开始审查"
        3. 查看生成的合规报告
        4. 可在追问区继续提问

        **提示**: 描述越具体，分析越准确。
        尽量包含：
        - 产品类型
        - 涉及的数据类型
        - 数据的流向和用途
        """)

        st.divider()
        st.caption("Built with Streamlit + ChromaDB + DeepSeek")
        st.caption("法学+AI+产品 交叉能力展示项目")


def render_main_area():
    """主区域：输入区 + 报告展示区"""
    st.title("法审助手")
    st.caption("输入产品功能描述，获得结构化的合规风险分析报告")

    # ====== 输入区 ======
    st.subheader("📝 产品功能描述")

    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_area(
            label="请描述你要审查的产品功能",
            value=st.session_state.current_input,
            placeholder="例如：我们的社交App允许用户查看附近100米内的其他用户位置信息...",
            height=120,
            label_visibility="collapsed",
        )
    with col2:
        st.write("")  # 对齐
        analyze_btn = st.button("🔍 开始审查", type="primary", use_container_width=True)
        clear_btn = st.button("🗑️ 清空对话", use_container_width=True)

    if clear_btn:
        st.session_state.chat_history = []
        st.session_state.current_input = ""
        st.session_state.current_report = ""
        st.rerun()

    # ====== 执行审查 ======
    if analyze_btn and user_input.strip():
        with st.spinner("正在进行合规审查...\n\n正在检索相关法规 → 分析法律风险 → 生成报告"):
            try:
                report = generate_report(user_input.strip())
                st.session_state.current_report = report
                st.session_state.current_input = user_input.strip()

                # 保存到对话历史
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input.strip(),
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": report,
                })
            except Exception as e:
                st.error(f"审查过程出错: {e}")
                st.info(
                    "常见原因：\n"
                    "1. 未配置 DEEPSEEK_API_KEY（请检查 .env 文件）\n"
                    "2. 向量数据库未初始化（请先运行 python main.py --build）\n"
                    "3. 网络连接问题"
                )

    # ====== 报告展示区 ======
    if st.session_state.current_report:
        st.divider()
        st.subheader("📄 合规审查报告")

        with st.container(border=True):
            st.markdown(st.session_state.current_report)

    # ====== 追问区 ======
    if st.session_state.chat_history:
        st.divider()
        st.subheader("💬 追问修改建议")

        followup = st.text_input(
            "如果对方案有修改想法，或者想深入了解某个风险点，请在此输入：",
            placeholder='例如："如果去掉实时位置，只显示距离，风险会变化吗？"',
            key="followup_input",
        )
        followup_btn = st.button("发送追问", type="secondary")

        if followup_btn and followup.strip():
            with st.spinner("正在分析追问..."):
                try:
                    response = generate_followup(
                        original_input=st.session_state.current_input,
                        followup_question=followup.strip(),
                        chat_history=st.session_state.chat_history,
                    )
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": followup.strip(),
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                    })
                    st.session_state.current_report = response
                    st.rerun()
                except Exception as e:
                    st.error(f"追问处理出错: {e}")

        # ====== 历史对话 ======
        if len(st.session_state.chat_history) > 2:
            st.divider()
            st.subheader("📜 对话历史")
            # 跳过最新一轮（已在上面展示），显示之前的对话
            history_to_show = st.session_state.chat_history[:-2]
            for i in range(0, len(history_to_show), 2):
                msg_user = history_to_show[i] if i < len(history_to_show) else None
                msg_assistant = history_to_show[i+1] if i+1 < len(history_to_show) else None
                with st.expander(
                    f"第{i//2 + 1}轮: {msg_user['content'][:50]}..." if msg_user else "对话",
                    expanded=False,
                ):
                    if msg_user:
                        st.caption("用户输入:")
                        st.text(msg_user["content"])
                    if msg_assistant:
                        st.caption("审查结果:")
                        st.markdown(msg_assistant["content"])


def main():
    init_session_state()
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
