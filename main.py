"""
法审助手 - 主入口脚本

在启动Web界面之前，用此脚本完成数据预处理和向量索引构建。
你也可以直接启动Web界面，在界面上完成这些操作。

使用方法：
    python main.py              # 交互式菜单
    python main.py --build      # 一键完成：解析法规 + 构建索引
    python main.py --test       # 命令行测试一条审查请求
"""
import os
import sys

# 确保项目根目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def step1_parse_documents():
    """第一步：解析Word法规文件"""
    print("\n" + "=" * 60)
    print("  第一步：解析法规文档")
    print("=" * 60)
    from src.preprocess.parser import parse_all_docx, save_articles, print_statistics

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "processed", "articles.json"
    )
    articles = parse_all_docx()
    save_articles(articles, output_path)
    print_statistics(articles)
    return output_path


def step2_build_index(articles_path: str):
    """第二步：构建向量索引"""
    print("\n" + "=" * 60)
    print("  第二步：构建向量索引")
    print("=" * 60)
    from src.rag.vector_store import build_index
    build_index(articles_path)


def step3_test():
    """测试：用一条示例运行完整RAG流水线"""
    print("\n" + "=" * 60)
    print("  测试：运行合规审查")
    print("=" * 60)
    from src.rag.generator import generate_report

    test_cases = [
        "我们的社交App允许用户查看附近100米内的其他用户的实时位置信息",
        "用户上传身份证照片和人脸照片后，系统自动进行实名认证",
    ]

    for i, test_input in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1} ---")
        print(f"输入: {test_input}")
        report = generate_report(test_input)
        print(f"\n{report}")
        print("\n" + "-" * 40)


def main():
    if "--build" in sys.argv:
        # 一键构建模式
        articles_path = step1_parse_documents()
        step2_build_index(articles_path)
        print("\n✅ 全部完成！现在可以运行 python -m streamlit run src/ui/app.py")
    elif "--test" in sys.argv:
        # 测试模式
        step3_test()
    else:
        # 交互式菜单
        while True:
            print("\n" + "=" * 60)
            print("  法审助手 - Legal Review Assistant")
            print("=" * 60)
            print("  1. 解析Word法规文件 -> 条款级JSON")
            print("  2. 构建向量索引（ChromaDB）")
            print("  3. 一键完成 (解析 + 建索引)")
            print("  4. 测试合规审查")
            print("  5. 启动Web界面 (Streamlit)")
            print("  0. 退出")
            print("=" * 60)

            choice = input("请选择 [0-5]: ").strip()

            if choice == "1":
                step1_parse_documents()
            elif choice == "2":
                articles_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "data", "processed", "articles.json"
                )
                if not os.path.exists(articles_path):
                    print("错误: 请先执行步骤1（解析文档）")
                else:
                    step2_build_index(articles_path)
            elif choice == "3":
                articles_path = step1_parse_documents()
                step2_build_index(articles_path)
                print("\n✅ 全部完成！")
            elif choice == "4":
                step3_test()
            elif choice == "5":
                print("启动Streamlit界面...")
                import subprocess
                subprocess.run([
                    sys.executable, "-m", "streamlit", "run",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "ui", "app.py")
                ])
            elif choice == "0":
                print("再见！")
                break
            else:
                print("无效选择，请重新输入")


if __name__ == "__main__":
    main()
