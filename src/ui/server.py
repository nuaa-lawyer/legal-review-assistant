"""
法审助手 - 零依赖 Web 服务器
只使用 Python 内置模块，无需安装任何额外包。

启动: python src/ui/server.py
访问: http://localhost:8080
"""
import json
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.generator import generate_report, generate_followup
from src.rag.vector_store import get_collection_stats
from src.agents.orchestrator import multi_agent_review
from src.feedback.store import log_review, log_feedback, stats as feedback_stats
from src.utils.usage import reset as usage_reset, snapshot as usage_snapshot

# 模式选择: "multi_agent" 或 "single"
REVIEW_MODE = "multi_agent"

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>法审助手 - AI合规审查系统</title>
<style>
  :root {
    --bg: #f5f5f5;
    --card: #fff;
    --text: #1a1a1a;
    --sub: #666;
    --border: #e0e0e0;
    --red: #dc3545;
    --yellow: #e6a817;
    --green: #28a745;
    --blue: #2563eb;
    --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }

  .layout { display: flex; min-height: 100vh; }
  .sidebar {
    width: 320px; background: var(--card); border-right: 1px solid var(--border);
    padding: 24px; display: flex; flex-direction: column; gap: 20px;
  }
  .main { flex: 1; padding: 32px 40px; overflow-y: auto; }
  .logo { font-size: 22px; font-weight: 700; }
  .logo span { color: var(--blue); }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px;
    background: #e8f0fe; color: var(--blue); margin-left: 8px; }

  .status-card { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: var(--radius); padding: 14px; font-size: 13px; }
  .status-card .count { font-size: 28px; font-weight: 700; color: var(--green); }

  .section-title { font-size: 13px; font-weight: 600; color: var(--sub); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
  .example-btn {
    display: block; width: 100%; text-align: left; padding: 10px 14px; border: 1px solid var(--border);
    border-radius: 8px; background: var(--card); cursor: pointer; font-size: 13px; margin-bottom: 8px;
    transition: 0.15s;
  }
  .example-btn:hover { border-color: var(--blue); background: #f8faff; }
  .example-btn .cat { font-size: 11px; color: var(--blue); font-weight: 500; }

  textarea {
    width: 100%; height: 140px; padding: 14px; border: 2px solid var(--border); border-radius: var(--radius);
    font-size: 14px; font-family: inherit; resize: vertical; transition: 0.15s;
  }
  textarea:focus { outline: none; border-color: var(--blue); }

  .btn-row { display: flex; gap: 10px; margin-top: 12px; }
  .btn {
    padding: 10px 24px; border-radius: 8px; font-size: 14px; font-weight: 600;
    cursor: pointer; border: none; transition: 0.15s;
  }
  .btn-primary { background: var(--blue); color: #fff; }
  .btn-primary:hover { background: #1d4ed8; }
  .btn-secondary { background: var(--bg); color: var(--text); border: 1px solid var(--border); }
  .btn-secondary:hover { background: #e5e5e5; }

  .report { margin-top: 24px; }
  .trace-box {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: var(--radius);
    padding: 14px 18px; margin-bottom: 20px;
  }
  .trace-step { display: flex; align-items: flex-start; gap: 10px; padding: 6px 0; font-size: 13px; }
  .trace-step .icon { font-size: 16px; flex-shrink: 0; }
  .trace-step .meta { color: var(--sub); font-size: 11px; }
  .loading { text-align: center; padding: 40px; color: var(--sub); }
  .spinner { display: inline-block; width: 24px; height: 24px; border: 3px solid var(--border);
    border-top-color: var(--blue); border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .followup { margin-top: 32px; padding-top: 24px; border-top: 1px solid var(--border); }
  .followup input { width: 100%; padding: 10px 14px; border: 1px solid var(--border); border-radius: 8px; font-size: 14px; }
  .followup input:focus { outline: none; border-color: var(--blue); }

  .error { background: #fef2f2; border: 1px solid #fecaca; border-radius: var(--radius); padding: 16px; color: var(--red); }

  /* 反馈区 */
  .feedback-box { margin-top: 24px; padding: 18px; border: 1px dashed #cbd5e1; border-radius: var(--radius); background: #fbfdff; }
  .feedback-box .q { font-weight: 600; font-size: 14px; margin-bottom: 10px; }
  .fb-btn { padding: 8px 18px; border-radius: 8px; border: 1px solid var(--border); background: #fff;
    cursor: pointer; font-size: 14px; margin-right: 10px; }
  .fb-btn.active { border-color: var(--blue); background: #e8f0fe; color: var(--blue); font-weight: 600; }
  .fb-arts { margin: 12px 0; font-size: 13px; color: var(--sub); line-height: 2; }
  .fb-arts label { margin-right: 16px; cursor: pointer; }
  .feedback-box textarea { height: 64px; margin-top: 8px; }
  .fb-done { color: var(--green); font-size: 13px; margin-top: 10px; }
  .stat-row { display: flex; justify-content: space-between; font-size: 13px; color: var(--sub); }

  /* Markdown-like report styling */
  .report h2 { font-size: 18px; margin: 20px 0 10px; }
  .report h3 { font-size: 16px; margin: 16px 0 8px; }
  .report table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  .report th { background: #f8f9fa; padding: 10px 12px; text-align: left; border-bottom: 2px solid var(--border); font-weight: 600; }
  .report td { padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
  .report tr:hover td { background: #fafafa; }
  .report ul, .report ol { margin: 8px 0 8px 20px; }
  .report li { margin: 4px 0; }
  .report strong { font-weight: 600; }
  .report code { background: #f1f1f1; padding: 2px 6px; border-radius: 4px; font-size: 13px; }
  .report hr { border: none; border-top: 1px solid var(--border); margin: 20px 0; }

  @media (max-width: 768px) {
    .layout { flex-direction: column; }
    .sidebar { width: 100%; border-right: none; border-bottom: 1px solid var(--border); }
  }
</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="logo">⚖️ <span>法审助手</span><span class="badge">MVP</span></div>
    <div class="status-card">
      <div>📊 法规库状态</div>
      <div class="count" id="articleCount">--</div>
      <div style="color:var(--sub)">条条款已索引</div>
    </div>
    <div class="status-card" style="background:#f8fafc;border-color:#e2e8f0;">
      <div style="font-weight:600;">🔁 持续学习</div>
      <div class="stat-row"><span>累计审查</span><span id="fbReviews">--</span></div>
      <div class="stat-row"><span>反馈满意度</span><span id="fbSat">--</span></div>
      <div class="stat-row"><span>已采纳法条</span><span id="fbAdopted">--</span></div>
    </div>
    <div>
      <div class="section-title">📋 预设示例</div>
      <button class="example-btn" onclick="setExample(0)"><div class="cat">位置隐私</div>社交App允许用户查看附近100米内其他用户实时位置</button>
      <button class="example-btn" onclick="setExample(1)"><div class="cat">生物识别</div>用户上传身份证+人脸照片，系统自动实名认证并保存信息</button>
      <button class="example-btn" onclick="setExample(2)"><div class="cat">未成年人保护</div>面向中小学生的在线教育平台，收集学生姓名、年龄、成绩、位置</button>
      <button class="example-btn" onclick="setExample(3)"><div class="cat">数据跨境</div>跨境电商App将中国用户订单数据和收货地址传输到海外服务器</button>
      <button class="example-btn" onclick="setExample(4)"><div class="cat">敏感信息</div>健康管理App收集用户心率、睡眠、步数等健康数据生成健康报告</button>
    </div>
    <div style="margin-top:auto; font-size:11px; color:var(--sub);">
      法学 + AI + 产品 交叉能力展示项目
    </div>
  </aside>

  <main class="main">
    <h1 style="font-size:24px; margin-bottom:4px;">AI合规审查 <span style="font-size:12px;background:#e8f0fe;color:#2563eb;padding:3px 8px;border-radius:12px;vertical-align:middle;">多Agent协作模式</span></h1>
    <p style="color:var(--sub); margin-bottom:20px;">三个AI Agent协作：分类器 → 检索器 → 报告器，输出结构化合规分析</p>

    <label style="font-weight:600; font-size:14px;">📝 产品功能描述</label>
    <textarea id="inputText" placeholder="请描述你要审查的产品功能，例如：我们的社交App允许用户查看附近100米内的其他用户位置信息..."></textarea>
    <div class="btn-row">
      <button class="btn btn-primary" onclick="submitReview()">🔍 开始审查</button>
      <button class="btn btn-secondary" onclick="clearAll()">🗑️ 清空</button>
    </div>

    <div id="reportArea" class="report"></div>
    <div id="feedbackArea"></div>
    <div id="followupArea" class="followup" style="display:none;">
      <label style="font-weight:600; font-size:14px;">💬 追问修改建议</label>
      <div style="display:flex; gap:10px; margin-top:8px;">
        <input id="followupInput" placeholder="例如：如果去掉实时位置，只显示距离，风险会变化吗？" style="flex:1;">
        <button class="btn btn-primary" onclick="submitFollowup()">发送</button>
      </div>
    </div>
  </main>
</div>

<script>
const examples = [
  "我们的社交App允许用户查看附近100米内的其他用户的实时位置信息",
  "用户上传身份证照片和人脸照片后，系统自动进行实名认证并保存信息",
  "面向中小学生的在线教育平台，收集学生姓名、年龄、成绩、位置信息",
  "跨境电商App将中国用户的订单数据和收货地址传输到海外服务器处理",
  "健康管理App收集用户心率、睡眠、步数等健康数据，用于生成健康报告"
];

let currentInput = "";
let currentReviewId = "";
let currentArticles = [];
let fbRating = "";

async function init() {
  try {
    let r = await fetch("/api/stats");
    let d = await r.json();
    document.getElementById("articleCount").textContent = d.total_articles || "--";
  } catch(e) {}
  loadFeedbackStats();
}

async function loadFeedbackStats() {
  try {
    let r = await fetch("/api/feedback_stats");
    let d = await r.json();
    document.getElementById("fbReviews").textContent = d.reviews ?? "--";
    document.getElementById("fbSat").textContent =
      (d.satisfaction === null || d.satisfaction === undefined) ? "暂无" : Math.round(d.satisfaction*100)+"%";
    document.getElementById("fbAdopted").textContent = d.adopted_articles ?? "--";
  } catch(e) {}
}

function setExample(i) {
  document.getElementById("inputText").value = examples[i];
}

async function submitReview() {
  let text = document.getElementById("inputText").value.trim();
  if (!text) return;

  currentInput = text;
  let area = document.getElementById("reportArea");
  area.innerHTML = '<div class="loading"><div class="spinner"></div><p>正在检索相关法规并生成合规报告...</p></div>';

  try {
    let r = await fetch("/api/review", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({input: text})
    });
    let d = await r.json();
    if (d.error) {
      area.innerHTML = '<div class="error">'+d.error+'</div>';
    } else {
      let html = "";
      if (d.trace) {
        html += '<div class="trace-box">';
        html += '<div style="font-weight:600;margin-bottom:8px;">🤖 多Agent协作过程</div>';
        for (let t of d.trace) {
          html += '<div class="trace-step"><span class="icon">'+(t.agent.includes("分类")?"🔍":t.agent.includes("检索")?"📚":"📝")+'</span>';
          html += '<div><strong>'+t.agent+'</strong> <span class="meta">('+t.elapsed+'s)</span>';
          html += '<div style="color:var(--sub);">'+t.summary+'</div></div></div>';
        }
        html += '<div style="font-size:11px;color:var(--sub);margin-top:6px;">总耗时: '+d.total_elapsed+'s';
        if (d.usage) { html += ' · Token: '+d.usage.total_tokens+' · 成本约 ¥'+d.usage.cost_cny+' · LLM调用: '+d.usage.llm_calls+' 次'; }
        html += '</div>';
        html += '</div>';
      }
      html += markdownToHtml(d.report);
      area.innerHTML = html;
      document.getElementById("followupArea").style.display = "block";

      currentReviewId = d.review_id || "";
      currentArticles = d.articles || [];
      fbRating = "";
      renderFeedback();
      loadFeedbackStats();
    }
  } catch(e) {
    area.innerHTML = '<div class="error">请求失败: '+e.message+'</div>';
  }
}

async function submitFollowup() {
  let q = document.getElementById("followupInput").value.trim();
  if (!q) return;

  let area = document.getElementById("reportArea");
  area.innerHTML = '<div class="loading"><div class="spinner"></div><p>正在分析追问...</p></div>';

  try {
    let r = await fetch("/api/followup", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({original: currentInput, question: q})
    });
    let d = await r.json();
    if (d.error) {
      area.innerHTML = '<div class="error">'+d.error+'</div>';
    } else {
      area.innerHTML = markdownToHtml(d.report);
    }
  } catch(e) {
    area.innerHTML = '<div class="error">请求失败: '+e.message+'</div>';
  }
}

function renderFeedback() {
  if (!currentReviewId) return;
  let arts = currentArticles.map((a, i) =>
    '<label><input type="checkbox" class="adopt-box" value="'+a.replace(/"/g,"&quot;")+'"> '+a+'</label>').join("");
  let html = '<div class="feedback-box">';
  html += '<div class="q">📝 这份报告准确吗？（反馈会进入持续学习闭环，用于优化检索与判断口径）</div>';
  html += '<div><button class="fb-btn" id="fbUp" onclick="setRating(\'up\')">👍 准确可用</button>';
  html += '<button class="fb-btn" id="fbDown" onclick="setRating(\'down\')">👎 不准确 / 有遗漏</button></div>';
  if (arts) html += '<div class="fb-arts"><div style="margin-bottom:4px;">实际采纳了哪些法条：</div>'+arts+'</div>';
  html += '<textarea id="fbComment" placeholder="可选：具体哪里不对 / 遗漏了哪条法规 / 建议如何改进"></textarea>';
  html += '<div style="margin-top:10px;"><button class="btn btn-primary" onclick="submitFeedback()">提交反馈</button>';
  html += '<span id="fbDone" class="fb-done"></span></div></div>';
  document.getElementById("feedbackArea").innerHTML = html;
}

function setRating(r) {
  fbRating = r;
  document.getElementById("fbUp").className = "fb-btn" + (r==="up" ? " active" : "");
  document.getElementById("fbDown").className = "fb-btn" + (r==="down" ? " active" : "");
}

async function submitFeedback() {
  if (!fbRating) { alert("请先选择 👍 或 👎"); return; }
  let adopted = Array.from(document.querySelectorAll(".adopt-box:checked")).map(b => b.value);
  let comment = document.getElementById("fbComment").value.trim();
  try {
    let r = await fetch("/api/feedback", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({review_id: currentReviewId, rating: fbRating,
                            adopted_articles: adopted, comment: comment})
    });
    let d = await r.json();
    if (d.error) { document.getElementById("fbDone").textContent = "提交失败: "+d.error; }
    else {
      document.getElementById("fbDone").textContent = "✓ 已记录，感谢反馈（已采纳 "+adopted.length+" 条法条）";
      loadFeedbackStats();
    }
  } catch(e) {
    document.getElementById("fbDone").textContent = "提交失败: "+e.message;
  }
}

function clearAll() {
  document.getElementById("inputText").value = "";
  document.getElementById("reportArea").innerHTML = "";
  document.getElementById("feedbackArea").innerHTML = "";
  document.getElementById("followupArea").style.display = "none";
  document.getElementById("followupInput").value = "";
  currentInput = "";
  currentReviewId = "";
  currentArticles = [];
  fbRating = "";
}

// 简易 Markdown → HTML
function markdownToHtml(md) {
  let html = md
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    // 标题
    .replace(/^#### (.+)$/gm,"<h4>$1</h4>")
    .replace(/^### (.+)$/gm,"<h3>$1</h3>")
    .replace(/^## (.+)$/gm,"<h2>$1</h2>")
    // 粗体
    .replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>")
    // 表格
    .replace(/^\|(.+)\|$/gm, function(m) {
      let cells = m.split("|").filter(c => c.trim());
      if (cells.every(c => /^[-: ]+$/.test(c.trim()))) return "<!--sep-->";
      let tag = m.startsWith("|") && m.endsWith("|") ? "td" : "th";
      return "<tr>" + cells.map(c => "<"+tag+">"+c.trim()+"</"+tag+">").join("") + "</tr>";
    })
    // 列表
    .replace(/^- (.+)$/gm,"<li>$1</li>")
    .replace(/^\d+\. (.+)$/gm,"<li>$1</li>")
    // 代码
    .replace(/`([^`]+)`/g,"<code>$1</code>")
    // 分割线
    .replace(/^---+$/gm,"<hr>")
    // 段落
    .replace(/\n\n/g,"</p><p>")
    .replace(/\n/g,"<br>");

  // 包裹表格
  html = html.replace(/((?:<tr>.*?<\/tr>\s*)+)/g, function(m) {
    if (m.includes("<!--sep-->")) {
      let rows = m.replace(/<!--sep-->/g, "").match(/<tr>.*?<\/tr>/g) || [];
      let thead = rows[0] ? "<thead>"+rows[0].replace(/<td>/g,"<th>").replace(/<\/td>/g,"</th>")+"</thead>" : "";
      let tbody = rows.length > 1 ? "<tbody>"+rows.slice(1).join("")+"</tbody>" : "";
      return "<table>"+thead+tbody+"</table>";
    }
    return m;
  });

  // 包裹列表项
  html = html.replace(/((?:<li>.*?<\/li>\s*)+)/g, "<ul>$1</ul>");

  return "<p>" + html + "</p>";
}

init();
</script>
</body>
</html>"""


class APIHandler(BaseHTTPRequestHandler):
    """处理 API 请求 + 静态页面"""

    review_history = []  # 简单的对话历史存储

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif path == "/api/stats":
            try:
                stats = get_collection_stats()
                self._send_json(stats)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif path == "/api/feedback_stats":
            try:
                self._send_json(feedback_stats())
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/review":
            try:
                body = self._read_body()
                user_input = body.get("input", "").strip()
                if not user_input:
                    self._send_json({"error": "请输入产品功能描述"}, 400)
                    return

                print(f"\n收到审查请求: {user_input[:60]}...")

                usage_reset()
                if REVIEW_MODE == "multi_agent":
                    result = multi_agent_review(user_input, verbose=True)
                    report = result["report"]
                    usage = usage_snapshot()
                    result["usage"] = usage
                    review_id = log_review(user_input, result, mode="multi_agent")
                    self._send_json({
                        "report": report,
                        "trace": result["trace"],
                        "mode": "multi_agent",
                        "total_elapsed": result["total_elapsed"],
                        "usage": usage,
                        "review_id": review_id,
                        "articles": [f"{a.get('law_name')} {a.get('article_id')}"
                                     for a in (result.get("articles") or [])],
                    })
                else:
                    report = generate_report(user_input)
                    usage = usage_snapshot()
                    review_id = log_review(user_input, {"report": report}, mode="single")
                    self._send_json({
                        "report": report,
                        "mode": "single",
                        "usage": usage,
                        "review_id": review_id,
                    })

                self.review_history = [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": report},
                ]
            except Exception as e:
                print(f"审查错误: {e}")
                self._send_json({"error": f"审查失败: {str(e)}"}, 500)

        elif path == "/api/feedback":
            try:
                body = self._read_body()
                review_id = body.get("review_id", "")
                rating = body.get("rating", "neutral")
                if not review_id:
                    self._send_json({"error": "缺少 review_id"}, 400)
                    return
                record = log_feedback(
                    review_id=review_id,
                    rating=rating,
                    adopted_articles=body.get("adopted_articles") or [],
                    comment=body.get("comment", ""),
                    source="local_web",
                )
                print(f"  收到反馈: {rating} ({review_id})")
                self._send_json({"ok": True, "feedback": record,
                                 "stats": feedback_stats()})
            except Exception as e:
                print(f"反馈记录错误: {e}")
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/followup":
            try:
                body = self._read_body()
                original = body.get("original", "")
                question = body.get("question", "").strip()
                if not question:
                    self._send_json({"error": "请输入追问内容"}, 400)
                    return

                print(f"收到追问: {question[:60]}...")
                report = generate_followup(original, question, self.review_history)
                self.review_history.append({"role": "user", "content": question})
                self.review_history.append({"role": "assistant", "content": report})
                self._send_json({"report": report})
            except Exception as e:
                print(f"追问错误: {e}")
                self._send_json({"error": f"追问失败: {str(e)}"}, 500)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"  [{self.command}] {args[0]}")


def main():
    port = 8080
    print("=" * 50)
    print("  法审助手 Web 服务启动")
    print("=" * 50)
    print(f"\n  打开浏览器访问: http://localhost:{port}\n")
    print("  按 Ctrl+C 停止服务\n")

    server = HTTPServer(("0.0.0.0", port), APIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()
