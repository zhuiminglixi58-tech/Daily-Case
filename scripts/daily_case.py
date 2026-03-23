"""
每日财经案例精选 · Claude API 版本
流程：Claude web_search 联网搜索 → 筛选5个案例 → 深度分析最佳案例 → 飞书推送

Claude API 要点：
- 使用 anthropic SDK
- 联网搜索使用内置 web_search_20260209 服务端工具（由 Anthropic 托管执行）
- 推荐模型：claude-opus-4-6
"""

import os
import json
import requests
import anthropic
from datetime import datetime, timezone, timedelta

# ── 配置 ──────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
FEISHU_WEBHOOK_URL = os.environ["FEISHU_WEBHOOK_URL"]
MODEL = "claude-sonnet-4-6"

# 北京时间
BJT = timezone(timedelta(hours=8))
now_bjt = datetime.now(BJT)
today_str = now_bjt.strftime("%Y年%m月%d日")
weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now_bjt.weekday()]

# Claude 客户端
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# web_search 服务端工具声明（由 Anthropic 托管执行，无需手动处理搜索结果）
WEB_SEARCH_TOOL = [
    {"type": "web_search_20260209", "name": "web_search"}
]


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def extract_json_text(text: str) -> str:
    """从模型输出中尽量提取最外层 JSON 文本。"""
    text = (text or "").strip()

    if not text:
        return ""

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].lstrip("json").strip()
        else:
            text = text.strip("`").lstrip("json").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    return text


# ── 核心：带 web_search 服务端工具的多轮 chat ──────────────────────────────────
def chat_with_search(system: str, user: str, max_tokens: int = 6000) -> str:
    messages = [
        {"role": "user", "content": user},
    ]

    last_text = ""

    for _ in range(10):
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=WEB_SEARCH_TOOL,
        )

        # 提取文本内容
        for block in response.content:
            if block.type == "text":
                last_text = block.text
                break

        if response.stop_reason == "end_turn":
            return last_text

        # 服务端工具触发迭代上限（pause_turn）或 tool_use，追加 assistant 消息并继续
        if response.stop_reason in ("pause_turn", "tool_use"):
            messages.append({"role": "assistant", "content": response.content})
            continue

        return last_text

    return last_text


# ── Step 1: 筛选5个候选案例 ───────────────────────────────────────────────────
def fetch_five_cases() -> list[dict]:
    print("▶ Step 1: Claude 联网搜索，筛选5个案例...")

    system = f"""你是商学院案例研究助手。今天是{today_str}（{weekday_cn}）。

任务：联网搜索凤凰网财经（finance.ifeng.com）、新浪财经（finance.sina.com.cn）、腾讯财经（finance.qq.com）近7天内发布的财经报道，筛选出5篇最适合商学院案例分析写作的文章。

筛选标准：
1. 来自上述三个平台的原创或深度报道（非纯通稿）
2. 记者/编辑有明确的初步判断或论点（如"这说明XX"、"背后反映的是XX"、"值得警惕的是XX"）
3. 报道本身没有详细数据论证或理论分析——留有充分的写作论证空间
4. 优先选非头条、有深度但关注度不算最高的案例（避免过度曝光话题）
5. 覆盖不同行业和议题类型

严格只返回JSON，不含markdown、注释或多余文字：
{{"cases":[
  {{
    "rank": 1,
    "title": "文章原标题",
    "source": "凤凰财经|新浪财经|腾讯财经",
    "date": "发布日期如3月22日",
    "category": "商业模式|危机管理|创新颠覆|战略转型|组织管理",
    "company": "核心企业名称",
    "industry": "所属行业",
    "journalist_view": "记者核心论点（60-80字）",
    "writing_gap": "报道论证空白——案例写作切入点（50-70字）",
    "why_pick": "为什么值得写作（一句话20-30字）",
    "url": "文章原始URL"
  }}
]}}"""

    user = f"请联网搜索凤凰财经、新浪财经、腾讯财经近7天的企业深度报道，筛选5篇有记者初步判断但缺乏深入论证的文章，按JSON返回。今天是{today_str}。"

    raw = chat_with_search(system, user, max_tokens=4000)
    clean = extract_json_text(raw)

    print("RAW RESPONSE:")
    print(raw)
    print("CLEAN RESPONSE:")
    print(clean)

    if not clean:
        raise RuntimeError("Claude 返回了空内容，无法解析 cases JSON")

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Claude 返回内容不是合法 JSON: {clean}") from e

    if "cases" not in data:
        raise RuntimeError(f"Claude 返回内容里没有 cases 字段: {clean}")

    cases = data["cases"]
    print(f"  ✓ 获取到 {len(cases)} 个案例")
    return cases


# ── Step 2: 深度分析最佳案例 ──────────────────────────────────────────────────
def analyze_top_case(cases: list[dict]) -> dict:
    print("▶ Step 2: 深度分析最值得关注的案例...")

    cases_summary = "\n".join(
        f"{c['rank']}. 【{c['source']}】{c['title']}（{c['company']}·{c['category']}）\n"
        f"   记者论点：{c['journalist_view']}\n"
        f"   写作空白：{c['writing_gap']}"
        for c in cases
    )

    system = f"""你是商学院教授级案例研究专家，擅长从财经新闻中挖掘深层商业逻辑。今天是{today_str}。

从给定的5个候选案例中，选出最值得深度写作的1个，并进行完整的商学院式深度分析。

选案标准（综合评估）：
- 论证空间大（记者论点清晰，但未展开）
- 议题普适性强（有商学院教学价值）
- 数据可获取（可通过搜索补充关键数据）
- 分析框架适配（能匹配成熟的管理学理论）

深度分析须包含：
- 推荐理由（3-4句，说明为何选这个而非其他）
- 核心研究问题（1个精准问题）
- 事件背景与关键数据（结合联网搜索补充，200字左右）
- 核心矛盾界定（2-3个具体管理/战略矛盾，每个说明"记者提出了什么、但没有论证什么"）
- 推荐分析框架（2个，说明适用角度和关键问题）
- 写作提纲（5章，每章标题+核心论点30-50字）
- 管理启示（3条，可用于结论段）

严格只返回JSON，不含markdown、注释或多余文字：
{{
  "selected_rank": 数字,
  "reason": "推荐理由",
  "research_question": "核心研究问题",
  "background": "背景与关键数据",
  "contradictions": [
    {{"title": "矛盾名称", "journalist_said": "记者提出了什么", "gap": "但没有论证什么"}}
  ],
  "frameworks": [
    {{"name": "框架名称", "angle": "适用角度", "key_questions": "关键问题"}}
  ],
  "outline": [
    {{"chapter": "章节标题", "thesis": "核心论点"}}
  ],
  "insights": ["启示1", "启示2", "启示3"]
}}"""

    user = f"今天的5个候选案例：\n\n{cases_summary}\n\n请选出最值得深度写作的1个，进行完整商学院式分析，返回JSON。"

    raw = analyze_with_search(system, user)
    clean = extract_json_text(raw)

    print("ANALYSIS RAW RESPONSE:")
    print(raw)
    print("ANALYSIS CLEAN RESPONSE:")
    print(clean)

    if not clean:
        raise RuntimeError("Claude 返回了空内容，无法解析 analysis JSON")

    try:
        analysis = json.loads(clean)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Claude 返回的 analysis 不是合法 JSON: {clean}") from e

    print(f"  ✓ 选中第 {analysis['selected_rank']} 个案例，完成深度分析")
    return analysis


def analyze_with_search(system: str, user: str) -> str:
    """分析步骤同样允许联网补充数据"""
    return chat_with_search(system, user, max_tokens=6000)


# ── Step 3: 组装飞书卡片消息 ──────────────────────────────────────────────────
def build_feishu_card(cases: list[dict], analysis: dict) -> dict:
    top = cases[analysis["selected_rank"] - 1]

    cat_emoji = {
        "商业模式": "🔵", "危机管理": "🔴",
        "创新颠覆": "🟢", "战略转型": "🟠", "组织管理": "🟣"
    }

    # 5个案例速览
    cases_md = []
    for i, c in enumerate(cases):
        star = "⭐ " if (i + 1) == analysis["selected_rank"] else ""
        em = cat_emoji.get(c["category"], "⚪")
        cases_md.append(
            f"{star}{em} **{i+1}. {c['title']}**\n"
            f"   {c['source']} · {c['company']} · {c['date']}\n"
            f"   记者论点：{c['journalist_view'][:65]}…\n"
            f"   写作空白：{c['writing_gap'][:50]}…"
        )

    # 矛盾块
    contradictions_md = "\n\n".join(
        f"**{ct['title']}**\n记者提出：{ct['journalist_said']}\n论证空白：{ct['gap']}"
        for ct in analysis["contradictions"]
    )

    # 框架块
    frameworks_md = "\n\n".join(
        f"**{fw['name']}**（{fw['angle']}）\n{fw['key_questions']}"
        for fw in analysis["frameworks"]
    )

    # 写作提纲
    outline_md = "\n".join(
        f"{j+1}. **{ch['chapter']}**\n   {ch['thesis']}"
        for j, ch in enumerate(analysis["outline"])
    )

    # 管理启示
    insights_md = "\n".join(f"• {ins}" for ins in analysis["insights"])

    return {
        "msg_type": "interactive",
        "card": {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"📰 每日财经案例精选 · {today_str}"
                },
                "subtitle": {
                    "tag": "plain_text",
                    "content": f"来源：凤凰财经 · 新浪财经 · 腾讯财经 | {weekday_cn} · 案例写作选题库"
                },
                "template": "blue"
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": "## 今日5个候选案例\n\n" + "\n\n".join(cases_md)
                    },
                    {"tag": "hr"},
                    {
                        "tag": "markdown",
                        "content": (
                            f"## ⭐ 今日重点深度分析\n\n"
                            f"**{top['title']}**\n"
                            f"{top['source']} · {top['company']} · {top['industry']} · {top['date']}"
                        )
                    },
                    {
                        "tag": "markdown",
                        "content": f"**推荐理由**\n{analysis['reason']}"
                    },
                    {
                        "tag": "markdown",
                        "content": f"**核心研究问题**\n> {analysis['research_question']}"
                    },
                    {"tag": "hr"},
                    {
                        "tag": "markdown",
                        "content": f"### 一、事件背景与关键数据\n\n{analysis['background']}"
                    },
                    {"tag": "hr"},
                    {
                        "tag": "markdown",
                        "content": f"### 二、核心矛盾界定\n\n{contradictions_md}"
                    },
                    {"tag": "hr"},
                    {
                        "tag": "markdown",
                        "content": f"### 三、推荐分析框架\n\n{frameworks_md}"
                    },
                    {"tag": "hr"},
                    {
                        "tag": "markdown",
                        "content": f"### 四、写作提纲\n\n{outline_md}"
                    },
                    {"tag": "hr"},
                    {
                        "tag": "markdown",
                        "content": f"### 五、管理启示\n\n{insights_md}"
                    },
                    {"tag": "hr"},
                    {
                        "tag": "markdown",
                        "content": f"🔗 [查看原文]({top.get('url', '#')})"
                    }
                ]
            }
        }
    }


# ── Step 4: 发送飞书 ──────────────────────────────────────────────────────────
def send_to_feishu(card: dict):
    print("▶ Step 3: 发送飞书消息...")
    resp = requests.post(FEISHU_WEBHOOK_URL, json=card, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") != 0:
        raise RuntimeError(f"飞书返回错误: {result}")
    print(f"  ✓ 飞书发送成功: {result}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*52}")
    print(f"  每日财经案例精选（Claude Sonnet 4.6）· {today_str} {weekday_cn}")
    print(f"{'='*52}\n")

    cases = fetch_five_cases()
    analysis = analyze_top_case(cases)
    card = build_feishu_card(cases, analysis)
    send_to_feishu(card)

    print("\n✅ 全部完成！")


if __name__ == "__main__":
    main()
