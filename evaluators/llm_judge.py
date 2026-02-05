from openai import OpenAI

from config import get_openai_config


RUBRIC = (
    "评分0~1：\n"
    "- 结构合法性 0.2\n"
    "- 角色正确性 0.4\n"
    "- filter 正确性 0.4\n"
    "错误进入 publishDate 一律扣分。"
)


def llm_judge(question, entities, output_json, goal):
    openai_config = get_openai_config()
    client = OpenAI(
        api_key=openai_config["api_key"],
        base_url=openai_config["base_url"]
    )
    prompt = (
        "你是评分器。\n"
        f"目标: {goal}\n"
        f"用户问句: {question}\n"
        f"已识别实体: {entities}\n"
        f"模型输出: {output_json}\n"
        f"评分规则: {RUBRIC}\n"
        "只输出0~1小数。"
    )
    resp = client.chat.completions.create(
        model=openai_config["model_name"],
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        score = float(resp.choices[0].message.content.strip())
    except (TypeError, ValueError):
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
