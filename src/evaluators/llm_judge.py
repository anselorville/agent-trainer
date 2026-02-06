from openai import OpenAI

from settings import OPTIMIZER_CONFIG
from src.client.openai_httpx import build_httpx_client


RUBRIC = (
    "评分0~1：\n"
    "- 结构合法性 0.2\n"
    "- 角色正确性 0.4\n"
    "- filter 正确性 0.4\n"
    "错误进入 publishDate 一律扣分。"
)

JUDGE_GUIDE = (
    "角色判断要点：\n"
    "- 机构: subject(查询主体) vs publisher(发布机构)\n"
    "- 人物: author(作者) vs subject(主体)\n"
    "- 时间: content_descriptor / filter_time / prediction_time / context\n"
    "- 时间不要误当 publishDate 过滤\n"
)


def _extract_roles_from_structured(entities: dict) -> dict:
    roles = {}
    if not isinstance(entities, dict):
        return roles
    for key in ("ner_enterprise", "ner_time", "ner_person"):
        for item in entities.get(key, []) or []:
            entity_id = item.get("id")
            role = item.get("role")
            if entity_id and role:
                roles[entity_id] = role
    return roles


def _extract_roles_from_string(output: str) -> dict:
    roles = {}
    if not output:
        return roles
    for chunk in output.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split("-", 2)
        if len(parts) < 2:
            continue
        entity_id = parts[0].strip()
        role = parts[1].strip()
        if entity_id and role:
            roles[entity_id] = role
    return roles


def score_with_gold(output_json, gold=None, gold_struct=None) -> float:
    if gold_struct is not None:
        gold_roles = _extract_roles_from_structured(gold_struct)
    else:
        gold_roles = _extract_roles_from_string(gold or "")
    pred_roles = _extract_roles_from_string(output_json or "")

    if not gold_roles and not pred_roles:
        return 1.0
    if not gold_roles or not pred_roles:
        return 0.0

    correct = sum(
        1 for entity_id, role in pred_roles.items() if gold_roles.get(entity_id) == role
    )
    precision = correct / len(pred_roles) if pred_roles else 0.0
    recall = correct / len(gold_roles) if gold_roles else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def llm_judge(question, entities, output_json, goal):
    client = OpenAI(
        api_key=OPTIMIZER_CONFIG.api_key,
        base_url=OPTIMIZER_CONFIG.base_url,
        http_client=build_httpx_client(),
    )
    prompt = (
        "你是评分器。\n"
        f"目标: {goal}\n"
        f"用户问句: {question}\n"
        f"已识别实体: {entities}\n"
        f"模型输出: {output_json}\n"
        f"评分规则: {RUBRIC}\n"
        f"{JUDGE_GUIDE}\n"
        "只输出0~1小数。"
    )
    
    from src.utils.rate_limiter import limiter
    limiter.wait()

    resp = client.chat.completions.create(
        model=OPTIMIZER_CONFIG.model_name,
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
