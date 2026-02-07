import json

from openai import OpenAI
from src.workflow.prepare_data import convert_results_to_dict

import agentlightning as agl

from src.evaluators.human_feedback import get_human_score
from src.evaluators.llm_judge import llm_judge, score_with_gold

@agl.rollout
def entity_filter_agent(task, prompt_template: agl.PromptTemplate) -> float:
    entities = task.get("entities")
    if entities is None:
        raise ValueError("Missing entities in task payload")

    entities_json = json.dumps(entities, ensure_ascii=False)
    
    # Define all known variables that might be used in prompt templates
    # APO may generate prompts with additional variables, so we provide safe defaults
    valid_roles = """subject (查询主体), publisher (发布机构), author (作者), filter_time (过滤时间), prediction_time (预测时间), context (背景信息)"""
    
    format_kwargs = {
        "question": task["question"],
        "entities": entities_json,
        # Fallback values for variables APO might introduce
        "valid_roles": valid_roles,
        "roles": valid_roles,
        "role_list": valid_roles,
        "available_roles": valid_roles,
        "task": task.get("goal", "Identify entity roles"),
        "goal": task.get("goal", "Identify entity roles"),
        "instructions": "Assign exactly one role to each entity with a confidence score.",
        "format": "EntityID-Role-Confidence | EntityID-Role-Confidence",
        "example": "USCF-subject-0.9 | KJOC-filter_time-0.8",
    }
    
    try:
        prompt = prompt_template.format(**format_kwargs)
    except KeyError as e:
        # If there's still a missing key, add it with a placeholder and retry
        missing_key = str(e).strip("'")
        format_kwargs[missing_key] = f"[{missing_key}]"
        prompt = prompt_template.format(**format_kwargs)

    from src.utils.rate_limiter import limiter
    limiter.wait()

    client = OpenAI(
        api_key=task.get("model_api_key"),
        base_url=task.get("model_base_url")
    )
    resp = client.chat.completions.create(
        model=task.get("model"),
        messages=[{"role": "user", "content": prompt}],
    )
    output = resp.choices[0].message.content

    # 输出是一个结果字符串，需要结合entities还原成json
    # convert_results_to_dict(entities:dict, api_result: str) 
    output_json = convert_results_to_dict(entities, output)

    eval_mode = task.get("eval_mode", "llm")
    human_score = get_human_score(task)
    if eval_mode == "human":
        return human_score if human_score is not None else 0.0

    gold_struct = task.get("gold_struct") or task.get("format_output")
    gold = task.get("gold") or task.get("output")
    if gold_struct is not None or gold is not None:
        return score_with_gold(output_json=output, gold=gold, gold_struct=gold_struct)
    else:
        return score_with_gold(output_json=output, gold=gold, gold_struct=output_json)

    if human_score is not None:
        return human_score

    return llm_judge(
        question=task["question"],
        entities=entities,
        output_json=output,
        goal=task["goal"],
    )
