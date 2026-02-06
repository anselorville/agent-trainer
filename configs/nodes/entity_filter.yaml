import json

from openai import OpenAI

import agentlightning as agl

from src.evaluators.human_feedback import get_human_score
from src.evaluators.llm_judge import llm_judge, score_with_gold

@agl.rollout
def entity_filter_agent(task, prompt_template: agl.PromptTemplate) -> float:
    entities = task.get("entities")
    if entities is None:
        raise ValueError("Missing entities in task payload")

    entities_json = json.dumps(entities, ensure_ascii=False)
    prompt = prompt_template.format(
        question=task["question"],
        entities=entities_json,
    )

    client = OpenAI(
        api_key=task.get("model_api_key"),
        base_url=task.get("model_base_url")
    )
    resp = client.chat.completions.create(
        model=task.get("model"),
        messages=[{"role": "user", "content": prompt}],
    )
    output = resp.choices[0].message.content

    eval_mode = task.get("eval_mode", "llm")
    human_score = get_human_score(task)
    if eval_mode == "human":
        return human_score if human_score is not None else 0.0

    gold_struct = task.get("gold_struct") or task.get("format_output")
    gold = task.get("gold") or task.get("output")
    if gold_struct is not None or gold is not None:
        return score_with_gold(output_json=output, gold=gold, gold_struct=gold_struct)

    if human_score is not None:
        return human_score

    return llm_judge(
        question=task["question"],
        entities=entities,
        output_json=output,
        goal=task["goal"],
    )
