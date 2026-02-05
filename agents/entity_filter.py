import json

from openai import OpenAI

import agentlightning as agl

from config import get_openai_config
from evaluators.human_feedback import get_human_score
from evaluators.llm_judge import llm_judge


@agl.rollout
def entity_filter_agent(task, prompt_template: agl.PromptTemplate) -> float:
    prompt = prompt_template.format(
        question=task["question"],
        disambiguated_entities=json.dumps(
            task["disambiguated_entities"], ensure_ascii=False
        ),
    )

    openai_config = get_openai_config()
    client = OpenAI(
        api_key=openai_config["api_key"],
        base_url=openai_config["base_url"],
    )
    resp = client.chat.completions.create(
        model=task.get("model", openai_config["model_name"]),
        messages=[{"role": "user", "content": prompt}],
    )
    output = resp.choices[0].message.content

    eval_mode = task.get("eval_mode", "llm")
    human_score = get_human_score(task)
    if eval_mode == "human":
        return human_score if human_score is not None else 0.0

    if human_score is not None:
        return human_score

    return llm_judge(
        question=task["question"],
        entities=task["disambiguated_entities"],
        output_json=output,
        goal=task["goal"],
    )
