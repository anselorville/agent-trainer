import json
import random
import copy
import requests
from datetime import datetime
import sys
from pathlib import Path

from src.client.openai_httpx import run_chat
from src.workflow.data_processor import main as process_ner_result

GENERATE_MODEL_NAME = "glm-4.5-flash"
CORRECTING_MODEL_NAME = "glm-4.7"

PROMPT_TEMPLATE = """## 任务描述
你是一个专业的金融信息处理专家，专门负责识别金融问句中出现的实体的在ES存储的信息中的角色，并输出符合格式的结果。
现在有3类实体：企业、时间、人物。请根据以下定义识别实体：
- 企业：指公司、机构等组织实体，通常包含公司的全称/简称(name)、股票代码(codes)等信息。
- 时间：指与时间相关的实体，可以是具体日期、时间段、时间点等文字描述。
- 人物：指与人相关的实体，一般指人物姓名。
这些实体在问句中在ES中可能会有歧义，需要准确识别其角色。请根据以下规则处理实体：
1. 公司/机构实体角色
- **subject**: 查询主体，用户想查找关于该公司的信息
  - 示例: "腾讯的营收情况" → 腾讯是subject
- **publisher**: 发布机构，用户想查找该机构发布的文档
  - 示例: "中信证券的研报" → 中信证券是publisher

2. 人物实体角色
- **author**: 作者，用户想查找该人撰写的文档
  - 示例: "张三写的研报" → 张三是author
- **subject**: 主体，用户想查找关于该人的信息
  - 示例: "关于马云的新闻" → 马云是subject

3. 时间实体角色（最关键）
- **content_descriptor**: 内容描述时间，时间是文档内容的一部分
  - 示例: "2024年年报" → 2024年描述的是年报的周期，不是发布时间，金融场景下对于周期性纰漏信息，时间通常是描述性的内容，而非过滤条件
- **filter_time**: 过滤时间，用于过滤文档发布日期，注意：多数口语化表达的问句中，时间都可以用来作为过滤使用
  - 示例: "2024年发布的公告" → 2024年是发布时间
- **prediction_time**: 预测时间，未来时间引用
  - 示例: "2027年的市场趋势" → 2027年是预测对象
- **context**: 上下文时间，相对时间表达
  - 示例: "最近的研报" → "最近"是模糊时间范围，视语境中的信息对时间的敏感度设置默认值为近一个月/近一个星期

## Input
用户的问句：{question}
识别的实体：
{entities}

## 输出格式要求
输出形如：[id]-[role]-[confidence]|[id]-[role]-[confidence]|...
其中，id就是输入实体中的id，role是根据上述规则识别的实体角色，confidence是一个0-10之间的评分，表示模型对该角色判断的置信度。
输出值时不用[]包裹，不同实体之间的输出用|分隔，id、role、confidence之间用-连接。

## Output

"""

PROMPT_TEMPLATE_CORRECTING = """
你是一个专业的金融信息处理专家，你对于用户提出的问题中不同实体的角色非常敏感，尤其是时间实体的角色。现在有3类实体：企业、时间、人物。
现在前置处理中根据问题和实体识别的结果，已经给出了一个判断结果，追加在不同实体的属性中，你需要认真分析之后确定是否需要纠正这个结果，并输出符合格式的结果。
ES中保存了海量的金融领域中的公告、研究报告、新闻舆情等文档数据，用户在提出问题时，有可能想找文档、有可能想找文档中的内容，也有可能是一个自然问句，但是如果要回答这个问题，需要一些相关文档中的相关内容作为依据。
为了能精准且不遗漏的找到这些文档，需要结合ES的filter过滤方法和query的关键词查询方法，但是在面对企业/机构、人物与时间实体这些实体时，往往难以正确的进行利用。
前置的判断角色后，会直接角色与方法对应关系，讲实体或者作为槽位信息在ES中直接过滤，或者作为关键词查询在ES中进行搜索。
容易造成困扰的点在于：
1. 企业/机构：尤其是在研报这类数据中会频繁出现发布机构，发布机构一般既是金融机构、也可能是上市公司，所以如果要使用publishAgent进行过滤，需要确认用户在提问时提到该企业确实意指发布机构，而非查询主体。
2. 人物：人物实体一般指作者，如果直接作为过滤条件，往往会导致没有结果返回，因为用户提问时可能会提到一个人物，但是这个人物并非文档的作者，而是文档中提到的一个人物，这时候如果把这个实体作为过滤条件，就会导致无法找到相关文档。
3. 时间：时间实体的角色判断是最关键的，时间实体可能是内容描述时间，也可能是过滤时间，还可能是预测时间或者上下文时间，如果把内容描述时间作为过滤条件，就会导致无法找到相关文档，如果把过滤时间作为内容描述时间，就会导致返回大量无关文档
前置处理时，输出的结果形如：[id]-[role]-[confidence]|[id]-[role]-[confidence]|...
其中，id就是输入实体中的id，role是根据上述规则识别的实体角色，confidence是一个0-10之间的评分，表示模型对该角色判断的置信度。
企业/机构的类型分为subject或者publisher，人物的类型分为author或者subject，时间的类型分为content_descriptor、filter_time、prediction_time或者context。
publisher，author，filter_time会被处理为查询的过滤条件，其他的会作为关键词进行去ES中查询。

## Task
现在针对这个问题：{question}
前置处理给出的结果是：{pre_result}
前置处理的结果还原成json的内容为：{entities}
请你分析这个结果，判断是否需要纠正，如果需要纠正，请给出正确的结果，输出格式与前置处理一致，如果不需要纠正，请直接输出前置处理的结果，尤其要注意，切不可修改id信息。

## Output

"""

def call_ner_api(query: str) -> str:
    """
    curl --location --request POST 'http://10.106.40.74:30803/ner_pred' \
        --header 'Content-Type: application/json' \
        --data-raw '{
            "filtered": "0",
            "out_type": "tuple",
            "wind.sessionId": "11",
            "text": "最近5年恒生电子年报中关于战略的描述",
            "source": "wind.search",
            "with_weight": "1"
        }'
    """
    url = "http://10.106.40.74:30803/ner_pred"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "filtered": "0",
        "out_type": "tuple",
        "wind.sessionId": "11",
        "text": query,
        "source": "wind.search",
        "with_weight": "1"
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.text.strip()

def convert_results_to_dict(entities:dict, api_result: str) -> dict:
    # "CMV5-subject-10|2J8D-content_descriptor-9"
    results = api_result.split("|")
    result_dict = {}
    for result in results:
        parts = result.split("-")
        if len(parts) != 3:
            continue
        entity_id, role, confidence = parts
        result_dict[entity_id] = {
            "role": role,
            "confidence": confidence
        }
    
    ner_enterprise_list = entities["ner_enterprise"]
    ner_time_list = entities["ner_time"]
    ner_person_list = entities["ner_person"]

    seen_ids = []
    all_ids = [k for k,_ in result_dict.items()]

    for item1 in ner_enterprise_list:
        if item1["id"] not in seen_ids and item1["id"] in all_ids:
            item1["role"] = result_dict[item1["id"]]["role"]
            item1["confidence"] = result_dict[item1["id"]]["confidence"]
            seen_ids.append(item1["id"])

    for item2 in ner_time_list:
        if item2["id"] not in seen_ids and item2["id"] in all_ids:
            item2["role"] = result_dict[item2["id"]]["role"]
            item2["confidence"] = result_dict[item2["id"]]["confidence"]
            seen_ids.append(item2["id"])
        
    for item3 in ner_person_list:
        if item3["id"] not in seen_ids and item3["id"] in all_ids:
            item3["role"] = result_dict[item3["id"]]["role"]
            item3["confidence"] = result_dict[item3["id"]]["confidence"]
            seen_ids.append(item3["id"]) 

    entities["ner_enterprise"] = ner_enterprise_list
    entities["ner_time"] = ner_time_list
    entities["ner_person"] = ner_person_list

    return entities

def invoke_generation_api(query: str, debug: bool = False) -> dict:
    ner_result = call_ner_api(query)
    entities = process_ner_result(ner_result)

    ner_enterprise = entities["ner_enterprise"]
    ner_time = entities["ner_time"]
    ner_person = entities["ner_person"]
    is_empty_entity = len(ner_enterprise) == 0 and len(ner_time) == 0 and len(ner_person) == 0
    if is_empty_entity:
        return None, None, None

    input_text = PROMPT_TEMPLATE.format(
        question=query,
        entities=json.dumps(entities, ensure_ascii=False)
    )
    t1 = datetime.now()
    llm_result = run_chat(input_text,model=GENERATE_MODEL_NAME, temperature=0.01)
    t2 = datetime.now()
    delta = (t2 - t1).total_seconds()

    output = None
    # 处理结果
    if "choices" in llm_result and len(llm_result["choices"]) > 0:
        output = llm_result["choices"][0]["message"]["content"]
        print(f"生成模型耗时：{delta} s，输出：{output}")
    
    if debug:
        format_entities = copy.deepcopy(entities)
        convert_results_to_dict(format_entities, output)

    return output, entities, format_entities if debug else None

def invoke_correcting_api(query, entities:dict, output, format_entities) -> dict:
    input_text = PROMPT_TEMPLATE_CORRECTING.format(
        question=query,
        pre_result=output,
        entities=format_entities
    )
    t1 = datetime.now()
    llm_result = run_chat(input_text, model=CORRECTING_MODEL_NAME, temperature=0.01)
    t2 = datetime.now()
    delta = (t2 - t1).total_seconds()

    new_output = None
    # 处理结果
    if "choices" in llm_result and len(llm_result["choices"]) > 0:
        new_output = llm_result["choices"][0]["message"]["content"]
        print(f"纠偏模型耗时：{delta} s，输出：{new_output}, 结果是否一致：{new_output == output}")
        print("="*10)
    
    if new_output != None and new_output != output:
        format_entities = copy.deepcopy(entities)
        convert_results_to_dict(format_entities, new_output)

    return new_output, entities, format_entities

def pipeline(lines: list, output_path: str, sampling: bool = False, sample_size: int = 0):
    train_samples = []
    assert output_path.endswith(".jsonl"), "输出文件必须是jsonl格式"

    if sampling:
        lines = random.sample(lines, sample_size)
    if sample_size > 0 and sample_size < len(lines):
        lines = lines[:sample_size]

    for query in lines:
        ner_result = call_ner_api(query)
        entities = process_ner_result(ner_result)

        train_samples.append({
            "input":{
                "question": query,
                "entities": entities
            }
        })
    # 写入jsonl文件中
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def pipeline_with_gold(lines: list, output_path: str, sampling: bool = False, sample_size: int = 0):
    val_samples = []
    assert output_path.endswith(".jsonl"), "输出文件必须是jsonl格式"
    if sampling:
        lines = random.sample(lines, sample_size)
    if sample_size > 0 and sample_size < len(lines):
        lines = lines[:sample_size]

    for query in lines:
        output, entities, format_output = invoke_generation_api(query, True)
        if output == None:
            continue
        new_output, new_entities, new_formats = invoke_correcting_api(query, entities, output, format_output)
        val_sample = {
            "input":{
                "question": query,
                "entities": new_entities
            },
            "output": new_output,
            "format_output": new_formats
        }
        if new_output != output:
            val_sample["legacy"] = {
                "entities":entities,
                "output": output,
                "format_output": format_output,
            }
        val_samples.append(val_sample)
    # 写入jsonl文件中
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        f.close()
    # 写入json文件中，方便查看
    with open(output_path.replace(".jsonl", ".view.json"), "w", encoding="utf-8") as sf:
        sf.write(json.dumps(val_samples, ensure_ascii=False, indent=2) + "\n")
        sf.close()


if __name__ == "__main__":
    # query = "最近5年恒生电子年报中关于战略的描述"
    # output, entities, format_entities= invoke_generation_api(query,debug=True)

    # print(output)
    # print(json.dumps(format_entities, ensure_ascii=False, indent=2))
    ##################
    stamp = datetime.now().strftime("%m%d%H%M")
    input_path = "samples/source/questions_full.txt"
    # 读取文件
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines() 
    lines = [line.strip() for line in lines if line.strip() != ""]
    lines = random.sample(lines, len(lines))
    train_input = lines[:int(len(lines)*0.8)]
    val_input = lines[int(len(lines)*0.8):]

    pipeline(train_input, f"samples/entity_filter/train_{stamp}.jsonl")
    pipeline(val_input, f"samples/entity_filter/val_{stamp}.jsonl")
