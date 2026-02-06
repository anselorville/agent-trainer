import json
import random
import string
from datetime import datetime

def generate_random_id():
    # 生成一个4位长度的随机码(数字和英文字符全部大写)
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=4)).upper()

def main(ner_result: str) -> dict:
    if isinstance(ner_result, (bytes, bytearray)):
        try:
            ner_result = ner_result.decode("utf-8-sig")
        except UnicodeDecodeError:
            ner_result = ner_result.decode("gbk", errors="replace")
    elif isinstance(ner_result, str) and ner_result.startswith("\ufeff"):
        ner_result = ner_result.lstrip("\ufeff")

    payload = json.loads(ner_result) if ner_result else {}
    # 新格式直接在顶层有 data 字段，不再有 windNerPlugInfo 包装
    data_list = payload.get("data", [])
    
    # 分别收集四类NER信息
    ner_enterprise_list = []
    ner_time_list = []
    ner_location_list = []
    ner_person_list = []

    ner_enterprise_set = []
    ner_time_set = []
    ner_location_set = []
    ner_person_set = []

    reference_set = set()  # 使用 set 来去重，用于扩展词
    
    # 需要排除的 nerType（除了 enterprise、time、person）
    excluded_ner_types = {"post", "code", "index"}
    
    # 需要排除的 type
    excluded_types = {
        "bond", "commodity", "bankWealthManage", 
        "insurance", "options", "nz", "module"
    }
    
    for group in data_list:
        items = group if isinstance(group, list) else [group]
        for item in items:
            if isinstance(item, dict):
                # 一级类别
                ner_type = item.get("nerType")
                entity_name = item.get("entity")
                entity_id = item.get("id", "")
                # 二级类别
                entity_type = item.get("type", "")
                
                # 1. 处理企业实体（enterprise）
                if ner_type == "enterprise" or entity_type in ["stockCN", "stockHK", "stockUS", "stockNTB", 
                "stockTW", "stockFN"] and entity_name != "":
                    # 跳过排除的 type
                    if entity_type in excluded_types:
                        continue
                    
                    # 收集企业信息：name 和 codes
                    enterprise_info = {
                        "id": generate_random_id(),
                        "name": entity_name ,
                        "codes": [entity_id] if entity_id and not entity_id.isdigit() else []
                    }
                    # 去重：检查是否已存在相同的 name 和 codes
                    if entity_name in ner_enterprise_set:
                        for existing in ner_enterprise_list:
                            if existing["name"] == enterprise_info["name"]:
                                existing["codes"].extend(enterprise_info["codes"])
                                existing["codes"] = list(set(existing["codes"]))
                                break
                    ner_enterprise_set.append(entity_name)
                    ner_enterprise_list.append(enterprise_info)
                    
                # 2. 处理时间实体（time）
                elif ner_type == "time":
                    if entity_name and entity_name not in ner_time_set:
                        time_obj = {
                            "id":generate_random_id(),
				            "raw":entity_name
                        }
                        ner_time_list.append(time_obj)
                        ner_time_set.append(entity_name)
                    # time 类型不添加到 reference（原来就排除）

                # 处理地点实体（location）
                elif ner_type == "location":
                    if entity_name and entity_name not in ner_location_set:
                        location_obj = {
                            "id":generate_random_id(),
				            "location":entity_name
                        }
                        ner_location_list.append(location_obj)
                        ner_location_set.append(entity_name)
                
                # 处理人物实体（person）
                elif ner_type == "person":
                    if entity_name and entity_name not in ner_person_set:
                        person_obj = {
                            "id":generate_random_id(),
				            "name":entity_name
                        }
                        ner_person_list.append(person_obj)
                        ner_person_set.append(entity_name)
                
                # 跳过其他排除的 nerType
                elif ner_type in excluded_ner_types:
                    continue
                
                # 跳过排除的 type
                elif entity_type in excluded_types:
                    continue
                
                else:
                    # 其他 nerType：只使用 entity（不使用 id）添加到 reference
                    # 排除纯数字字符串
                    if entity_name and not entity_name.isdigit():
                        reference_set.add(entity_name)
                    if entity_id and not entity_id.isdigit():
                        reference_set.add(entity_id)
    
    # 将 set 转换为列表
    reference = list(reference_set)

    # 确保所有字段都有值，避免 None
    # result = {
    #     "current_date": datetime.now().strftime("%Y-%m-%d"),
    #     "ner_enterprise": json.dumps(ner_enterprise_list, ensure_ascii=False),
    #     "ner_time": json.dumps(ner_time_list, ensure_ascii=False),
    #     "ner_person": json.dumps(ner_person_list, ensure_ascii=False),
    #     "reference": json.dumps(reference, ensure_ascii=False),
    # }

    # 确保所有字段都有值，避免 None
    result = {
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "ner_enterprise": ner_enterprise_list,
        "ner_time": ner_time_list,
        "ner_person": ner_person_list,
        # "reference": reference,
    }
    
    # # 验证所有返回值都是字符串类型
    # for key, value in result.items():
    #     if not isinstance(value, str):
    #         result[key] = str(value)
    
    return result

if __name__ == "__main__":
    ner_result = """{
        "data": [
            [
                {
                    "entity": "元旦之后",
                    "type": "time",
                    "id": "",
                    "startIndex": 0,
                    "endIndex": 3,
                    "fullName": "元旦之后",
                    "nerType": "time"
                },
                {
                    "entity": "钟才平",
                    "type": "person",
                    "id": "",
                    "startIndex": 4,
                    "endIndex": 6,
                    "fullName": "钟才平",
                    "nerType": "person"
                },
                {
                    "entity": "贵州茅台",
                    "type": "stockCN",
                    "id": "600519.SH",
                    "startIndex": 11,
                    "endIndex": 14,
                    "fullName": "贵州茅台",
                    "nerType": "enterprise"
                }
            ]
        ],
        "succeed": true,
        "status_code": 200,
        "message": "",
        "from_cache": "0",
        "cost_time": 0.03
    }
    """

    result = main(ner_result)
    print(json.dumps(result, ensure_ascii=False,indent=2))
