# output_schema 含义与迭代时结构传承

## 1. output_schema 在 agent-lightning 中的含义

- **Agent Lightning 官方**：APO 文档中**没有** `output_schema` 概念，APO 只优化「单个 prompt 模板」的文本，不涉及 schema 配置。
- **本项目中**：`output_schema: entity_filter_v1` 是**项目自定义**的配置项，表示：
  - 该节点（entity_filter）的**输出格式/结构**遵循名为 `entity_filter_v1` 的规范；
  - 对应文件为 `src/configs/nodes/entity_filter_v1.yaml`，其中 prompt 里约定了同一套输出格式（`EntityID-Role-Confidence | ...`）。
- **当前实现**：`train.py` 并未读取或使用 `output_schema`，因此它目前主要是**语义约定/文档**，或为后续「按 schema 约束 prompt 优化」预留。

## 2. 下游对“结构”的依赖

- **解析**：`convert_results_to_dict(entities, api_result)` 依赖固定格式：
  - 每段为 `id-role-confidence`，段之间用 `|` 分隔；
  - 若 LLM 输出格式变化（如改成 JSON 或别的分隔符），解析会失败或结果错位。
- **评估**：`llm_judge.py` 中 `_extract_roles_from_string` 同样按 `|` 和 `-` 解析；`score_with_gold` 依赖 `gold_struct` 或字符串格式一致。

因此「输出结构」必须与当前约定一致，否则 pipeline 会坏。

## 3. 如何防止迭代更新时结构被破坏（保证结构传承）

| 手段 | 说明 |
|------|------|
| **在 prompt 中固化输出格式** | 在 prompt_template 里保留醒目的「输出格式要求」段落（如 entity_filter.yaml 中的「输出形如：...」「用 \| 分隔」），并尽量让 APO 少改这部分。 |
| **使用 APO 的 gradient/apply_edit 约束** | 通过 `gradient_prompt_files` / `apply_edit_prompt_files` 在「编辑指导」中明确写：**必须保留输出格式**（`id-role-confidence`，用 `\|` 分隔），不得删除或改写该段。 |
| **评估时惩罚格式错误** | 在 rollout 的 reward 中：若 `convert_results_to_dict` 解析失败或解析出的条数异常，直接给低分（如 0），使格式错误的候选 prompt 被淘汰。 |
| **在代码中真正使用 output_schema** | 加载 `output_schema` 指向的 yaml（如 entity_filter_v1），提取其中「输出格式说明」作为**不可变片段**：要么拼进 initial prompt 的固定块，要么在 apply_edit 的 system/约束中传入，保证每次优化都带「必须遵守该格式」的约束。 |

推荐组合：**固化 prompt 中的格式段落 + 评估时格式合法性检查 +（可选）在 train 里用 output_schema 注入格式约束**。
