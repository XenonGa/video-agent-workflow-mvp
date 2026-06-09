SCRIPT_SYSTEM = """你是一个专业影视编剧和分镜导演。
只输出合法 JSON，不要输出 Markdown，不要解释。
字段必须完整，内容使用中文。"""

SCRIPT_USER_TEMPLATE = """请根据用户 prompt 生成一个短片脚本结构。

用户 prompt:
{prompt}

输出 JSON schema:
{{
  "title": "短片标题",
  "logline": "一句话梗概",
  "style": "视觉风格，例如电影感、动画、写实等",
  "characters": [
    {{
      "id": "char_001",
      "name": "人物名",
      "age": "年龄",
      "gender": "性别",
      "appearance": "外貌特征",
      "costume": "服装",
      "personality": "性格",
      "role": "剧情作用"
    }}
  ],
  "scenes": [
    {{
      "id": "scene_001",
      "name": "场景名",
      "location": "地点",
      "time_of_day": "时间",
      "mood": "情绪氛围",
      "visual_description": "可用于绘图的场景描述",
      "props": ["道具1", "道具2"],
      "color_palette": "主色调"
    }}
  ],
  "shots": [
    {{
      "id": "shot_001",
      "scene_id": "scene_001",
      "description": "镜头描述",
      "characters": ["char_001"],
      "shot_size": "景别",
      "camera_angle": "机位/视角",
      "duration_seconds": 5,
      "dialogue": ["对白"]
    }}
  ]
}}

要求:
- 生成 1 到 3 个主要人物。
- 生成 1 到 3 个主要场景。
- 生成 4 到 8 个镜头。
- 人物、场景、镜头 id 必须稳定且可引用。
"""

CHARACTER_PROMPT_SYSTEM = """你是角色概念设计师。
只输出合法 JSON 数组，不要输出 Markdown，不要解释。"""

CHARACTER_PROMPT_USER_TEMPLATE = """根据以下脚本，为每个人物生成一张三视图人设图的绘图 prompt。

脚本 JSON:
{script_json}

输出 JSON 数组，每项:
{{
  "id": "char_001",
  "name": "人物名",
  "positive_prompt": "英文绘图 prompt",
  "negative_prompt": "英文负面 prompt"
}}

positive_prompt 要求:
- 必须包含 character design sheet, full body, three views, front view, side view, back view。
- 必须强调同一人物、同一服装、白色或浅灰背景、干净设定图。
- 描述年龄、体型、发型、五官、服装、关键道具。
- 不要生成多人互动场景。
"""

SCENE_PROMPT_SYSTEM = """你是影视美术概念设计师。
只输出合法 JSON 数组，不要输出 Markdown，不要解释。"""

SCENE_PROMPT_USER_TEMPLATE = """根据以下脚本，为每个场景生成一张场景概念图 prompt。

脚本 JSON:
{script_json}

输出 JSON 数组，每项:
{{
  "id": "scene_001",
  "name": "场景名",
  "positive_prompt": "英文绘图 prompt",
  "negative_prompt": "英文负面 prompt"
}}

positive_prompt 要求:
- cinematic concept art, establishing shot。
- 包含地点、时间、色彩、灯光、道具、气氛。
- 不要把人物画成主角；可以出现 very small silhouettes 但重点是场景。
"""
