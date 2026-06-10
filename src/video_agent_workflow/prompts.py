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

STORYBOARD_SYSTEM = """你是专业分镜导演和摄影指导。
只输出合法 JSON 数组，不要输出 Markdown，不要解释。"""

STORYBOARD_USER_TEMPLATE = """根据脚本、人物参考图路径和场景参考图路径，为每个镜头生成可执行分镜计划。

脚本 JSON:
{script_json}

人物参考图:
{character_images_json}

场景参考图:
{scene_images_json}

输出 JSON 数组，每项:
{{
  "shot_id": "shot_001",
  "scene_id": "scene_001",
  "duration_seconds": 5,
  "shot_size": "wide shot / medium shot / close-up",
  "camera_angle": "eye level / low angle / high angle / over shoulder",
  "camera_movement": "static / slow dolly in / pan left / handheld",
  "blocking": [
    {{
      "character_id": "char_001",
      "position": "left / center / right / foreground / background",
      "action": "动作",
      "scale": "small / medium / large"
    }}
  ],
  "props": ["道具"],
  "color_palette": "色彩与光线",
  "visual_prompt": "英文视频生成 prompt，包含人物站位、景别、视角、道具、色彩、动作、电影感",
  "negative_prompt": "英文负面 prompt"
}}

要求:
- 每个脚本镜头都必须有一个分镜项。
- visual_prompt 要适合图生视频/参考图生视频模型。
- blocking 必须描述人物站位和动作。
"""

AUDIO_PLAN_SYSTEM = """你是电影声音设计师。
只输出合法 JSON，不要输出 Markdown，不要解释。"""

AUDIO_PLAN_USER_TEMPLATE = """根据以下脚本和分镜计划生成声音设计。

脚本 JSON:
{script_json}

分镜 JSON:
{storyboard_json}

输出 JSON:
{{
  "bgm_prompt": "英文背景音乐生成 prompt，包含风格、情绪、节奏、乐器",
  "dialogues": [
    {{
      "shot_id": "shot_001",
      "speaker": "人物名或人物id",
      "text": "对白文本",
      "start_seconds": 0,
      "duration_seconds": 3
    }}
  ],
  "sfx": [
    {{
      "shot_id": "shot_001",
      "cue_type": "ambient / foley / impact",
      "prompt": "英文音效生成 prompt",
      "start_seconds": 0,
      "duration_seconds": 3
    }}
  ]
}}

要求:
- 对白来自脚本，不要凭空增加长对白。
- 每个场景至少给一个环境音或 Foley。
- BGM prompt 要适合文本生成音乐模型。
"""
