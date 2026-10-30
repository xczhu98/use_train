"""噪声类别角色定义与推导工具。

本模块集中管理 wind / music 等“角色”类别集合，供生成器与消费端共享，
避免在 `core.data.dynamic_dataset_base` 与 `tasks.tse.data.dynamic_dataset` 中重复
硬编码类别名。角色推导同时支持：

1. 清单/记录内嵌的 ``roles`` 字段（推荐，新分类 JSONL）。
2. 对旧分类 JSONL 按 ``category_name`` 做规范化后回退匹配。
"""

from __future__ import annotations


def _canonical_name(category_name: str) -> str:
    """把类别名规范化为小写、去首尾空格、下划线替换为空格的形式。"""
    return category_name.strip().lower().replace("_", " ")


# 分类 JSONL 内嵌的风噪类别（category_name 精确匹配，按 _canonical_name 规范化）。
WIND_CATEGORY_NAMES = frozenset(
    {
        _canonical_name("Wind"),
        _canonical_name("Wind noise (microphone)"),
        _canonical_name("Wind_noise_microphone"),
    }
)

# 分类 JSONL 内嵌的音乐类别（category_name 精确匹配，按 _canonical_name 规范化）。
# 核心集合：音乐流派 / 歌唱人声 / 乐器演奏（含弦乐、键盘、管乐、打击乐），
# 均与显式 music_scp 数据合并作为音乐池。排除铃声（Bell/Church bell 等）、
# 环境声（Singing bowl、Tuning fork 等）与信号音。
MUSIC_CATEGORY_NAMES = frozenset(
    {
        _canonical_name(name)
        for name in [
            # 音乐流派
            "Music",
            "Ambient music",
            "Background music",
            "Theme music",
            "Drum and bass",
            "Dubstep",
            "Electronic dance music",
            "Electronic music",
            "Electronica",
            "Heavy metal",
            "Lullaby",
            "New-age music",
            "Progressive rock",
            "Psychedelic rock",
            "Techno",
            "Trance music",
            "Flamenco",
            # 歌唱 / 人声
            "Singing",
            "Choir",
            "Opera",
            "Beatboxing",
            "Humming",
            "Yodeling",
            # 通用器乐
            "Musical instrument",
            "Orchestra",
            # 弦乐
            "Acoustic guitar",
            "Bass guitar",
            "Bowed string instrument",
            "Cello",
            "Double bass",
            "Guitar",
            "Harp",
            "Mandolin",
            "Pizzicato",
            "Plucked string instrument",
            "Sitar",
            "Ukulele",
            "Violin, fiddle",
            "Zither",
            # 键盘 / 电子乐器
            "Electric piano",
            "Hammond organ",
            "Harpsichord",
            "Keyboard (musical)",
            "Organ",
            "Piano",
            "Synthesizer",
            "Theremin",
            # 管乐
            "Accordion",
            "Bagpipes",
            "Brass instrument",
            "Clarinet",
            "Didgeridoo",
            "Flute",
            "French horn",
            "Harmonica",
            "Saxophone",
            "Trombone",
            "Trumpet",
            "Wind instrument, woodwind instrument",
            # 打击乐
            "Bass drum",
            "Cowbell",
            "Cymbal",
            "Drum",
            "Drum kit",
            "Drum machine",
            "Drum roll",
            "Glockenspiel",
            "Hi-hat",
            "Mallet percussion",
            "Maraca",
            "Marimba, xylophone",
            "Percussion",
            "Rattle (instrument)",
            "Rimshot",
            "Snare drum",
            "Steelpan",
            "Tabla",
            "Tambourine",
            "Timpani",
            "Tubular bells",
            "Vibraphone",
            "Wood block",
        ]
    }
)


def category_balance_name(metadata: dict) -> str | None:
    """从清单记录取“均衡类别名”：``balance_category`` 优先，``category_name`` 兼容。

    发布端 strict validator 会约束字段一致性；训练 runtime 不执行该 admission
    校验，并允许轻量 JSON/JSONL 只提供 ``category_name``，因此保留兼容回退。
    """
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("balance_category")
    if isinstance(value, str) and value and value == value.strip():
        return value
    value = metadata.get("category_name")
    if isinstance(value, str) and value and value == value.strip():
        return value
    return None


def compute_category_roles(category_name: str) -> list[str]:
    """根据类别名推导角色集合（用于生成器打标）。

    返回按字母序排序的角色列表，保证输出稳定。 Wind / Music 之外的所有
    类别默认角色为 ``["noise"]``。
    """
    canonical = _canonical_name(category_name)
    roles: set[str] = set()
    if canonical in WIND_CATEGORY_NAMES:
        roles.add("wind")
    if canonical in MUSIC_CATEGORY_NAMES:
        roles.add("music")
    if not roles:
        roles.add("noise")
    return sorted(roles)


def has_role(category_name: str | None, role: str) -> bool:
    """检查类别名是否命中某个角色（旧清单回退匹配）。"""
    if category_name is None:
        return False
    canonical = _canonical_name(category_name)
    if role == "wind":
        return canonical in WIND_CATEGORY_NAMES
    if role == "music":
        return canonical in MUSIC_CATEGORY_NAMES
    return role == "noise"
