# -*- coding: utf-8 -*-
"""生成噪声标签 → 四维声学属性静态映射表。

产物: ``core/data/noise_attributes.py``（纯数据，训练构建期零计算）。
标签集合来自当前训练 manifest（data/shared/noise/filter/v2/train_noise_sources.jsonl），
随筛选管线演进，数量不固定。

背景
----
属性表不对应独立的 ``sampling.noise.balance_mode``；当前 schema 只接受
``category`` / ``temperature``。它把 AudioSet 风格语义标签映射到四个声学属性维度，
其中 ``(statistics, temporal)`` 组合为 temperature 模式提供 difficulty 映射，
其余维度用于报告与备查：

- physical    物理来源（12 类，领域知识骨干，AudioSet 标签本身按来源组织）
- statistics  统计特性（平稳/准平稳/非平稳/脉冲）
- temporal    时间特性（连续/间歇/瞬态/周期）
- spectral    频率分布（低频/宽带/高频/纯音/谐波）

生成策略（路 A：标签级领域知识 + 人工审校）
--------------------------------------------
1. physical：整词短语匹配关键词规则（避免子串误判，如 Train↛rain）。
2. statistics / temporal：整词规则命中优先；未命中则按 physical 兜底的
   领域默认值（如 impact→脉冲×瞬态、music→非平稳×连续、mechanical→准平稳×连续）。
3. temporal 的人工审校修正集中在 ``TEMPORAL_OVERRIDES``（25 项 hard case，
   即"间歇/瞬态但被机械兜底误标成连续"的标签）。
4. spectral：整词规则命中优先；未命中按 physical 兜底，仅作报告/备查维度。

用法::

    python -m tools.ops.data.generate_noise_attributes

手动修改产物文件后，如需保持可复现请同步更新本脚本的规则与覆盖表。
产物按「同目录临时文件 + os.replace」原子写入，中断不会留下截断的源码文件。
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

from core.data.noise_roles import MUSIC_CATEGORY_NAMES, _canonical_name

# ---------------------------------------------------------------------------
# 维度取值集合（与 core/data/sampling.py 的校验保持一致）
# ---------------------------------------------------------------------------
PHYSICAL_SOURCES = (
    "mechanical",
    "aerodynamic",
    "electrical",
    "liquid",
    "biological",
    "human_body",
    "vocal",
    "music",
    "impact",
    "signal",
    "synthetic",
    "ambient",
)
STATISTICS = (
    "stationary",
    "quasi_stationary",
    "non_stationary",
    "impulsive",
)
TEMPORAL = (
    "continuous",
    "intermittent",
    "transient",
    "periodic",
)
SPECTRAL = (
    "low_freq",
    "broadband",
    "high_freq",
    "tonal",
    "harmonic",
)

# ---------------------------------------------------------------------------
# 物理来源：整词短语关键词规则（按优先级顺序匹配）｜兜底 ambient
# ---------------------------------------------------------------------------
PHYSICAL_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "vocal",
        (
            "speech", "male speech", "babbling", "whispering", "whisper", "laughter",
            "laughing", "screaming", "scream", "crying", "sobbing", "giggle", "snicker",
            "chuckle", "whimper", "sigh", "gasp", "pant", "whistling", "baby cry",
            "infant cry", "humming", "singing", "shout", "yodeling", "mantra",
        ),
    ),
    (
        "human_body",
        (
            "heart sounds", "heart murmur", "breathing", "cough", "sneeze", "hiccup",
            "snoring", "snore", "burping", "eructation", "fart", "stomach rumble",
            "chewing", "mastication", "throat clearing", "sniff", "fingers snapping",
            "clapping", "applause", "hands", "biting", "snort", "chomp",
        ),
    ),
    (
        "biological",
        (
            "bird", "animal", "frog", "duck", "crow", "owl", "sheep", "fowl",
            "chicken", "rooster", "goose", "turkey", "cat", "dog", "cattle",
            "bovinae", "moo", "horse", "neigh", "pig", "oink", "goat", "bleat",
            "snake", "cricket", "insect", "bee", "wasp", "fly", "mosquito", "whale",
            "lion", "tiger", "roaring cats", "rodents", "rats", "mice", "pigeon",
            "dove", "canidae", "wolves", "wolf", "wild animals", "livestock",
            "domestic animals", "purr", "meow", "bark", "quack", "gobble", "cluck",
            "chirp", "tweet", "caw", "roar", "howl", "croak", "bird vocalization",
            "bird call", "bird song", "whinny", "clip-clop",
        ),
    ),
    (
        "impact",
        (
            "explosion", "gunshot", "gunfire", "machine gun", "artillery", "fusillade",
            "firecracker", "fireworks", "cap gun", "knock", "slam", "bang", "boom",
            "crack", "thump", "thud", "whack", "smash", "crash", "shatter",
            "breaking", "slap", "clatter", "clang", "thunk", "bounce",
            "basketball bounce", "tap", "thunder", "thunderstorm", "eruption",
            "glass", "crushing", "chink", "smack", "whip", "burst", "pop", "squeal",
            "plop", "boing", "arrow",
        ),
    ),
    (
        "signal",
        (
            "siren", "alarm", "buzzer", "beep", "bleep", "ringtone", "bell",
            "church bell", "jingle bell", "jingle", "tinkle", "ding", "ding-dong",
            "ping", "chime", "gong", "dial tone", "busy signal", "telephone",
            "telephone bell", "telephone dialing", "horn", "vehicle horn",
            "car horn", "honking", "train horn", "air horn", "steam whistle",
            "whistle", "foghorn", "sonar", "chirp tone", "pulse", "emergency vehicle",
            "police car", "ambulance", "fire engine", "civil defense siren",
            "reversing beeps", "ice cream truck", "doorbell", "alarm clock",
            "fire alarm", "smoke detector", "change ringing", "tuning fork",
        ),
    ),
    (
        "electrical",
        (
            "sine wave", "mains hum", "hum", "buzz", "static", "distortion",
            "squeal", "crackle", "electronic tuner", "effects unit",
            "electric toothbrush", "electric razor", "electric shaver",
            "electric piano",
        ),
    ),
    (
        "aerodynamic",
        (
            "wind", "steam", "jet engine", "helicopter", "propeller", "whoosh",
            "swoosh", "swish", "air brake", "hiss", "blowing", "air conditioning",
            "airplane", "aircraft", "fixed-wing",
        ),
    ),
    (
        "liquid",
        (
            "rain", "raindrop", "rain on surface", "water", "waterfall", "stream",
            "ocean", "waves", "surf", "water tap", "faucet", "sink", "bathtub",
            "toilet flush", "pour", "drip", "gurgling", "trickle", "dribble",
            "splatter", "splash", "slosh", "squish", "boiling", "frying", "gush",
            "fill", "liquid", "spray", "bubbling", "stir",
        ),
    ),
    (
        "mechanical",
        (
            "vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "boat",
            "motorboat", "speedboat", "sailboat", "rowboat", "canoe", "kayak",
            "ship", "train", "rail transport", "railroad car", "subway", "metro",
            "engine", "idling", "heavy engine", "medium engine", "light engine",
            "engine knocking", "accelerating", "revving", "vroom", "lawn mower",
            "chainsaw", "jackhammer", "power tool", "drill", "hammer", "sawing",
            "sanding", "filing", "tools", "mechanism", "gears", "ratchet", "pawl",
            "vibration", "rumble", "mechanical fan", "vacuum cleaner", "blender",
            "hair dryer", "microwave oven", "sewing machine", "typewriter",
            "printer", "computer keyboard", "typing", "writing", "clock", "tick",
            "tick-tock", "cash register", "drawer", "zipper", "scissors", "camera",
            "slr camera", "keys jangling", "door", "sliding door", "cupboard",
            "chopping", "chop", "cutlery", "silverware", "dishes", "pots and pans",
            "footsteps", "walk", "run", "skateboard",
        ),
    ),
    (
        "synthetic",
        ("white noise", "pink noise", "sound effect", "electronic music", "sampler",
         "synthesizer", "theremin", "drum machine"),
    ),
]
PHYSICAL_FALLBACK = "ambient"

# 人工审校修正（物理来源）：整词规则对复合标签仍有子串语义误判，显式覆盖。
# 例如 "Boat, Water vehicle" 因含 water 被误判为 liquid；"Water tap, faucet"
# 因含 tap 被误判为 impact；"Mouse" 应指老鼠（biological）等。
PHYSICAL_OVERRIDES: dict[str, str] = {
    "Bow-wow": "biological",  # 狗叫
    "Crowing, cock-a-doodle-doo": "biological",  # 公鸡叫
    "Coin (dropping)": "impact",  # 硬币落地
    "Finger snapping": "human_body",  # 响指
    "Honk": "signal",  # 汽车喇叭
    "Mechanisms": "mechanical",  # 机械运转
    "Mouse": "biological",  # 老鼠（非电脑鼠标）
    "Power windows, electric windows": "mechanical",  # 电动车窗
    "Skidding": "mechanical",  # 车辆打滑
    "Traffic noise, roadway noise": "mechanical",  # 道路交通噪声
    "Boat, Water vehicle": "mechanical",  # 水上交通工具（非水流）
    "Water tap, faucet": "liquid",  # 水龙头（非 tap 冲击）
    "Singing bowl": "music",  # 颂钵乐器
    "Chirp tone": "signal",  # 电子啁啾信号音（非鸟叫）
}

# ---------------------------------------------------------------------------
# statistics / temporal：整词规则（高确定性领域锚点）
# ---------------------------------------------------------------------------
STATISTICS_RULES: dict[str, tuple[str, ...]] = {
    "impulsive": (
        "gunshot", "gunfire", "machine gun", "artillery", "fusillade", "explosion",
        "fireworks", "firecracker", "cap gun", "knock", "slam", "bang", "boom",
        "crack", "thump", "thud", "whack", "smash", "crash", "shatter", "breaking",
        "clap", "clapping", "slap", "tap", "drip", "tick", "coin", "camera",
        "slr camera", "keys jangling", "ding", "ding-dong", "ping", "thunder",
        "thunderstorm", "eruption", "sneeze", "cough", "hiccup", "basketball bounce",
        "bounce", "thunk", "clang", "glass", "crackle", "zipper", "click", "chink",
        "tinkle", "splash", "plop", "burst", "pop", "whip", "smack", "gasp",
        "throat clearing", "door", "sliding door", "doorbell", "chopping", "chop",
    ),
    "stationary": (
        "sine wave", "white noise", "pink noise", "mains hum", "hum", "buzz",
        "static", "hiss", "mechanical fan", "air conditioning", "vibration",
        "pulse", "humming",
    ),
    "quasi_stationary": (
        "wind", "rain", "rain on surface", "engine", "vehicle", "traffic",
        "keyboard", "typing", "typewriter", "printer", "crowd", "ocean", "bus",
        "truck", "car", "vacuum cleaner", "microwave oven", "blender", "hair dryer",
        "lawn mower", "chainsaw", "jackhammer", "power tool", "drill", "sawing",
        "siren", "alarm", "alarm clock", "buzzer", "beep", "ringtone",
        "telephone bell", "snoring", "purr", "gurgling", "boiling", "stream",
        "waterfall", "waves", "surf", "applause", "tick-tock", "clock",
        "sewing machine", "aircraft", "helicopter", "jet engine", "propeller",
        "engine starting", "steam", "steam whistle", "foghorn",
    ),
}
TEMPORAL_RULES: dict[str, tuple[str, ...]] = {
    "continuous": (
        "white noise", "pink noise", "sine wave", "hum", "buzz", "static", "wind",
        "engine", "vehicle", "traffic", "rain", "rain on surface", "stream",
        "waterfall", "ocean", "waves", "surf", "crowd", "music", "speech",
        "air conditioning", "vibration", "rumble", "mains hum", "hiss", "silence",
        "applause", "bus", "truck", "car", "boat", "ship", "train", "subway",
        "metro", "humming", "purr", "boiling", "gurgling", "helicopter",
        "jet engine", "propeller", "aircraft", "steam", "engine starting", "drill",
        "lawn mower", "vacuum cleaner", "chainsaw", "jackhammer", "mechanical fan",
    ),
    "transient": (
        "gunshot", "explosion", "fireworks", "firecracker", "knock", "slam", "bang",
        "boom", "crack", "clap", "clapping", "slap", "thump", "thud", "smash",
        "crash", "shatter", "breaking", "sneeze", "thunder", "thunderstorm",
        "eruption", "clang", "clatter", "splash", "camera", "slr camera", "coin",
        "basketball bounce", "chink", "tinkle", "cap gun", "whack", "thunk",
        "burst", "pop", "gasp", "cough", "tap", "click", "drip", "door",
        "sliding door", "doorbell", "chop", "chopping", "whip", "glass", "hiccup",
        "bounce", "zipper", "scissors",
    ),
    "periodic": (
        "clock", "tick", "tick-tock", "heartbeat", "purr", "snoring", "hiccup",
        "gurgling", "siren", "alarm clock", "buzzer", "ding-dong", "bell",
        "church bell", "gong", "pulse", "beep", "ringtone", "telephone bell",
        "snore", "cricket", "alarm", "sonar", "chirp tone", "heart sounds",
    ),
}

# ---------------------------------------------------------------------------
# spectral：整词规则（仅报告/备查维度）｜兜底按 physical
# ---------------------------------------------------------------------------
SPECTRAL_RULES: dict[str, tuple[str, ...]] = {
    "tonal": (
        "sine wave", "beep", "bleep", "buzzer", "siren", "whistle", "dial tone",
        "busy signal", "horn", "alarm", "ringtone", "telephone bell", "ding", "ping",
        "sonar", "tuning fork", "theremin", "chirp tone", "pulse", "reversing beeps",
        "foghorn", "train whistle", "steam whistle", "air horn", "doorbell", "chime",
        "gong", "electronic tuner", "squeal",
    ),
    "low_freq": (
        "engine", "vehicle", "car", "truck", "bus", "motorcycle", "thunder",
        "thunderstorm", "explosion", "rumble", "wind", "fan", "mains hum",
        "helicopter", "jet engine", "heavy engine", "medium engine", "light engine",
        "idling", "lawn mower", "purr", "roar", "lion", "tiger", "propeller",
        "air conditioning", "mechanical fan", "engine knocking", "engine starting",
    ),
    "high_freq": (
        "tick", "clock", "keyboard", "cricket", "insect", "bee", "wasp", "fly",
        "mosquito", "bird", "chirp", "tweet", "whistle", "hiss", "sizzle", "frying",
        "crackle", "squeak", "mouse", "glass", "chink", "tinkle", "jingle", "typing",
        "zipper", "scissors", "tap", "click", "camera", "slr camera", "keys jangling",
        "jingle bell", "tick-tock",
    ),
    "harmonic": (
        "music", "piano", "guitar", "violin", "cello", "singing", "choir", "opera",
        "bell", "church bell", "gong", "drum", "theremin", "synthesizer", "humming",
        "bagpipes", "flute", "trumpet", "saxophone", "clarinet", "harp", "accordion",
        "harmonica", "organ", "harpsichord", "didgeridoo", "shofar", "french horn",
        "trombone", "mandolin", "sitar", "zither", "ukulele", "electric piano",
        "pizzicato", "bowed string instrument", "plucked string instrument",
        "wind instrument", "brass instrument", "mallet percussion", "timpani",
        "glockenspiel", "vibraphone", "marimba", "xylophone", "steelpan", "cymbal",
        "cowbell", "wood block", "rimshot", "snare drum", "bass drum", "drum kit",
        "drum machine", "drum roll", "hi-hat", "tabla", "tambourine", "tubular bells",
    ),
}
# spectral 的 physical 兜底（音乐→谐波、冲击→宽带、信号→纯音、电气→纯音/宽带…）
SPECTRAL_PHYSICAL_DEFAULT = {
    "music": "harmonic",
    "impact": "broadband",
    "liquid": "broadband",
    "biological": "high_freq",
    "vocal": "broadband",
    "human_body": "broadband",
    "mechanical": "broadband",
    "signal": "tonal",
    "electrical": "tonal",
    "aerodynamic": "broadband",
    "synthetic": "broadband",
    "ambient": "broadband",
}

# ---------------------------------------------------------------------------
# statistics/temporal 的 physical 兜底（领域默认；见模块顶部说明）
# ---------------------------------------------------------------------------
ST_PHYSICAL_DEFAULT: dict[str, tuple[str, str]] = {
    "impact": ("impulsive", "transient"),
    "music": ("non_stationary", "continuous"),
    "biological": ("non_stationary", "intermittent"),
    "vocal": ("non_stationary", "continuous"),
    "human_body": ("impulsive", "intermittent"),
    "mechanical": ("quasi_stationary", "continuous"),
    "signal": ("non_stationary", "intermittent"),
    "liquid": ("quasi_stationary", "continuous"),
    "aerodynamic": ("quasi_stationary", "continuous"),
    "electrical": ("stationary", "continuous"),
    "synthetic": ("stationary", "continuous"),
    "ambient": ("quasi_stationary", "continuous"),
}

# ---------------------------------------------------------------------------
# 人工审校的间歇/瞬态 hard case 覆盖。
# ---------------------------------------------------------------------------
TEMPORAL_OVERRIDES: dict[str, str] = {
    "Applause": "intermittent",  # 掌声是间歇性事件
    "Cheering": "intermittent",
    "Clicking": "transient",
    "Computer keyboard": "intermittent",  # 击键 = 间歇突发
    "Crunch": "intermittent",
    "Cupboard open or close": "transient",
    "Drawer open or close": "transient",
    "Engine knocking": "intermittent",
    "Flap": "intermittent",
    "Hammer": "intermittent",  # 锤击
    "Jackhammer": "intermittent",  # 冲击钻为周期冲击，归间歇
    "Patter": "intermittent",
    "Raindrop": "transient",  # 单次滴答
    "Rattle": "intermittent",
    "Run": "intermittent",  # 脚步声
    "Scratch": "intermittent",
    "Scratching (performance technique)": "intermittent",
    "Squeak": "intermittent",
    "Trickle, dribble": "intermittent",
    "Typewriter": "intermittent",
    "Typing": "intermittent",
    "Walk, footsteps": "intermittent",
    "Wind chime": "intermittent",
    "Writing": "intermittent",
    "Keyboard (musical)": "continuous",  # 乐器弹奏，保持连续
}


# ---------------------------------------------------------------------------
# 整词短语匹配：canonical 后按非字母数字分词，关键词短语须为词集子集
# ---------------------------------------------------------------------------
def _tokens(label: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _canonical_name(label)))


def _match_rules(rules: dict[str, tuple[str, ...]], label: str) -> str | None:
    tokens = _tokens(label)
    for value, phrases in rules.items():
        for phrase in phrases:
            required = set(re.findall(r"[a-z0-9]+", phrase))
            if required and required <= tokens:
                return value
    return None


def _classify(label: str) -> tuple[str, str, str, str]:
    tokens = _tokens(label)
    # 1) physical（music 由 MUSIC_CATEGORY_NAMES 精确判定；随后应用人工覆盖）
    if _canonical_name(label) in MUSIC_CATEGORY_NAMES:
        physical = "music"
    else:
        physical = PHYSICAL_FALLBACK
        for value, phrases in PHYSICAL_RULES:
            if _match_rules({value: phrases}, label):
                physical = value
                break
        if _canonical_name(label) in MUSIC_CATEGORY_NAMES:
            physical = "music"
    physical = PHYSICAL_OVERRIDES.get(label, physical)
    # 2) statistics / temporal：规则命中优先，否则按（覆盖后的）physical 兜底
    statistics = _match_rules(STATISTICS_RULES, label)
    temporal = _match_rules(TEMPORAL_RULES, label)
    if statistics is None or temporal is None:
        default_stat, default_temp = ST_PHYSICAL_DEFAULT[physical]
        statistics = statistics or default_stat
        temporal = temporal or default_temp
    # 3) 人工审校修正（temporal hard case）
    if label in TEMPORAL_OVERRIDES:
        temporal = TEMPORAL_OVERRIDES[label]
    # 4) spectral：规则命中优先，否则按 physical 兜底
    spectral = _match_rules(SPECTRAL_RULES, label)
    spectral = spectral or SPECTRAL_PHYSICAL_DEFAULT[physical]
    return physical, statistics, temporal, spectral


def _read_manifest_labels(manifest: Path) -> list[str]:
    """逐行读取 manifest 的 category_name，去重后按字典序返回。"""
    labels: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                labels.add(json.loads(line)["category_name"])
    return sorted(labels)


def _main() -> None:
    # 当前训练使用的噪声 manifest（v2，由 build_stage_d_noise_manifests.py 生成）；
    # 标签集合随筛选管线演进，不再断言固定数量。
    manifest = Path("data/shared/noise/filter/v2/train_noise_sources.jsonl")
    labels = _read_manifest_labels(manifest)
    if not labels:
        raise SystemExit(f"manifest 中未解析到任何 category_name：{manifest}")

    # 按 (physical, label) 排序生成，便于阅读
    classified = [(_classify(label), label) for label in labels]
    classified.sort(key=lambda item: (item[0][0], item[1]))

    lines: list[str] = []
    lines.append("# -*- coding: utf-8 -*-")
    lines.append('"""噪声标签 → 四维声学属性静态映射表。')
    lines.append("")
    lines.append(f"当前 manifest 标签集合共 {len(labels)} 个，数量随 manifest 演进。")
    lines.append("当前 sampling.noise.balance_mode 仅支持 ``category`` / ``temperature``；")
    lines.append("属性表仅用于将标签的 (statistics, temporal) 映射到 difficulty，")
    lines.append("并提供其余维度的报告/备查信息。映射由领域知识规则 + 人工审校生成，")
    lines.append("可复现脚本见 ``python -m tools.ops.data.generate_noise_attributes``。")
    lines.append("")
    lines.append("四维属性：physical(物理来源) / statistics(统计特性) /")
    lines.append("temporal(时间特性) / spectral(频率分布)。difficulty 见")
    lines.append("``core/data/sampling_schema.py::DEFAULT_DIFFICULTY``。")
    lines.append('"""')
    lines.append("")
    lines.append("# 维度取值枚举（与 core/data/sampling.py 校验一致）")
    for name, values in (
        ("PHYSICAL_SOURCES", PHYSICAL_SOURCES),
        ("STATISTICS", STATISTICS),
        ("TEMPORAL", TEMPORAL),
        ("SPECTRAL", SPECTRAL),
    ):
        lines.append(f"{name} = {values!r}")
    lines.append("")
    lines.append("# category_name -> (physical, statistics, temporal, spectral)")
    lines.append("NOISE_ATTRIBUTES: dict[str, tuple[str, str, str, str]] = {")

    current_physical = None
    for (physical, statistics, temporal, spectral), label in classified:
        if physical != current_physical:
            if current_physical is not None:
                lines.append("")
            lines.append(f"    # ---- {physical} ----")
            current_physical = physical
        lines.append(
            f"    {label!r}: ({physical!r}, {statistics!r}, {temporal!r}, {spectral!r}),"
        )
    lines.append("}")
    lines.append("")

    # 校验：所有取值必须是合法枚举
    attrs = {label: _classify(label) for label in labels}
    for label, (physical, statistics, temporal, spectral) in attrs.items():
        assert physical in PHYSICAL_SOURCES, (label, physical)
        assert statistics in STATISTICS, (label, statistics)
        assert temporal in TEMPORAL, (label, temporal)
        assert spectral in SPECTRAL, (label, spectral)

    out = Path("core/data/noise_attributes.py")
    # 产物是仓库源码：先写同目录临时文件再 os.replace，中断不会留下截断的 .py
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=str(out.parent), prefix=f".{out.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, out)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    print(f"已生成 {out}（{len(attrs)} 标签）")

    # 汇总概览（只读辅助，不影响产物）
    dim_order = ("physical", "statistics", "temporal", "spectral")
    values_sets = (PHYSICAL_SOURCES, STATISTICS, TEMPORAL, SPECTRAL)
    print("\n属性分布：")
    for dim, values in zip(dim_order, values_sets):
        index = dim_order.index(dim)
        counter = Counter(attr[index] for attr in attrs.values())
        print(f"  {dim}: { {value: counter[value] for value in values} }")


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__ or "")
        raise SystemExit(0)
    _main()
