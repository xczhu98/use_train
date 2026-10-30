"""Audio Filter Pipeline 数据集展示元数据 catalog。

本文件只保存可读展示信息与证据状态；language/category/quality_profile
仍由生成器中的运行映射渲染，并在生成前与本 catalog 交叉校验。
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


_REQUIRED_FIELDS = frozenset({
    "display_name",
    "summary",
    "language_bucket",
    "category",
    "quality_profile",
    "traits",
    "sample_rate_or_evidence",
    "evidence_status",
})


def _entry(
    display_name: str,
    summary: str,
    language_bucket: str,
    category: str,
    quality_profile: str,
    traits: str,
    sample_rate_or_evidence: str,
    evidence_status: str,
) -> Mapping[str, str]:
    return MappingProxyType({
        "display_name": display_name,
        "summary": summary,
        "language_bucket": language_bucket,
        "category": category,
        "quality_profile": quality_profile,
        "traits": traits,
        "sample_rate_or_evidence": sample_rate_or_evidence,
        "evidence_status": evidence_status,
    })


# ID 是运行 inventory 的逻辑 ID；同一物理目录拆出的采样率视图分别保留。
_CATALOG = {
    "casia_spk": _entry("CASIA 情感语音", "中文情感语音的说话人整理版本。", "zh", "expressive", "quality_balanced", "语言与具体发布版本待核验。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "cremad_spk": _entry("CREMA-D", "英语演员情感语音，含多种情绪表达。", "en", "expressive", "quality_balanced", "91 位演员、情感标签。", "本地 header 抽样：16 kHz；官方采样率待核验。", "official_language_local_rate"),
    "emov-db_en_16k_spk": _entry("EmoV-DB English 16k", "EmoV-DB 英语情感语音的 16 kHz 逻辑视图。", "en", "expressive", "quality_balanced", "物理目录已统一为 16 kHz。", "本地 header：16 kHz；官方仅确认 16-bit WAV。", "official_language_local_rate"),
    "esd_zh_16k_spk": _entry("ESD Mandarin speakers", "ESD 中 0001–0010 说话人的中文情感语音视图。", "zh", "expressive", "quality_balanced", "按官方说话人编号拆分；本地 16 kHz。", "本地 header 抽样：16 kHz；官方 README 未给采样率。", "official_language_local_rate"),
    "esd_en_16k_spk": _entry("ESD English speakers", "ESD 中 0011–0020 说话人的英语情感语音视图。", "en", "expressive", "quality_balanced", "按官方说话人编号拆分；本地 16 kHz。", "本地 header 抽样：16 kHz；官方 README 未给采样率。", "official_language_local_rate"),
    "gigaspeech": _entry("GigaSpeech", "大规模英语自动语音识别语料。", "en", "read_speech", "quality_premium", "官方说明音频重采样至 16 kHz。", "官方与本地记录：16 kHz。", "official_language_and_rate"),
    "hq-conversations_spk": _entry("HQ Conversations", "高质量对话语音的说话人整理版本。", "en", "read_speech", "quality_high", "对话属性；语言的一手证据待补。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "iemocap_spk": _entry("IEMOCAP", "多说话人、表演式、多模态情感对话语料。", "en", "expressive", "quality_balanced", "acted、多说话人、多模态；语言/发布采样率待核验。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "librispeech_spk": _entry("LibriSpeech", "英语朗读语音语料。", "en", "read_speech", "quality_high", "约 1000 小时；read speech。", "官方与本地记录：16 kHz。", "official_language_and_rate"),
    "m3ed_spk": _entry("M3ED", "中文多模态情感语音的说话人整理版本。", "zh", "expressive", "quality_balanced", "情感/多模态属性；完整语言集合待 data card 核验。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "magicdata_spk": _entry("MagicData", "MagicData 语音的说话人整理版本。", "zh", "read_speech", "quality_high", "具体 MagicData 产品版本待绑定核验。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "mead_spk": _entry("MEAD", "带人脸视频的表演式情感语音语料。", "en", "expressive", "quality_balanced", "60 位演员、8 种情绪、audio-visual；语言待核验。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "meld_spk": _entry("MELD", "Friends 片段中的多说话人英语对话情感语料。", "en", "expressive", "quality_balanced", "多说话人、audio/visual/text、多模态情感。", "本地 header 抽样：16 kHz；官方采样率待核验。", "official_language_local_rate"),
    "mer2023": _entry("MER2023", "中文多模态情感识别语料。", "zh", "expressive", "quality_balanced", "语音与文本/视频情感属性；完整语言集合待核验。", "本地 header 抽样：16 kHz；官方采样率待核验。", "language_supported_local_rate"),
    "msp-podcast_spk": _entry("MSP-Podcast", "播客场景的英语情感语音说话人整理版本。", "en", "expressive", "quality_balanced", "播客/自然表达属性；官方 release 证据待补。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "seniortalk_spk": _entry("SeniorTalk", "面向老年说话人的语音数据集。", "zh", "elderly", "quality_balanced", "老人声（elderly speech）；语言证据待补。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "tess_spk": _entry("TESS", "两位英语母语演员的情感语音。", "en", "expressive", "quality_balanced", "两位演员、多情绪表达。", "本地 header 抽样：16 kHz；官方采样率待核验。", "official_language_local_rate"),
    "voicebank_spk": _entry("VoiceBank / VCTK release", "VCTK/VoiceBank 英语多口音语音的本地派生版本。", "en", "read_speech", "quality_high", "官方发布版 48 kHz，本地当前文件为再次派生的 16 kHz。", "官方发布：48 kHz；本地 header 抽样：16 kHz。", "official_rate_differs_local"),
    "aishell4": _entry("AISHELL-4", "中文多通道会议语音语料。", "zh", "read_speech", "quality_high", "Mandarin、多通道会议属性。", "本地 header 抽样：16 kHz；官方采样率待核验。", "official_language_local_rate"),
    "aishell5": _entry("AISHELL-5", "AISHELL-5 语音的本地整理版本。", "zh", "read_speech", "quality_high", "不可借用其他 AISHELL 版本规格；专属说明待补。", "本地 header 抽样：16 kHz；官方语言与采样率待核验。", "local_rate_only"),
    "alimeeting": _entry("AliMeeting", "多方会议语音转写语料。", "zh", "read_speech", "quality_high", "多通道、多方会议；语言与官方采样率待核验。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "ami": _entry("AMI Meeting Corpus", "多模态会议语音语料。", "en", "read_speech", "quality_high", "close-talking 与 far-field microphone；语言待核验。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "cnceleb_spk": _entry("CN-Celeb", "中文说话人识别语料。", "zh", "read_speech", "quality_high", "Chinese umbrella；官方未提供逐条语言集合。", "官方记录：16 kHz/16-bit mono；本地 header 抽样：16 kHz。", "official_rate_language_umbrella"),
    "cs_dialogue_spk": _entry("CS Dialogue", "中文对话语音的说话人整理版本。", "zh", "read_speech", "quality_high", "对话属性；官方语言/采样率证据待补。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "dipco": _entry("DiPCo", "会议/对话场景语音数据。", "en", "read_speech", "quality_high", "本地版本语言与发布规格待核验。", "本地 header 抽样：16 kHz；官方采样率待核验。", "local_rate_only"),
    "easyturn_spk": _entry("EasyTurn", "对话语音的说话人整理版本。", "en", "read_speech", "quality_high", "本地路径存在 ZH/CTS-CN 线索；语言必须以 manifest 核验。", "本地 header 抽样：16 kHz；官方采样率待核验。", "language_pending_manifest"),
    "kespeech_spk": _entry("KeSpeech", "普通话及八种子方言语音。", "zh", "read_speech", "quality_high", "Mandarin 与八种子方言；Chinese umbrella。", "本地 header 抽样：16 kHz；官方采样率待核验。", "official_language_local_rate"),
    "voxceleb_vox1_spk": _entry("VoxCeleb1", "名人访谈场景的说话人识别语料。", "en", "read_speech", "quality_high", "官方资源页提供 identity/gender/nationality，不等同逐条语言标签。", "本地 header 抽样：16 kHz；官方采样率待核验。", "language_pending_manifest"),
    "voxceleb_vox2_spk": _entry("VoxCeleb2", "名人访谈场景的说话人识别语料扩展集。", "en", "read_speech", "quality_premium", "官方资源页提供 identity/gender/nationality，不等同逐条语言标签。", "本地 header 抽样：16 kHz；官方采样率待核验。", "language_pending_manifest"),
    "libritts_r_spk": _entry("LibriTTS-R", "英语朗读语音的修复/重制版本。", "en", "read_speech", "quality_high", "R 版来源需与 LibriTTS 原版独立绑定。", "官方 LibriTTS 记录：24 kHz；本地 header 抽样：24 kHz。", "official_rate_local_rate"),
    "commonvoice_cn_spk": _entry("Common Voice Chinese", "Common Voice 中文 locale/version 的本地整理版本。", "zh", "read_speech", "quality_high", "具体 locale 与 release manifest 待绑定。", "本地 header 抽样：32 kHz；官方发布采样率待核验。", "local_rate_only"),
    "commonvoice_en_spk": _entry("Common Voice English", "Common Voice 英文 locale/version 的本地整理版本。", "en", "read_speech", "quality_premium", "具体 locale 与 release manifest 待绑定。", "本地 header 抽样：32 kHz；官方发布采样率待核验。", "local_rate_only"),
    "hifitts2_spk": _entry("HiFiTTS-2", "英语 TTS 语音的本地整理版本。", "en", "read_speech", "quality_premium", "不要与 Hi-Fi TTS 44.1 kHz 版本混称。", "本地 header 抽样：32 kHz；官方采样率待核验。", "local_rate_only"),
    "aishell-3_spk": _entry("AISHELL-3", "中文多说话人 TTS 语料。", "zh", "read_speech", "quality_balanced", "Mandarin、多说话人 TTS。", "本地 header 抽样：44.1 kHz；官方采样率待核验。", "official_language_local_rate"),
    "dailytalk_spk": _entry("DailyTalk", "对话式 TTS 语音语料。", "en", "read_speech", "quality_high", "conversation/dialogue 属性；官方采样率待核验。", "本地 header 抽样：44.1 kHz；官方采样率待核验。", "local_rate_only"),
    "hifi_tts_spk": _entry("Hi-Fi TTS", "英语高保真 TTS 语音。", "en", "read_speech", "quality_high", "不要与 HiFiTTS-2 本地 32 kHz 版本混称。", "官方与本地记录：44.1 kHz。", "official_language_and_rate"),
    "jlcorpus_spk": _entry("JL-Corpus", "New Zealand English 语音语料。", "en", "expressive", "quality_balanced", "New Zealand English；当前运行分类沿用 expressive。", "本地 header 抽样：44.1 kHz；官方采样率待核验。", "official_language_local_rate"),
    "nnces_read_spk": _entry("NN-CES read", "NN-CES 朗读语音子集。", "en", "read_speech", "quality_high", "UK/Ireland English dialect；年龄属性待 manifest 核验，不标注为儿童。", "本地 header 抽样：44.1 kHz；官方采样率待核验。", "official_language_local_rate"),
    "nnces_spontaneous_spk": _entry("NN-CES spontaneous", "NN-CES 自发语音子集。", "en", "read_speech", "quality_high", "UK/Ireland English dialect；年龄属性待 manifest 核验，不标注为儿童。", "本地 header 抽样：44.1 kHz；官方采样率待核验。", "official_language_local_rate"),
    "savee_spk": _entry("SAVEE", "英语演员情感语音。", "en", "expressive", "quality_balanced", "表演式情绪表达；具体发布证据待补。", "本地 header 抽样：44.1 kHz；官方采样率待核验。", "local_rate_only"),
    "vocalset_spk": _entry("VocalSet", "歌唱技巧与人声演唱语料。", "en", "singing", "quality_balanced", "歌唱（singing）；当前语言桶为运行路由，语言证据待补。", "本地 header 抽样：44.1 kHz；官方采样率待核验。", "local_rate_only"),
    "bibletts_spk": _entry("BibleTTS", "BibleTTS 语音合成/朗读语料。", "mixed", "read_speech", "quality_high", "可能涵盖多语种；完整 language manifest 待补；mixed 只是路由桶。", "本地 header 抽样：48 kHz；官方采样率与语言集合待核验。", "local_rate_manifest_pending"),
    "ears_spk": _entry("EARS", "EARS 语音数据集的说话人整理版本。", "en", "read_speech", "quality_high", "官方含 native language metadata，但完整音频语言集合待核验。", "官方与本地记录：48 kHz。", "official_rate_language_pending"),
    "emns_spk": _entry("EMNS", "EMNS 情感/表达语音的说话人整理版本。", "en", "expressive", "quality_balanced", "具体语言与发布规格待补。", "本地 header 抽样：48 kHz；官方采样率待核验。", "local_rate_only"),
    "expresso_spk": _entry("Expresso", "Expresso 表达性语音的说话人整理版本。", "en", "expressive", "quality_balanced", "表达性/风格属性；具体语言与发布规格待补。", "本地 header 抽样：48 kHz；官方采样率待核验。", "local_rate_only"),
    "gtsinger_spk": _entry("GTSinger", "多语言歌唱语料。", "mixed", "singing", "quality_balanced", "歌唱（singing）；官方确认中文、英文、日文、韩文、俄文、西文、法文、德文、意大利文九语种。", "官方与本地记录：48 kHz/24-bit WAV。", "official_language_and_rate"),
    "ravdess_spk": _entry("RAVDESS", "演员表演式情感语音/视听语料。", "en", "expressive", "quality_balanced", "情绪与表演属性；具体语言证据待补。", "本地 header 抽样：48 kHz；官方采样率待核验。", "local_rate_only"),
    "vctk_spk": _entry("VCTK", "英语多口音、多说话人语音语料。", "en", "read_speech", "quality_high", "110 位英语说话人、多口音。", "官方发布版与本地记录：48 kHz/16-bit；原录音 96 kHz/24-bit。", "official_language_and_rate"),
    "genshinvoice_48k_spk": _entry("GenshinVoice 48k", "GenshinVoice 游戏角色语音的 48 kHz 视图。", "mixed", "game_voice", "quality_high", "game_voice；仅保留 48 kHz 来源，使用共享物理目录按 header 分流。", "本地 header selector：48 kHz；官方完整语言集合待补。", "local_selector_language_pending"),
    "smoothconv": _entry("SmoothConv", "多方中文对话语音语料。", "zh", "read_speech", "quality_high", "多通道/多方对话属性；辅助页面支持中文，第一方证据待补。", "本地 header 抽样：48 kHz；辅助页面称 2ch/48 kHz，非一级发布记录。", "local_rate_auxiliary_language"),
    "genshinvoice_spk": _entry("GenshinVoice", "GenshinVoice 游戏角色语音的 Step6 48 kHz 视图。", "mixed", "game_voice", "quality_high", "game_voice；仅保留 48 kHz 来源。", "Step6 selector：48 kHz；官方完整语言集合待补。", "local_selector_language_pending"),
}

DATASET_CATALOG = MappingProxyType(_CATALOG)
# 简短别名，便于只读消费者使用统一命名。
CATALOG = DATASET_CATALOG


def get_dataset_metadata(dataset_id: str) -> Mapping[str, str]:
    """返回指定逻辑 ID 的只读展示元数据。"""
    try:
        return DATASET_CATALOG[dataset_id]
    except KeyError as exc:
        raise KeyError(f"dataset catalog has no entry for {dataset_id!r}") from exc


def validate_catalog(
    expected_ids: set[str],
    language_map: Mapping[str, list[str]],
    category_map: Mapping[str, list[str]],
    quality_profile_map: Mapping[str, list[str]],
) -> None:
    """校验 catalog 覆盖 inventory，并与运行时三组映射保持一致。"""
    if set(DATASET_CATALOG) != set(expected_ids):
        raise ValueError(
            "DATASET_CATALOG coverage mismatch: "
            f"{set(DATASET_CATALOG) ^ set(expected_ids)}"
        )
    runtime_fields = {
        "language_bucket": {dataset_id: bucket for bucket, ids in language_map.items() for dataset_id in ids},
        "category": {dataset_id: category for category, ids in category_map.items() for dataset_id in ids},
        "quality_profile": {dataset_id: profile for profile, ids in quality_profile_map.items() for dataset_id in ids},
    }
    for dataset_id, metadata in DATASET_CATALOG.items():
        missing = _REQUIRED_FIELDS - set(metadata)
        if missing:
            raise ValueError(f"catalog entry {dataset_id!r} missing fields: {sorted(missing)}")
        if dataset_id == "genshinvoice_spk":
            identity_id = "genshinvoice_48k_spk"
        else:
            identity_id = dataset_id
        for field, values in runtime_fields.items():
            if identity_id not in values:
                raise ValueError(f"runtime mapping {field!r} has no entry for {identity_id!r}")
            if metadata[field] != values[identity_id]:
                raise ValueError(
                    f"catalog/runtime drift for {dataset_id!r} {field}: "
                    f"{metadata[field]!r} != {values[identity_id]!r}"
                )
