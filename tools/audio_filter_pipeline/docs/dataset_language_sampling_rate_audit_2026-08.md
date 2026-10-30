# `SR_DATASETS` 数据集语种与采样率核验记录

> 研究范围：初始审计覆盖 52 个物理数据目录；当前 [`config/generate_configs.py`](../config/generate_configs.py) 的 Step4 `SR_DATASETS` 为 53 个逻辑 ID，并由 `DATASET_ROOTS` 映射到 51 个物理目录。Step6 `spk_filter.yaml` 从 Step4 inventory 中排除 clean/filter 已有结果的 6 个 `_spk` 逻辑项，将 `genshinvoice_48k_spk` 命名为 `genshinvoice_spk`，形成 36 个 active ID、覆盖 35 个物理目录。
>
> 结论中的“官方”仅指能够追溯到数据集拥有方、官方项目页、官方仓库或官方数据卡的记录；本地 FLAC/WAV header 只证明当前挂载目录中的派生/下载版本，不反推上游原始格式。

## 1. 结论摘要

- 初始审计 inventory 是 52 个物理目录：`16k` 29 项、`24k` 1 项、`32k` 3 项、`44k1` 10 项、`48k` 9 项；初始 `LANGUAGE_MAP` 覆盖 52/52，未发现重复或漏项。第 3 节保留这份 52 物理目录历史快照，不用当前逻辑 ID 覆盖它。
- 当前 Step4 生成器 inventory 是 53 个逻辑 ID：`16k` 31 项、`24k` 1 项、`32k` 3 项、`44k1` 9 项、`48k` 9 项；`LANGUAGE_MAP` 和 `DATASET_ROOTS` 均覆盖 53/53，并映射到 51 个唯一物理目录。Step6 speaker inventory 排除 clean/filter 已有结果的 6 个 ID，剩余 36 个 active ID，覆盖 35 个唯一物理目录；GenshinVoice 只保留 48 kHz 的 Step4 视图，并在 Step6 不再进入 `spk_filter`。
- 三个本地数据根均存在；初始审计的 52/52 个物理数据集目录均存在且至少有一个可读音频候选文件。固定种子、最多 200 文件/物理目录的 `soundfile.info()` 抽样共检查 10,303 个文件，读取错误 0。
- 物理目录抽样内部采样率已统一到当前路由；仍需关注 `iscslp2026_en`（44.1k/16k）、`iscslp2026_zh`（当前统一为 16k）、`genshinvoice_spk`（当前保留 48k 视图）。另外，`voicebank_spk`、Common Voice 两项提醒了“官方原始采样率”与“本地派生文件采样率”不能共用同一字段，但它们的本地抽样本身是单一且匹配当前分组的。
- 已完成的最终语言路由为：`gigaspeech`、`emov-db_spk`、`ears_spk` 进入 `en`；`jlcorpus_spk` 进入 `en`，`smoothconv` 进入 `zh`，`gtsinger_spk` 进入 `mixed`；真正按多语种整体处理的 `bibletts_spk`、`gtsinger_spk`、`genshinvoice_spk` 保持 `mixed`；`esd_spk` 不再整体进入 `mixed`，而是按 speaker `0001`–`0010` 路由到 `zh`、`0011`–`0020` 路由到 `en`。
- 上述“最终路由”是本地 pipeline 的运行决策，不自动升级为官方数据集事实。`gigaspeech` English、EmoV-DB English、JL-Corpus New Zealand English、GTSinger 九语种已有一级来源；SmoothConv 中文仍是辅助页面加本地证据；`ears_spk` 的 `en`、BibleTTS/GenshinVoice 的完整语言集合仍应保持“运行路由已定、官方证据待补”的等级。
- Step4 对采样率混合目录使用共享 canonical `input_root` 和互斥 `source_selector`：`iscslp2026_en` 为 16k/44.1k，`iscslp2026_zh` 为 16k，GenshinVoice 仅保留 48k；ESD 按 speaker ID 拆分。Step4 共有 6 个 selector。EmoV-DB 已统一为 16 kHz，不再使用采样率 selector。Step6 使用单一 `genshinvoice_spk`，其 `sample_rates_hz: [48000]` 保留 48 kHz 视图，44.1 kHz route 已移除。
- 三份 `audio_filter_*_32k.yaml` 的 `min_source_sample_rate_hz` floor 为 32000；两份 `audio_filter_*_16k.yaml` 与 `spk_filter.yaml` 的 floor 为 16000。所有 shipped 配置均显式给出 floor 与 `output_sample_rate_hz`。
- `bibletts_spk` 在 Step4 只作为 `audio_filter_mixed_32k.yaml` 的注释条目，且因 clean/filter 已有结果而排除出 `spk_filter.yaml`。同样排除 `aishell-3_spk`、`dailytalk_spk`、`genshinvoice_spk`、`gtsinger_spk`、`hifitts2_spk`；`spk_filter.yaml` 是包含 36 个 active ID、覆盖 35 个物理目录的 inventory 模板，但其 speaker 资产为占位，资产 preflight 必须拒绝执行。
- `cnceleb_spk` 的 `zh` 只能解释为 Chinese umbrella；官方 OpenSLR 页并未给出可审计的逐条语言集合。`VoxCeleb1/2` 的官方资源页只提供 identity/gender/nationality，不提供逐条语言标签，因此当前 `en` 也不能视为已核验真实语言集合，且 nationality 不能替代 language。`easyturn_spk` 的本地路径冲突同样只是待核验线索，不写成官方语言事实。
- `unknown`、`unknown_2`、`test__Discarded` 只标记为需要人工核验的路径/身份线索，不自动加入 exclusion，也不在 Step0 自动排除。

## 2. 证据与方法

### 2.1 配置核对

配置证据：[`config/generate_configs.py`](../config/generate_configs.py)。初始审计时，使用 Python `ast.literal_eval` 读取当时的 `SR_DATASETS` 与 `LANGUAGE_MAP` 字面量，避免导入项目模块及运行生成逻辑。当前生成器已包含逻辑拆分以及通过 comprehension/字典展开构造的 `DATASET_ROOTS`；当前 Step4 计数应按第 6 节示例读取 `SR_DATASETS`、`LANGUAGE_MAP`、`DATASET_ROOTS` 与 `SOURCE_SELECTORS`，并以 `DATASET_ROOTS[logical_id]` 解析物理根，不能再假设 `logical_id` 等于物理目录名。Step6 应单独读取 `SPK_FILTER_DATASETS` / `SPK_FILTER_INVENTORY`：它由 Step4 registry 的全部 `_spk` 项派生，并将唯一保留的 `genshinvoice_48k_spk` 映射为 `genshinvoice_spk`，得到 42 个 ID。

### 2.2 本地抽样

本次只读扫描以下目录，未写入数据目录：

- `/data_nvme1n1/clean_speech_download/datasets/`
- `/data_nvme2n1/clean_speech_download/datasets/`
- `/data_nvme3n1/clean_speech_download/datasets/`

下面保留初始审计实际使用的历史抽样脚本；它只适用于当时 52 个物理目录 ID 与目录名相同的 inventory 快照。脚本按物理目录递归收集常见音频扩展名，使用固定 dataset-specific seed 做 reservoir sample，最多 200 个；只读 `soundfile.info()` header，不解码整段音频、不改变文件。临时脚本与 JSON 只写 `/tmp`，不写数据盘或仓库。当前生成器已改为 53 个逻辑 ID，不能把本脚本中的 `root / dataset_id` 查找方式直接用于当前 inventory；当前复核必须通过 `DATASET_ROOTS[logical_id]` 取得物理 root，见第 6 节。

```bash
cat > /tmp/audio_inventory_scan.py <<'PY'
import ast, json, os, random
from pathlib import Path
import soundfile as sf

src = Path('config/generate_configs.py')
tree = ast.parse(src.read_text())
for node in tree.body:
    if (isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'SR_DATASETS'):
        sr_datasets = ast.literal_eval(node.value)
        break
roots = [
    Path('/data_nvme1n1/clean_speech_download/datasets'),
    Path('/data_nvme2n1/clean_speech_download/datasets'),
    Path('/data_nvme3n1/clean_speech_download/datasets'),
]
exts = {'.wav', '.flac', '.mp3', '.ogg', '.opus', '.m4a', '.aac',
        '.wma', '.sph', '.nis', '.aiff', '.aif', '.au', '.caf', '.wv'}
result = {}
for config_sr, datasets in sr_datasets.items():
    for dataset_id in datasets:
        root = next((r / dataset_id for r in roots if (r / dataset_id).is_dir()), None)
        record = {'config_sr': config_sr, 'root': str(root) if root else None,
                  'audio_candidates': 0, 'sampled': 0,
                  'sr_counts': {}, 'sample_errors': {}}
        reservoir = []
        rng = random.Random('audio_filter_pipeline:' + dataset_id)
        if root:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames.sort(); filenames.sort()
                for name in filenames:
                    path = Path(dirpath) / name
                    if path.suffix.lower() not in exts:
                        continue
                    record['audio_candidates'] += 1
                    n = record['audio_candidates']
                    if len(reservoir) < 200:
                        reservoir.append(path)
                    else:
                        j = rng.randrange(n)
                        if j < 200:
                            reservoir[j] = path
            for path in sorted(reservoir):
                try:
                    sr = str(sf.info(str(path)).samplerate)
                    record['sr_counts'][sr] = record['sr_counts'].get(sr, 0) + 1
                    record['sampled'] += 1
                except Exception as exc:
                    key = type(exc).__name__ + ': ' + str(exc)[:160]
                    record['sample_errors'][key] = record['sample_errors'].get(key, 0) + 1
        result[dataset_id] = record
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
python /tmp/audio_inventory_scan.py > /tmp/audio_inventory_scan.json
python - <<'PY'
import json
x = json.load(open('/tmp/audio_inventory_scan.json'))
for d, r in x.items():
    print(d, r['config_sr'], r['audio_candidates'], r['sampled'],
          r['sr_counts'], r['sample_errors'])
PY
```

`audio_candidates` 是递归发现的音频文件数，`sampled` 是成功读取 header 的样本数；不是完整音频内容质量检查，也不代表完整目录的采样率分布。完整目录计数和样本结果如下表已记录。

### 2.3 来源分级

以下 URL 以数据集拥有方、官方项目页、官方仓库或官方数据卡为一级证据；无法从一级来源确认的项目在表中明确写为 `unknown`。本节末尾另列用户提供的数据集目录/聚合页面，只作为辅助证据，不冒充第一方来源：

- [OpenSLR SLR12 — LibriSpeech](https://www.openslr.org/12/)：官方页明确为约 1000 小时、16 kHz、read English。
- [OpenSLR SLR60 — LibriTTS](https://www.openslr.org/60/)：官方页明确为 English、24 kHz。
- [OpenSLR SLR109 — Hi-Fi TTS](https://www.openslr.org/109/)：官方页明确为 English、44.1 kHz。
- [GigaSpeech 官方 GitHub](https://github.com/SpeechColab/GigaSpeech)：官方仓库明确 `Language: English`，并说明音频重采样至 16 kHz。
- [OpenSLR SLR33 — AISHELL-1](https://www.openslr.org/33/)：官方页明确 Mandarin、downsampled to 16 kHz；仅作为 AISHELL-1 证据，不外推其他 AISHELL 版本。
- [OpenSLR SLR93 — AISHELL-3](https://www.openslr.org/93/)：官方页明确 AISHELL-3 是 Mandarin multi-speaker TTS corpus；抓取正文未写采样率。
- [OpenSLR SLR111 — AISHELL-4](https://www.openslr.org/111/)：官方页明确 AISHELL-4 是 Mandarin multi-channel meeting corpus；抓取正文未写采样率。
- [OpenSLR SLR82 — CN-Celeb](https://www.openslr.org/82/)：官方页明确单声道、16 kHz、16-bit，并称其为 Chinese speaker recognition corpus；未提供逐条语言集合。
- [OpenSLR SLR49 — VoxCeleb resources](https://www.openslr.org/49/)：官方辅助资源页只确认 identity/gender/nationality metadata；不能由 nationality 推出语言，页面也未给出采样率。
- [OpenSLR SLR83 — NN-CES](https://www.openslr.org/83/)：官方页确认 UK/Ireland English dialect speech；页面未给出采样率。
- [OpenSLR SLR115 — EmoV-DB](https://www.openslr.org/115/) 与 [官方 GitHub](https://github.com/numediart/EmoV-DB)：均明确 4 位 English speakers；页面只写 16-bit WAV，没有给出采样率。
- [CREMA-D 官方 GitHub](https://github.com/CheyneyComputerScience/CREMA-D)：官方仓库说明 7,442 段、91 actors、英文句子/情绪及 AudioWAV；本地版本 header 仍以本地抽样为准。
- [ESD 官方 GitHub](https://github.com/HLTSingapore/Emotional-Speech-Data)：官方仓库明确 10 位 native Mandarin + 10 位 English speakers；README 未给出采样率。
- [MELD 官方 GitHub](https://github.com/declare-lab/MELD)：官方仓库明确 Friends 片段、多说话人、audio/visual/text；页面未给出可靠的原始音频采样率。
- [MEAD 官方 GitHub](https://github.com/uniBruce/Mead)：官方仓库确认 60 actors、8 emotions、talking-face audio-visual corpus；可访问页面未给出语言集合或采样率。
- [IEMOCAP 官方 USC 页面](https://sail.usc.edu/iemocap/)：官方拥有方页面确认其为 acted、multimodal、multispeaker、约 12 小时数据库；页面未给出语言或采样率。
- [AMI Corpus 官方页面](https://groups.inf.ed.ac.uk/ami/corpus/)：官方拥有方页面确认 100 小时多模态会议、close-talking/far-field microphones；页面未明确写语言集合或采样率。
- [Mozilla Common Voice datasets](https://commonvoice.mozilla.org/en/datasets)：官方入口主要由 JavaScript 渲染；本次抓取未取得具体 locale 版本或音频格式表，故不把 32 kHz 当作官方事实。
- [KeSpeech 官方 GitHub](https://github.com/tzyll/KeSpeech)：官方仓库明确为 Mandarin and its eight subdialects；README 未给出采样率。
- [AliMeeting 官方 baseline GitHub](https://github.com/yufan-aslp/AliMeeting)：官方挑战仓库确认多通道、多方会议转写数据；抓取正文未明确语言或采样率。
- [DailyTalk 官方 GitHub](https://github.com/keonlee9420/DailyTalk)：官方仓库确认 conversational TTS/dialogue corpus；抓取正文未明确语言或采样率。
- [MER2023 官方 baseline GitHub](https://github.com/CASIA-Affective-Computing-Group/MER2023-Baseline)：使用 Chinese HuBERT/BERT/Wav2Vec 模型提供中文强佐证；README 未明确声明完整语言集合或采样率。
- [EARS 官方 GitHub](https://github.com/facebookresearch/ears_dataset)：官方仓库明确 48 kHz；只说 metadata 含 `native language`，未声明数据集音频的完整语言集合。
- [GTSinger 官方 GitHub](https://github.com/AaronZ345/GTSinger)：官方仓库明确九种语言（Chinese、English、Japanese、Korean、Russian、Spanish、French、German、Italian）及 48 kHz/24-bit WAV。
- [JL-Corpus 官方 GitHub README](https://github.com/tli725/JL-Corpus/blob/master/README.md)：官方仓库明确语料采用 New Zealand English；README 未给出采样率。
- [TESS 官方 University of Toronto 页面](https://tspace.library.utoronto.ca/handle/1807/24487)：官方页明确两位母语 English actresses；页面未给出采样率。
- [VCTK v0.92 官方 University of Edinburgh 页面](https://datashare.ed.ac.uk/handle/10283/3443)：官方页明确 110 English speakers，原录音 96 kHz/24-bit，发布文件下采样为 48 kHz/16-bit。

辅助来源（非一级官方证据）：

- [Global Datasets 的 SmoothConv 数据集目录/聚合页面](https://www.global-datasets.com/d/smoothconv)：标题及正文称其为 100.53 小时“多方中文对话”语料，语言明细为中文，并称抽样检查的 2ch 音频为 48 kHz/16-bit PCM。页面列出发布单位，但数据集拥有方身份不明确；因此只用于支持 `zh`/48 kHz 判断，不能写成“官方确认”。本地 200/200 为 48 kHz 是独立且高置信度的本地证据。

## 3. 初始逐数据集核验表

本表保留路由修正前的 52 项 inventory 快照和当时的 header 抽样，用于说明第 4 节最终路由的依据；“初始配置”不是最终运行状态。表中“语言集合/发布采样率”采用：`confirmed` = 本次已访问的一级来源明确写出；`partial` = 一级来源确认了主要语种但未确认完整语言集合或采样率；`unknown` = 本次不能安全确认。非一级来源会在单元格中显式标注“辅助页面支持，第一方待补”。`本地结果`格式为 `候选文件数；抽样数；header 采样率计数`。最终语言桶、逻辑条目和 `source_selector` 以第 4 节为准。

| 数据集 | 初始配置（语言 / SR） | 语言集合（来源分级） | 发布记录采样率 | 本地实际 header 抽样 | 核验结论与建议 | 证据 / 置信度 |
|---|---|---|---|---|---|---|
| `casia_spk` | zh / 16k | unknown（未取得可定位到本地版本的一手说明） | unknown | 268；200；16k×200 | 本地版本可按 16k 路由；语言与官方 SR 暂不可核准 | 本地 header；低（官方）/高（本地） |
| `cremad_spk` | en / 16k | English（官方 README 列出英文句子） | unknown | 803；200；16k×200 | 语言正确；本地 16k 一致，官方 SR unknown | [CREMA-D GitHub](https://github.com/CheyneyComputerScience/CREMA-D)；高 |
| `emov-db_spk` | en / 16k | English（4 位 English speakers） | unknown（官方只写 16-bit WAV） | 5,011；200；16k×200 | 语言与当前路由一致；物理目录已完成 32k→16k 统一 | [OpenSLR SLR115](https://www.openslr.org/115/)；高 |
| `esd_spk` | mixed / 16k | Mandarin Chinese + English | unknown | 32,747；200；16k×200 | 初始 `mixed` 能表达多语种但过粗；最终按 `0001`–`0010` 中文、`0011`–`0020` 英文逻辑拆分；本地 16k 一致 | [ESD GitHub](https://github.com/HLTSingapore/Emotional-Speech-Data)；高 |
| `gigaspeech` | zh / 16k | English | 16 kHz | 4,924,355；200；16k×200 | **LANGUAGE_MAP 明显错误：zh→en**；SR 正确 | [GigaSpeech GitHub](https://github.com/SpeechColab/GigaSpeech)；高 |
| `hq-conversations_spk` | en / 16k | unknown | unknown | 46,131；200；16k×200 | 当前 `en` 没有可追溯官方证据；本地只确认 16k | 本地 header；低（官方）/高（本地） |
| `iemocap_spk` | en / 16k | unknown（已访问 USC 页面未明确语言） | unknown | 4,585；200；16k×200 | 当前 `en` 暂未由已访问一手页确认；本地只确认 16k | [IEMOCAP USC](https://sail.usc.edu/iemocap/) + 本地；低/中 |
| `librispeech_spk` | en / 16k | English read speech | 16 kHz | 230,047；200；16k×200 | 当前项完整正确 | [OpenSLR SLR12](https://www.openslr.org/12/)；高 |
| `m3ed_spk` | zh / 16k | unknown | unknown | 339；200；16k×200 | 本地只确认 16k；需官方论文/数据卡确认语言集合 | 本地 header；低（官方）/高（本地） |
| `magicdata_spk` | zh / 16k | unknown（具体 MagicData 版本未定位） | unknown | 607,911；200；16k×200 | 本地只确认 16k；同名产品较多，必须绑定版本 | 本地 header；低（官方）/高（本地） |
| `mead_spk` | en / 16k | unknown（官方 README 未明确语言） | unknown | 6,647；200；16k×200 | 当前 `en` 暂未由已访问一手页确认；本地版本为 16k | [MEAD GitHub](https://github.com/uniBruce/Mead) + 本地；低/中 |
| `meld_spk` | en / 16k | English（Friends 英文对白；官方 README 有英文 utterance） | unknown | 7,597；200；16k×200 | 语言有强一手佐证；SR 仅证明本地版本 | [MELD GitHub](https://github.com/declare-lab/MELD)；中高 |
| `mer2023` | zh / 16k | Chinese（baseline 使用 Chinese speech/text encoders；未显式声明完整集合） | unknown | 2,834；200；16k×200 | `zh` 有强佐证但仍需 data card 明示；本地 16k 一致 | [MER2023 GitHub](https://github.com/CASIA-Affective-Computing-Group/MER2023-Baseline)；中高 |
| `msp-podcast_spk` | en / 16k | unknown | unknown | 68,853；200；16k×200 | 本地只确认 16k；官方 release 未成功访问 | 本地 header；低（官方）/高（本地） |
| `seniortalk_spk` | zh / 16k | unknown | unknown | 28,677；200；16k×200 | 目录名不能证明语言；本地只确认 16k | 本地 header；低（官方）/高（本地） |
| `tess_spk` | en / 16k | English（两位演员均以 English 为第一语言） | unknown | 1,575；200；16k×200 | 语言正确；本地 16k 一致，官方 SR unknown | [TESS 官方页](https://tspace.library.utoronto.ca/handle/1807/24487)；高 |
| `voicebank_spk` | en / 16k | English（VCTK/Voice Bank v0.92） | 48 kHz（发布版由 96 kHz 下采样） | 10,075；200；16k×200 | **版本语义风险**：本地是再次派生的 16k；official 与 local SR 必须分列 | [VCTK v0.92](https://datashare.ed.ac.uk/handle/10283/3443) + 本地；高 |
| `aishell4` | zh / 16k | Mandarin Chinese meeting speech | unknown（OpenSLR 正文未写） | 211；200；16k×200 | 语言正确；本地 16k 一致，官方 SR unknown | [OpenSLR SLR111](https://www.openslr.org/111/)；高（语言/本地） |
| `aishell5` | zh / 16k | unknown（未取得 AISHELL-5 专属官方说明） | unknown | 4,537；200；16k×200 | 本地只确认 16k；不可借用 AISHELL-1/4 规格 | 本地 header；低（官方）/高（本地） |
| `alimeeting` | zh / 16k | unknown（已访问 baseline README 未明确语言） | unknown | 1,005；200；16k×200 | 本地只确认 16k；补 challenge data description 后再核准 `zh` | [AliMeeting GitHub](https://github.com/yufan-aslp/AliMeeting) + 本地；低/中 |
| `ami` | en / 16k | unknown（官方主页未明确语言） | unknown | 267,290；200；16k×200 | 本地只确认 16k；补 corpus release 文档确认语言 | [AMI official](https://groups.inf.ed.ac.uk/ami/corpus/) + 本地；低/中 |
| `cnceleb_spk` | zh / 16k | Chinese（官方未给逐条语言集合） | 16 kHz/16-bit mono | 669,292；200；16k×200 | 当前 `zh` 可作 Chinese 桶，但不能声称纯 Mandarin；SR 正确 | [OpenSLR SLR82](https://www.openslr.org/82/)；高 |
| `cs_dialogue_spk` | zh / 16k | unknown | unknown | 4,989；200；16k×200 | 本地只确认 16k | 本地 header；低（官方）/高（本地） |
| `dipco` | en / 16k | unknown | unknown | 103；103；16k×103 | 本地只确认 16k | 本地 header；低（官方）/高（本地） |
| `easyturn_spk` | en / 16k | unknown；本地路径含大量 `ZH_*`/`CTS-CN` 冲突线索 | unknown | 1,217,722；200；16k×200 | **LANGUAGE_MAP 高风险不确定项**：人工查官方 manifest 前不应当作已核验英文 | 本地路径/header；低（语言）/高（本地 SR） |
| `kespeech_spk` | zh / 16k | Mandarin and eight subdialects | unknown | 976,054；200；16k×200 | `zh` 仅可作 umbrella；建议记录方言集合，本地 16k 一致 | [KeSpeech GitHub](https://github.com/tzyll/KeSpeech)；高（语言） |
| `voxceleb_vox1_spk` | en / 16k | unknown（官方资源页仅有 nationality） | unknown | 153,516；200；16k×200 | 当前 `en` 未核验；需 utterance-level language metadata | [OpenSLR SLR49](https://www.openslr.org/49/) + 本地；低/中 |
| `voxceleb_vox2_spk` | en / 16k | unknown（同上） | unknown | 1,092,009；200；16k×200 | 同 VoxCeleb1 | [OpenSLR SLR49](https://www.openslr.org/49/) + 本地；低/中 |
| `libritts_r_spk` | en / 24k | English read speech | 24 kHz（LibriTTS；R 版需独立引用） | 290,171；200；24k×200 | 当前本地路由正确；官方 R 版来源待补 | [OpenSLR SLR60](https://www.openslr.org/60/)；中高 |
| `commonvoice_cn_spk` | zh / 32k | Chinese locale/version（具体 locale unknown） | unknown | 33,701；200；32k×200 | 32k 仅为本地 header；绑定 Common Voice release/locale manifest | [Common Voice](https://commonvoice.mozilla.org/en/datasets) + 本地；低/中 |
| `commonvoice_en_spk` | en / 32k | English locale/version | unknown | 1,097,760；200；32k×200 | 32k 仅为本地 header | [Common Voice](https://commonvoice.mozilla.org/en/datasets) + 本地；低/中 |
| `hifitts2_spk` | en / 32k | unknown | unknown | 11,219,789；200；32k×200 | 本地只确认 32k；不要与 Hi-Fi TTS 44.1k 混称 | 本地 header；低（官方）/高（本地） |
| `aishell-3_spk` | zh / 44k1 | Mandarin Chinese multi-speaker TTS | unknown（OpenSLR 正文未写） | 83,526；200；44.1k×200 | 语言正确；44.1k 仅由本地抽样确认 | [OpenSLR SLR93](https://www.openslr.org/93/) + 本地；高（语言/本地） |
| `dailytalk_spk` | en / 44k1 | unknown（官方 README 未明确语言） | unknown | 20,685；200；44.1k×200 | 本地只确认 44.1k | [DailyTalk GitHub](https://github.com/keonlee9420/DailyTalk) + 本地；低/中 |
| `hifi_tts_spk` | en / 44k1 | English | 44.1 kHz | 1,410；200；44.1k×200 | 当前完整正确 | [OpenSLR SLR109](https://www.openslr.org/109/)；高 |
| `jlcorpus_spk` | en / 44k1 | New Zealand English（一级来源） | unknown | 1,991；200；44.1k×200 | 当前 `en` 分类正确；本地 44.1k 一致，官方采样率仍 unknown | [JL-Corpus 官方 GitHub README](https://github.com/tli725/JL-Corpus/blob/master/README.md) + 本地；高（语言/本地） |
| `nnces_read_spk` | en / 44k1 | UK/Ireland English dialect speech | unknown | 4,979；200；44.1k×200 | 语言正确；read/spontaneous 子集边界仍需 manifest 佐证 | [OpenSLR SLR83](https://www.openslr.org/83/)；中高 |
| `nnces_spontaneous_spk` | en / 44k1 | UK/Ireland English dialect speech | unknown | 4,997；200；44.1k×200 | 同上 | [OpenSLR SLR83](https://www.openslr.org/83/)；中高 |
| `savee_spk` | en / 44k1 | unknown | unknown | 409；200；44.1k×200 | 本地只确认 44.1k | 本地 header；低（官方）/高（本地） |
| `vocalset_spk` | en / 44k1 | unknown | unknown | 3,546；200；44.1k×200 | 本地只确认 44.1k；需官方语言/格式说明 | 本地 header；低（官方）/高（本地） |
| `iscslp2026_en` | en / 44k1 | unknown（目录名不能作为官方证据） | unknown | 4,749,715；200；44.1k×190、16k×10 | **SR_DATASETS 不准确/目录混入**；拆分或统一后再路由 | 本地 header；高（本地） |
| `iscslp2026_zh` | zh / 44k1 | unknown（目录名不能作为官方证据） | unknown | 5,003,809；200；16k×200 | **SR_DATASETS 明显错误：抽样全为 16k**；应改 local group 或核对目录身份 | 本地 header；高（本地） |
| `bibletts_spk` | mixed / 48k | unknown | unknown | 126,247；200；48k×200 | `mixed` 不能替代语言集合；本地只确认 48k | 本地 header；低（官方）/高（本地） |
| `ears_spk` | mixed / 48k | unknown（官方仅称 metadata 含 native language） | 48 kHz | 17,059；200；48k×200 | SR 正确；`mixed` 真实语言集合未核验 | [EARS GitHub](https://github.com/facebookresearch/ears_dataset)；高（SR）/低（语言） |
| `emns_spk` | en / 48k | unknown | unknown | 1,157；200；48k×200 | 本地只确认 48k | 本地 header；低（官方）/高（本地） |
| `expresso_spk` | en / 48k | unknown | unknown | 896；200；48k×200 | 本地只确认 48k | 本地 header；低（官方）/高（本地） |
| `gtsinger_spk` | mixed / 48k | Chinese, English, Japanese, Korean, Russian, Spanish, French, German, Italian | 48 kHz/24-bit WAV | 39,857；200；48k×200 | `mixed` 桶方向正确，但应记录官方九语言集合；SR 正确 | [GTSinger GitHub](https://github.com/AaronZ345/GTSinger)；高 |
| `ravdess_spk` | en / 48k | unknown | unknown | 1,056；200；48k×200 | 本地只确认 48k | 本地 header；低（官方）/高（本地） |
| `vctk_spk` | en / 48k | English multi-accent speech | 48 kHz/16-bit release（96 kHz/24-bit recording） | 43,335；200；48k×200 | 当前完整正确 | [VCTK v0.92](https://datashare.ed.ac.uk/handle/10283/3443)；高 |
| `genshinvoice_spk` | mixed / 48k | unknown | unknown | 630,822；200；48k×199、44.1k×1 | **历史抽样为混合 SR**；当前 Step4 和 Step6 只保留 48 kHz selector，44.1 kHz route 已移除；本次未移动、复制、删除或访问原始文件 | 本地 header；高（本地 SR）/低（官方语言） |
| `smoothconv` | zh / 48k | Chinese（辅助页面支持；第一方待补） | 48 kHz/16-bit PCM（辅助页面仅称其抽样 2ch 文件；非一级官方记录） | 2,503；200；48k×200 | 当前 `zh` 分类可保留；48k 获辅助页面和本地 200/200 双重支持，但第一方语言/发布规格仍待补 | [Global Datasets 聚合页](https://www.global-datasets.com/d/smoothconv) + 本地；中（语言辅助）/高（本地 SR） |

## 4. 最终运行路由与逻辑拆分

本节记录已经确定的本地 pipeline 路由。这里的 `en`、`zh`、`mixed` 是处理桶；除非第 2、3 节已有一级来源，否则不得据此反推为数据集拥有方发布的官方语言声明。Step4 配置文件名统一采用 `audio_filter_{language_bucket}_{rate}.yaml`：语言桶由 `en`、`zh`、`mixed` 表达，`16k`/`32k` 表达 source floor 与固定 output rate；数据集的特殊属性不由 `mixed` route 推断，而由各 dataset entry 的 `category` 元数据表达。

### 4.1 最终语言路由

| 原始数据集 | 最终处理桶 | 落地方式 | 证据等级 |
|---|---|---|---|
| `gigaspeech` | `en` | 整体路由 | 一级来源已确认 English；高 |
| `emov-db_spk` | `en` | 整体 English，当前统一为 16k 的单一逻辑条目 | 一级来源已确认 4 位 English speakers；高（语言），本地 header 高（SR） |
| `ears_spk` | `en` | 整体路由 | pipeline 运行决策；官方仓库只确认 48 kHz 和存在 `native language` metadata，未确认完整音频语言集合，故官方语言证据仍低 |
| `jlcorpus_spk` | `en` | 整体路由 | 一级来源确认 New Zealand English；高 |
| `smoothconv` | `zh` | 整体路由 | 辅助页面支持中文，本地 header 支持 48 kHz；语言为中等辅助证据，仍待第一方补证 |
| `gtsinger_spk` | `mixed` | 真正多语种，整体路由 | 一级来源确认九语言；高 |
| `bibletts_spk` | `mixed` | Step4 只在 `audio_filter_mixed_32k.yaml` 保留为注释；Step6 在 `spk_filter.yaml` active 整体路由 | pipeline 运行决策；官方完整语言集合待补，不写成官方事实 |
| `genshinvoice_spk` | `mixed` | Step4 与 Step6 均只保留 48 kHz ID / selector | pipeline 运行决策；本地 header 高（SR），官方完整语言集合待补 |
| `esd_spk` | `zh` + `en` | `0001`–`0010` → `zh`；`0011`–`0020` → `en` | 一级来源确认 10 位 native Mandarin + 10 位 English speakers；高 |

这组最终路由也明确排除以下旧状态：`gigaspeech` 不再进入 `zh`；`emov-db_spk` 和 `ears_spk` 不再进入 `mixed`；`jlcorpus_spk` 不进入 `zh`；`smoothconv` 不进入 `en`；GTSinger 不按中文子集单独路由，而是与 BibleTTS、GenshinVoice 一样以完整多语种数据集进入 `mixed`。ESD 是例外：其 speaker 编号边界可审计，因此按中英文拆分，而不是整体保留在 `mixed`。

### 4.2 Step4 `source_selector` 逻辑拆分

逻辑拆分复用原始 canonical `input_root`，只在 Step0 选择源文件。不同逻辑条目必须通过 `source_selector` 证明互斥；不为拆分移动、复制、重命名或转码原始目录与音频。下表的六个“逻辑 ID”只对应 Step4 `SOURCE_SELECTORS` 的 key，不包含 Step6 inventory。

| 原始目录 | 逻辑 ID | 语言桶 | `source_selector` | 本地依据 |
|---|---|---|---|---|
| `emov-db_spk` | `emov-db_en_16k_spk` | `en` | 无 selector（物理目录已统一为 16k） | 全量转换后为 16k |
| `iscslp2026_en` | `iscslp2026_en_16k` | `en` | `sample_rates_hz: [16000]` | 抽样命中 16k；数据集官方语言/SR manifest 仍待补 |
| `iscslp2026_en` | `iscslp2026_en_44k1` | `en` | `sample_rates_hz: [44100]` | 抽样命中 44.1k；数据集官方语言/SR manifest 仍待补 |
| `iscslp2026_zh` | `iscslp2026_zh_16k` | `zh` | `sample_rates_hz: [16000]` | 抽样 200/200 为 16k；官方语言/SR manifest 仍待补 |
| `genshinvoice_spk` | `genshinvoice_48k_spk` | `mixed` | `sample_rates_hz: [48000]` | 抽样命中 48k；44.1 kHz route 已移除 |
| `esd_spk` | `esd_zh_16k_spk` | `zh` | `include_speaker_ids: ["0001", ..., "0010"]` | 官方 speaker 划分与本地第一层 speaker ID |
| `esd_spk` | `esd_en_16k_spk` | `en` | `include_speaker_ids: ["0011", ..., "0020"]` | 官方 speaker 划分与本地第一层 speaker ID |

表中的 `...` 是文档简写；实际 selector 应显式枚举范围内全部 10 个 speaker ID。采样率 selector 依据本地 header，只说明当前挂载版本如何进入处理分组，不是官方发布采样率声明。若声明的 selector 在 Step0 没有命中，应 fail-closed，而不是静默产出空逻辑数据集。Step6 的单一 `genshinvoice_spk` selector 也为 `sample_rates_hz: [48000]`。

### 4.3 BibleTTS 执行状态

`bibletts_spk` 的最终处理桶是 `mixed`。它在五份 Step4 配置中只作为 `audio_filter_mixed_32k.yaml` 的注释条目，并因 clean/filter 已有结果排除出 `spk_filter.yaml`。Step4 的注释状态与 Step6 排除状态分别由各自 inventory 规则维护。

### 4.4 scanner、mapping 与 namespace 契约

Step0 预期顺序分为两条：仅配置 `source_selector.include_speaker_ids` 且没有采样率门禁时，先解析并筛选 speaker（speaker-only fast path；未命中者在 descriptor 复核后跳过），命中者再读取 descriptor-bound header，最后计算完整 source SHA-256；其余情况先读取 descriptor-bound header，再应用 source floor / rate selector（如有），随后完成 speaker selector 或普通 speaker 标注，最后计算完整 source SHA-256。regular-file scanner 的 current identity 为 `regular3`，`streaming_flac_aggregate` 为 `aggregate4`；二者均把 descriptor 扫描结果写入临时 SQLite，并按路径进行有界磁盘排序，不累积全量 source list。相对 `speaker.mapping_path` 以配置文件 canonical 所在目录为 `mapping_base_dir` 解析，解析结果不得逃逸该目录；但 `mapping_file` 不作为共享 canonical root 上两个逻辑 ID 静态 speaker-set 互斥的证明，静态互斥只接受相同 `path_component` 或 `regex` metadata。

source floor、selector、speaker parser、mapping snapshot、dataset namespace、scanner/provenance/schema 任一变化都改变对应闭包的 current identity，必须从受影响 producer 重建其闭包。旧 artifact、manifest 与 FLAC 不自动清理。已移除的 `genshinvoice_44k1_spk` artifact、enrollment bundle 或 exclusion manifest 不能复用于当前 `genshinvoice_spk` namespace；48 kHz view 的既有资产同样必须通过当前契约校验。`unknown`、`unknown_2`、`test__Discarded` 仅作人工核验标记，不自动加入 exclusion 或由 Step0 排除。

### 4.5 当前 MOS route、producer 版本与断代边界

Step3 registry 支持 SCOREQ、DNSMOS、NISQA、UTMOS、URGENT-PK 五种模型，但配置策略决定真实 route 执行哪些模型。shipped 默认 route 只选择 SCOREQ + DNSMOS，并以严格 2/2 执行 Step4/Step6 MOS 决策；NISQA、UTMOS 与 URGENT-PK 的资产字段保留为可选，未被任何可达 policy 选择时不进入 Step3 preflight。K/N 只定义有效真实分数的决策 quorum，selected 模型的 asset 验证、loader 构造或推理失败仍会使执行 fail-closed，不能按“少一票”继续。

Step3 的 dataset-local scoring binding 保持 threshold-free，但仅适用于 CLI factories 显式传入 `dataset_id` 的生产路径：它只投影该数据集实际可达的评分 route，因此无关数据集策略不会触发本数据集的 Step3 重算；shared runtime 与实际 selected asset 仍进入 identity。Step4/Step6 的 dataset-local decision binding 也仅由该生产 factory 路径启用，不因不相关数据集的阈值或决策策略重算。direct 构造不指定 `dataset_id` 时保持全局 digest、可跨 dataset 复用；direct 构造指定 `dataset_id` 时使用 local binding，并在运行时校验 context。本轮 Step3 row contract 为 4，GPU provenance 是 `asset_load_binding: {"version": 3}`、`step3_mos_pool.v6` 和 `audio_filter_pipeline.step3_mos_worker.v6`；request、worker attestation 与 pool runtime attestation 均精确绑定 `MOS_SCORING_EXECUTION_POLICY`。持久恢复按稳定 request descriptor 规划并在 parent 校验结果后 seal；resume identity 还绑定 execution policy 与规划并行容量，避免在 scorer 执行策略或 GPU 并行容量变化时错误复用 partial plan。worker 对稀疏 ordinal subshard 使用 `ordinal_sidecar_path`、`ordinal_sidecar_sha256`、`ordinal_sidecar_size_bytes` 三字段。Step3 tuple spool 以 SQLite 去重 `segment_id`；每个逻辑 tuple 按 512 MiB physical JSONL part 自动 rollover，CPU/GPU scorer 按 tuple discovery order 与 part index 链式消费；compact index 单文件保持 512 MiB，tuple parts、compact index 与 SQLite logical `page_count * page_size`、固定 cache、已保留的 key 与当前 key reservation 的 aggregate 预算为 4 GiB，不使用文件 `getsize`。GPU weighted LPT 将 source group identity、ordinal 与 task payload digest 写入 owned temporary SQLite，以 rows 与 bytes 双重约束的小批次完成两遍扫描并保持稳定排序，不设置固定内存放行上限；worker input/result/ordinal sidecar 每个 physical file 限制为 512 MiB，input 与 sidecar 按相同边界自动 rollover。由 32 个逻辑 model tuple 和 4 GiB/512 MiB rollover 预算推导出的单 worker subshard canonical 上限为 40；worker 在打开任何 input/sidecar 前拒绝超限请求，并用 owned temporary SQLite `PRIMARY KEY` 以有界批次校验 request-global ordinal 唯一性。input 与 sparse sidecar 各只打开一次并保留同一 `O_NOFOLLOW` FD；全部 subshard 预验证后、每个 output 创建前及评分结束后都复核 FD 与 pathname 身份。

历史 GPU smoke 曾加载并推理 SCOREQ、DNSMOS、NISQA、UTMOS；它既不代表 NISQA/UTMOS 的生产阈值已经校准，也不代表本版本的 binding v3、pool v6、worker v6、持久 request 恢复或 sparse-ordinal 路径已完成真实 GPU 验证。URGENT-PK 只有 checkpoint、pinned commit 和实际 `source_package_sha256` 的严格内容匹配，并由 verified owned snapshot 加载时才允许执行；当前 shipped `urgentpk/model_mod.py` 是 symlink，安全 scanner 仍 fail-closed 拒绝该树，且 default route 不选择 URGENT-PK。

exact-source override 命中时，`route_identity` 只进入 Step3 scoring projection/hash，不写入 artifact row。Step5 的 `speaker_shadow.version` 为 9：每 GPU child 的 inner WeSpeaker model/session 跨最多 32 段的微批常驻复用，source 的打开与绑定复验仍在 worker 的 descriptor-bound source I/O 边界执行。enrollment exclusion manifest **只支持** `audio_filter_pipeline.enrollment_exclusion.v2`，且算法必须为 `mean_cosine_linear_sum.v1`；algorithm 后缀 `v1` 表示当前 O(nd) 算法版本，不是旧 manifest schema。旧 `enrollment_exclusion.v1` 不兼容，必须从 Step3 重建 enrollment bundle 与 exclusion manifest，并更新配置摘要；不能只改 schema 字段伪装迁移。artifact 只有在当前 producer identity、config digest、lineage digest 与 row contract 全部精确匹配时才能 resume，不能仅凭 generation 名称或 Step 编号推断兼容。当前 config、provenance 或 schema 改动必须重建其对应闭包；旧资产不自动删除，且不泛称所有旧 Step0–Step2 artifact 兼容。

### 4.6 SR 字段的语义建议

后续 registry 仍应区分三个概念：

```yaml
official_sample_rate_hz: 44100   # 上游发布规范，可为 unknown
local_sample_rate_distribution: {44100: 190, 16000: 10}
processing_sample_rate_group: 44100  # 本地 routing/筛选分组
```

对 Common Voice、VoiceBank、MEAD 等存在“本地派生版 vs 上游原始版”差异的项目，分别记录 official 与 local，不要把本地 header 当官方事实。`source_selector` 解决的是同一原始目录的本地逻辑分流，不会提高语言或上游发布规格的证据等级。

### 4.7 统一 dataset catalog

[`config/dataset_catalog.py`](../config/dataset_catalog.py) 是本项目数据集基本信息的唯一来源：它覆盖 Step4 `SR_DATASETS` 的全部逻辑 ID（包括 active 与注释项），以及 `spk_filter` 使用的相关 ID。每个条目集中记录可读名称、基本信息、`language_bucket`、`category`、`quality_profile`、特殊属性/备注和采样率或证据状态；这些字段用于解释和审计，不替代生成器现有的运行映射。

[`config/generate_configs.py`](../config/generate_configs.py) 在生成每个 dataset entry 前渲染 catalog 的统一多行 YAML 注释。注释只展示信息，不写入新的运行时字段；七份 shipped YAML 均应由生成器生成，禁止手工维护同一数据集的分散注释。生成器会校验 inventory 覆盖与运行时 `language`、`category`、`quality_profile` 映射一致，保证 catalog 展示不会悄然漂移。

catalog 对特殊数据集保持保守表述：SmoothConv 记录多通道/中文对话属性，SeniorTalk 记录老人声，VocalSet/GTSinger 记录歌唱，GenshinVoice 记录 `game_voice`；NNCES 只记录 read/spontaneous 子集及年龄属性待 manifest 核验，不能在证据不足时写成儿童数据。`mixed` 仍只是语言处理桶，不是 special 类别。

## 5. 置信度与已取消的补证项

2026-10-08 用户明确取消官方数据卡、语言 manifest 等补证任务。以下清单仅保留为
历史证据边界，不再属于待办或验收门禁；现有 `unknown` 不改写成已核验结论。

本次访问 WebSearch 时服务返回 weekly usage limit（HTTP 403），因此未能对全部 52 项逐一取得官方页面全文；已访问页面的内容和 URL 均保留在上面的证据清单中。对于表中标为 `unknown` 的发布采样率/完整语言集合，特意没有用二手聚合页面或文件名臆测填充。SmoothConv 是显式例外：用户提供的目录/聚合页面可辅助支持中文和 48 kHz，但因数据集拥有方身份不明确，报告始终将其与一级来源分开标注。

原报告提出、现已取消的补证清单：

1. `iscslp2026_en/zh` 的官方数据卡、manifest 和文件级语言/SR字段；
2. `easyturn`、`hq-conversations`、`cs-dialogue`、`SeniorTalk` 的官方语言说明，以及 SmoothConv 数据集拥有方的一手语言/发布规格页面（其中文与 48 kHz 当前已有辅助页面及本地 header 支持）；
3. `Common Voice`、`VoiceBank`、`MEAD`、`MELD`、`AMI` 的版本化原始采样率说明；
4. `bibletts`、`ears`、`GenshinVoice` 的官方 language manifest（GTSinger 九语言集合与 JL-Corpus New Zealand English 已确认）；
5. 将报告中的 `unknown` 逐项替换为 canonical URL + 版本/commit + 明确字段，而不是继续扩大 `LANGUAGE_MAP` 的粗粒度桶。

## 6. 可复核验证命令

从项目目录 `/data_nvme0n1/zxc/code/use_train/tools/audio_filter_pipeline` 执行：

```bash
# 当前生成器 inventory：53 个逻辑 ID 通过 DATASET_ROOTS 映射到 51 个物理目录
python - <<'PY'
from pathlib import Path
from runpy import run_path

v = run_path('config/generate_configs.py')
logical_ids = [dataset_id for ids in v['SR_DATASETS'].values() for dataset_id in ids]
language_ids = [dataset_id for ids in v['LANGUAGE_MAP'].values() for dataset_id in ids]
dataset_roots = v['DATASET_ROOTS']
selector_ids = set(v['SOURCE_SELECTORS'])
spk_inventory = v['SPK_FILTER_INVENTORY']

assert len(logical_ids) == len(set(logical_ids)) == 53
assert set(language_ids) == set(logical_ids)
assert set(dataset_roots) == set(logical_ids)
assert selector_ids <= set(logical_ids) and len(selector_ids) == 6
assert len(spk_inventory) == 36
assert 'bibletts_spk' not in spk_inventory
assert 'genshinvoice_44k1_spk' not in logical_ids
assert spk_inventory['genshinvoice_spk']['source_selector'] == {'sample_rates_hz': [48_000]}
physical_roots = {str(Path(dataset_roots[logical_id]).resolve()) for logical_id in logical_ids}
spk_physical_roots = {
    str(Path(spec['input_root']).resolve()) for spec in spk_inventory.values()
}
assert len(physical_roots) == 51
assert len(spk_physical_roots) == 41
print('step4 logical IDs =', len(logical_ids))
print('language coverage =', len(set(language_ids)))
print('physical roots =', len(physical_roots))
print('step4 source selectors =', len(selector_ids))
print('step6 active IDs =', len(spk_inventory), sorted(spk_inventory))
print('step6 physical roots =', len(spk_physical_roots))
for logical_id in sorted(selector_ids):
    print(logical_id, '->', dataset_roots[logical_id])
PY

# 如需重跑 header inventory，必须先按 DATASET_ROOTS 映射物理 root，
# 不得使用 root / logical_id，因为拆分后的 logical_id 不等于物理目录名。

# §2.2 的 /tmp/audio_inventory_scan.py 仅用于初始 52 物理目录历史快照；
# 不要直接拿它扫描当前 53 logical ID inventory。

# 本项目规定的 pipeline unit 测试（从本项目目录执行）
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/data_nvme0n1/zxc/code/use_train:/data_nvme0n1/zxc/code/use_train/tools \
python -m pytest tests -q -W error -p no:cacheprovider

# Markdown whitespace、存在性与 ignore 状态
python - <<'PY'
from pathlib import Path
p = Path('docs/dataset_language_sampling_rate_audit_2026-08.md')
lines = p.read_text().splitlines()
print('exists =', p.is_file(), 'lines =', len(lines))
print('trailing whitespace =', [i for i, line in enumerate(lines, 1) if line.rstrip() != line])
PY
# -v 应显示 pipeline .gitignore 的 !docs/**/*.md 例外；-q 应返回 1（未被忽略）
git check-ignore -v docs/dataset_language_sampling_rate_audit_2026-08.md || true
if git check-ignore --no-index -q docs/dataset_language_sampling_rate_audit_2026-08.md; then
  echo 'unexpected: audit document is ignored' >&2
  exit 1
fi
```

初始审计实际结果保留为：结构校验 `SR count = 52`、`language coverage = 52`、报告数据行 `52`、缺失/额外/重复均为空；52 个物理目录的本地抽样读取 10,303 个 header、错误 0；`PYTHONPATH=.. python -m pytest tests -q` 得到 `1630 passed, 1 warning`。第一次误用 `PYTHONPATH=../..` 时出现 5 个 `ModuleNotFoundError: audio_filter_pipeline` collection error，修正为 `PYTHONPATH=..` 后全套通过；该失败不涉及代码或报告内容缺陷。

当前生成器复核应得到 `step4 logical IDs = 53`、`language coverage = 53`、`physical roots = 51`、`step4 source selectors = 6`、`step6 active IDs = 36`、`step6 physical roots = 35`，且 Step6 inventory 排除 clean/filter 已有结果的 6 个 ID、保留 `genshinvoice_48k_spk` 的 `[48000]` Step4 selector，并不含 `genshinvoice_44k1_spk`。Markdown 应无 trailing whitespace。上级仓库虽有 `*.md` 规则，pipeline `.gitignore` 已用 `!docs/**/*.md` 精确放行本目录的 Markdown 文档；本文件可由普通 `git status` 发现并按维护者选择纳入版本控制，无需强制暂存。
