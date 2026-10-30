#!/usr/bin/env python3
"""
生成 noise_filter_pipeline.html：当前文本链、canonical 链接与配置参数表。

当前准入契约统一为 strict schema v6、policy/evidence 4/5、parser v10、
generator v6.1 与 promotion marker v4：C1 决定接受类别，C2 只持久化固定
top-5，C3 无阈值、无 calibration，仅执行 Speech MID veto；babble MID 与旧
B5 AED 均不参与决策，旧证据契约 fail-closed。

用法（在 use_train 仓库根目录）:
  PYTHONPATH=tools python -m noise_filter_pipeline.scripts.generate_pipeline_html

重要：本脚本会以同目录临时文件 + fsync + 原子替换更新 noise_filter_pipeline.html。
      若只手改 .html，下次运行本脚本后修改会全部丢失；请改下方「页面可编辑常量」或生成逻辑。
"""
from __future__ import annotations

import argparse
import fcntl
import html
import json
import os
import secrets
import stat
from datetime import datetime, timezone
from pathlib import Path

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PIPELINE_ROOT / "config"
OUT_HTML = PIPELINE_ROOT / "noise_filter_pipeline.html"

# ---------------------------------------------------------------------------
# 页面可编辑常量（改这里后运行 ``PYTHONPATH=tools python -m noise_filter_pipeline.scripts.generate_pipeline_html``）
# ---------------------------------------------------------------------------
PAGE_INTRO_SUB = (
    "统一音频源库构建平台流程：<strong>A</strong> 数据接入 → "
    "<strong>B</strong>（B1 分片 / B2 静音过滤 / B3 VAD / B4 策略过滤）→ "
    "<strong>C</strong>（C1 决定接受类别 / C2 fixed top-5 / "
    "C3 speech-only veto）→ "
    "<strong>D</strong>（validated snapshot / D1 Apollo〔可选〕 / D2 质检）→ "
    "<code>d_completion.json</code>；随后显式并列执行 strict JSONL v6 export 或 "
    "promote 到 final 噪声源库。<br>"
    "B3/B4 当前为 fail_count=0→noise、1→suspected、≥2→rejected；"
    "单路命中进入 C1，strict 模型技术失败仍可拒绝。C3 无阈值、无 calibration；"
    "babble MID 与旧 B5 AED 不参与决策。<br>"
    "发布身份：strict v6、policy/evidence 4/5、parser v10、generator v6.1、"
    "promotion marker v4；当前配置跳过 D1。"
)

# 流程图所在小节的标题（h2）
FLOW_SECTION_H2 = "全链路流程图"


def _open_directory_without_symlinks(directory: Path) -> int:
    """逐级打开目录并拒绝符号链接或非目录 ancestor。"""
    absolute = Path(os.path.abspath(os.fspath(directory)))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in absolute.parts[1:]:
            component_stat = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(component_stat.st_mode):
                raise RuntimeError(f"HTML 输出路径包含符号链接 ancestor: {directory}")
            if not stat.S_ISDIR(component_stat.st_mode):
                raise NotADirectoryError(f"HTML 输出路径 ancestor 不是目录: {directory}")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _file_identity(info: os.stat_result | None) -> tuple[int, ...] | None:
    """返回足以检测目标替换或原位改写的稳定身份。"""
    if info is None:
        return None
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _write_text_atomically(path: Path, content: str) -> None:
    """跨进程锁定父目录，并在同一 dirfd 内复验身份后原子发布。"""
    target = Path(os.path.abspath(os.fspath(path)))
    parent_fd = _open_directory_without_symlinks(target.parent)
    temporary_name: str | None = None
    temporary_fd: int | None = None
    locked = False
    try:
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        locked = True
        parent_identity = os.fstat(parent_fd)
        visible_parent = os.stat(target.parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(visible_parent.st_mode)
            or (visible_parent.st_dev, visible_parent.st_ino)
            != (parent_identity.st_dev, parent_identity.st_ino)
        ):
            raise RuntimeError(f"HTML 输出父目录已被替换: {target.parent}")

        try:
            target_stat = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            target_stat = None
        if target_stat is not None and stat.S_ISLNK(target_stat.st_mode):
            raise RuntimeError(f"HTML 输出目标不能是符号链接: {target}")
        if target_stat is not None and not stat.S_ISREG(target_stat.st_mode):
            raise RuntimeError(f"HTML 输出目标必须是普通文件: {target}")
        target_identity = _file_identity(target_stat)

        temporary_name = f".{target.name}.{secrets.token_hex(8)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(temporary_name, flags, 0o644, dir_fd=parent_fd)
        if target_stat is not None:
            os.fchmod(temporary_fd, stat.S_IMODE(target_stat.st_mode))
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as output:
            temporary_fd = None
            output.write(content)
            output.flush()
            os.fsync(output.fileno())

        try:
            current_stat = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            current_stat = None
        if _file_identity(current_stat) != target_identity:
            raise RuntimeError(f"HTML 输出目标在生成期间被替换或改写: {target}")
        visible_parent = os.stat(target.parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(visible_parent.st_mode)
            or (visible_parent.st_dev, visible_parent.st_ino)
            != (parent_identity.st_dev, parent_identity.st_ino)
        ):
            raise RuntimeError(f"HTML 输出父目录已被替换: {target.parent}")

        os.replace(
            temporary_name,
            target.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        os.fsync(parent_fd)
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        if locked:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        os.close(parent_fd)


def _load_primary_config() -> tuple[Path, dict]:
    """加载用于展示参数的主配置。

    优先级：config/pipeline.yaml
    """
    candidates = [CONFIG_DIR / "pipeline.yaml"]
    for preferred in candidates:
        if preferred.is_file():
            with open(preferred, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return preferred, data
    return candidates[0], {}


def _fmt_val(v: object, max_len: int = 220) -> str:
    if v is None:
        return "<em>null</em>"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return html.escape(str(v))
    if isinstance(v, str):
        s = v if len(v) <= max_len else v[: max_len - 3] + "..."
        return f"<code>{html.escape(s)}</code>"
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(isinstance(x, str) for x in v) and len(v) <= 16:
            parts = ", ".join(html.escape(str(x)) for x in v)
            return f"<code>{parts}</code>"
        j = json.dumps(v, ensure_ascii=False)
        if len(j) > 420:
            j = j[:417] + "..."
        return f'<pre class="cell-pre">{html.escape(j)}</pre>'
    if isinstance(v, dict):
        j = json.dumps(v, ensure_ascii=False, indent=2)
        if len(j) > 420:
            j = j[:417] + "..."
        return f'<pre class="cell-pre">{html.escape(j)}</pre>'
    return html.escape(str(v))


def _cell(data: dict, key: str) -> str:
    if key not in data:
        return "—"
    return _fmt_val(data[key])


def _parameter_rows(data: dict, rows: list[tuple[str, str]]) -> str:
    out = []
    for key, desc in rows:
        out.append(
            f'<tr><th scope="row"><code>{html.escape(key)}</code></th>'
            f'<td class="thr-val">{_cell(data, key)}</td>'
            f'<td class="thr-desc">{html.escape(desc)}</td></tr>'
        )
    return "".join(out)


def _stage_table(stage_key: str, title: str, cfg: dict, rows: list[tuple[str, str]]) -> str:
    body = _parameter_rows(cfg, rows)
    if not body:
        body = '<tr><td colspan="3" style="color:#8b939e">本节无可用参数</td></tr>'
    hid = f"param-{stage_key.lower()}"
    return (
        f'<h3 class="tbl-h" id="{html.escape(hid)}">{html.escape(title)}</h3>'
        f'<div class="table-scroll"><table class="thresholds" aria-labelledby="{html.escape(hid)}">'
        '<thead><tr><th scope="col">参数</th><th scope="col">当前值</th><th scope="col">含义</th></tr></thead>'
        f'<tbody>{body}</tbody></table></div>'
    )


def _parameter_section(cfg_path: Path, data: dict) -> str:
    if not data:
        return f"""
    <section id="parameters" aria-labelledby="param-h2">
      <h2 id="param-h2">配置参数</h2>
      <p class="thr-missing">未在 <code>{html.escape(str(CONFIG_DIR))}</code> 下找到可用于展示参数的 YAML（<code>pipeline.yaml</code>）。放入上述文件后重新运行本生成脚本即可。</p>
    </section>"""

    name = cfg_path.name
    stage_b = data.get("stage_b", {})
    stage_c = data.get("stage_c", {})
    stage_d = data.get("stage_d", {})

    sections = []
    sections.append(
        _stage_table(
            "B",
            "Stage B · 源音频纯化",
            stage_b,
            [
                ("unified", "是否启用 unified 单文件 in-memory 流水线"),
            ],
        )
    )
    sections.append(
        _stage_table(
            "B1",
            "Stage B1 · 分片",
            stage_b.get("b1", {}),
            [
                ("mode", "分片模式：adaptive / hive / none"),
                ("chunk_duration", "分片目标时长（秒）"),
                ("min_chunk_duration", "过短分片丢弃阈值（秒）"),
            ],
        )
    )
    sections.append(
        _stage_table(
            "B2",
            "Stage B2 · 静音过滤",
            stage_b.get("b2", {}),
            [
                ("enable", "是否启用静音过滤"),
                ("max_silence_ratio", "静音占比上限，超过则丢弃"),
                ("min_chunk_duration_filter", "静音过滤后片段最短时长（秒）"),
            ],
        )
    )
    sections.append(
        _stage_table(
            "B3",
            "Stage B3 · VAD 语音检测",
            stage_b.get("b3", {}),
            [
                ("enable", "是否启用 VAD"),
                ("use_multi_vad", "是否多路 VAD 投票"),
                ("vad_types", "并行 VAD 列表"),
                ("speech_threshold", "帧级语音阈值"),
                ("vad_speech_ratio_threshold", "段级语音占比阈值"),
                ("reject_on_any_speech", "当前为 false；单路命中归为 suspected 并交给 Stage C，多路命中按 fail_count 处理"),
                ("batch_size", "VAD 批大小"),
            ],
        )
    )
    sections.append(
        _stage_table(
            "B4",
            "Stage B4 · 策略过滤",
            stage_b.get("b4", {}),
            [
                ("keep_noise", "VAD 判为 noise 档时是否保留"),
                ("keep_suspected", "suspected 档是否保留"),
                ("keep_rejected", "rejected 档是否保留"),
                ("vad_speech_ratio_threshold", "策略过滤使用的段级语音占比阈值"),
            ],
        )
    )
    sections.append(
        _stage_table(
            "C",
            "Stage C · 单事件挖掘",
            stage_c,
            [
                ("c1", "Qwen 三分类；Speech 终态，Noise/Babble 决定接受类别；max_new_tokens 进入 processor/checkpoint fingerprint 与模型/LLM cache key"),
                ("c2", "AudioTag 完整 527 仅内部排序，保存固定 top-5；发布需 production + subprocess_daemon 与完整资源"),
                ("c3", "policy v4 speech-only veto；无阈值、无 calibration"),
            ],
        )
    )
    sections.append(
        _stage_table(
            "C1",
            "Stage C1 · Qwen 分类",
            stage_c.get("c1", {}),
            [
                ("model_path", "Qwen 模型路径"),
                ("batch_size", "推理批大小"),
                ("max_new_tokens", "默认 8；进入 processor/checkpoint fingerprint、Qwen model cache key 与 LLM cache key，修改后不得复用旧结果"),
                ("num_workers", "Qwen worker 数量"),
            ],
        )
    )
    sections.append(
        _stage_table(
            "C2",
            "Stage C2 · AudioTag 标签",
            stage_c.get("c2", {}),
            [
                ("enable", "是否启用 AudioTag"),
                ("mode", "运行模式：dry_run / subprocess_daemon"),
                ("checkpoint", "模型 checkpoint 路径；subprocess_daemon 必需，dry_run 不需要"),
                ("label_dict", "AudioSet 标签表路径；dry_run 与 subprocess_daemon 均需"),
                ("sample_rate", "输入音频采样率"),
                ("batch_size", "推理批大小；输出由代码固定为 5 项 audioset_top5"),
            ],
        )
    )
    sections.append(
        _stage_table(
            "C3",
            "Stage C3 · 最终噪声 admission",
            stage_c.get("c3", {}),
            [
                ("enable", "是否启用最终 admission 防线"),
                ("policy_path", "C3 MID 三分类 policy v4 路径；仅 speech MID veto"),
            ],
        )
    )
    sections.append(
        _stage_table(
            "D",
            "Stage D · 质量增强",
            stage_d,
            [
                ("d1", "Apollo 超分配置"),
                ("d2", "响度归一化与质检配置"),
            ],
        )
    )
    sections_html = "\n      ".join(sections)
    return f"""
    <section id="parameters" aria-labelledby="param-h2">
      <h2 id="param-h2">配置参数（当前配置：<code>{html.escape(name)}</code>）</h2>
      {sections_html}
    </section>"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据当前 pipeline.yaml 生成 Noise Filter 流程 HTML。"
    )
    return parser.parse_args()


def main() -> None:
    _parse_args()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cfg_path, cfg_data = _load_primary_config()
    parameter_html = _parameter_section(cfg_path, cfg_data)

    page = f"""<!DOCTYPE html>
<!-- 本页由 scripts/generate_pipeline_html.py 自动生成；请勿直接编辑。改 PAGE_INTRO_SUB 等常量后重新运行脚本。 -->
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="theme-color" content="#12161c" />
  <title>噪声过滤与源库构建流程</title>
  <style>
    :root {{
      --bg: #12161c;
      --panel: #1b122d;
      --text: #e8eaed;
      --muted: #8b939e;
      --accent: #6b8cae;
      --border: #2c3544;
      --radius-md: 6px;
      --focus-ring: #6b8cae;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      font-size: 16px;
    }}
    header {{
      padding: 2rem 1.5rem 1.25rem;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(165deg, #1a222e 0%, var(--bg) 100%);
    }}
    .title-en {{
      font-size: 0.72rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 0.35rem;
      font-weight: 600;
    }}
    h1 {{
      margin: 0 0 0.5rem;
      font-size: 1.45rem;
      font-weight: 600;
      font-family: Georgia, "Times New Roman", "Songti SC", serif;
      letter-spacing: 0.02em;
    }}
    .sub {{ color: var(--muted); font-size: 0.88rem; max-width: 52rem; }}
    main {{ max-width: min(1120px, 100%); margin: 0 auto; padding: 1.5rem 1rem 3rem; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 1.25rem 1.35rem 1.4rem;
      margin-bottom: 1.25rem;
    }}
    h2 {{
      margin: 0 0 0.75rem;
      font-size: 1.05rem;
      color: var(--accent);
      font-weight: 600;
      font-family: Georgia, "Songti SC", serif;
    }}
    .fig-label {{
      font-size: 0.68rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0 0 0.35rem;
    }}
    .diagram-wrap {{
      overflow-x: auto;
      overflow-y: hidden;
      -webkit-overflow-scrolling: touch;
      background: #0e1218;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 1.75rem 1.25rem 2rem;
      margin: 0.35rem 0 0;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }}
    .flow-chain {{
      margin: 0;
      color: #dce4ee;
      font-size: 1rem;
      line-height: 1.8;
    }}
    .flow-note, .flow-links {{ margin: 0.75rem 0 0; color: var(--muted); }}
    .flow-links a {{ color: #9cc4eb; }}
    footer {{
      text-align: center;
      color: var(--muted);
      font-size: 0.78rem;
      padding: 1rem;
      font-style: italic;
    }}
    .tbl-h {{
      margin: 1.25rem 0 0.5rem;
      font-size: 0.95rem;
      color: var(--accent);
      font-weight: 600;
    }}
    .tbl-h:first-of-type {{ margin-top: 0.65rem; }}
    .thr-missing {{
      font-size: 0.88rem;
      color: #e8b3b3;
      line-height: 1.6;
      margin: 0 0 0.25rem;
    }}
    table.thresholds {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
    }}
    table.thresholds th,
    table.thresholds td {{
      border: 1px solid var(--border);
      padding: 0.5rem 0.65rem;
      text-align: left;
      vertical-align: top;
    }}
    table.thresholds thead th {{
      background: #252f3d;
      color: #c8d2dc;
      font-weight: 600;
    }}
    table.thresholds tbody tr:nth-child(even) {{ background: rgba(0,0,0,0.12); }}
    table.thresholds th[scope="row"] {{
      width: 26%;
      font-weight: 500;
      color: #dce4ee;
    }}
    .thr-val {{ font-variant-numeric: tabular-nums; word-break: break-word; }}
    .thr-desc {{ color: #aeb8c4; }}
    table.thresholds code {{ font-size: 0.82rem; }}
    pre.cell-pre {{
      margin: 0;
      font-size: 0.78rem;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, Consolas, monospace;
    }}
  </style>
</head>
<body>
  <a class="skip-link" href="#main-content">跳到正文</a>
  <header>
    <div class="title-en">Noise filter pipeline</div>
    <h1>噪声过滤与源库构建流程</h1>
    <p class="sub">{PAGE_INTRO_SUB}</p>
  </header>
  <main id="main-content" tabindex="-1">
    <section id="flow" aria-labelledby="fig-main-label">
      <h2 id="fig-main-label">{FLOW_SECTION_H2}</h2>
      <div class="diagram-wrap" role="img" aria-label="当前合法链：Stage A，Stage B 的 B1 到 B4；B3/B4 技术失败或 fail_count 大于等于 2 拒绝，fail_count 等于 1 的 suspected 和等于 0 的 noise 进入 Stage C；Stage C 由 C1 决定接受类别、C2 仅保存固定 top-5、C3 无阈值执行 Speech MID veto；Stage D 构造私有 validated snapshot 并执行 D2，d_completion.json 完成后可显式并列执行 strict JSONL v6 export 或 promote 进入 final">
        <p class="flow-chain"><strong>Stage A</strong> → <strong>Stage B（B1→B2→B3→B4：技术失败或 fail_count≥2 拒绝；1→suspected、0→noise 进入 C1）</strong> → <strong>Stage C（C1 决定接受类别 → C2 fixed top-5 → C3 speech-only veto）</strong> → <strong>Stage D 私有 validated snapshot（仅供 D1/D2）</strong> → <strong>d_completion.json</strong> → <strong>显式并列后处理：strict JSONL v6 export 或 promote → final</strong></p>
        <p class="flow-note"><strong>C3 三决策：</strong>top-5 任一 MID 映射为 speech 即拒绝；否则按 C1 Noise/Babble 分支输出 <code>accept_noise</code> 或 <code>accept_babble</code>。babble MID 不参与决策；平衡类别取 top-5 中最高排名的 Noise MID，无则回退 <code>/m/096m7z</code> / <code>Noise</code>。</p>
        <p class="flow-note"><strong>发布边界：</strong>C1/C2 需满足 production + subprocess_daemon、<code>publishable=true</code> 并通过 policy/evidence/fingerprint 复验；C2 dry-run 仅供测试，C2 disabled 不能发布。validated snapshot 不是发布事实源，promote 只接受 <code>d_completion.json</code> 绑定的 final_stage 与 exact manifest 集合。strict schema v6，policy/evidence 4/5，parser v10，generator v6.1，promotion marker v4；旧契约 fail-closed。</p>
        <p class="flow-links">仓内可复现图：<a href="assets/noise_filter_pipeline_flow_v3.png">Stage A-D 与显式后处理详细 PNG</a>（<a href="assets/noise_filter_pipeline_flow_v3.dot">同名 DOT</a>）和 <a href="assets/noise_filter_pipeline_overview.png">执行与发布边界概览 PNG</a>（<a href="assets/noise_filter_pipeline_overview.dot">同名 DOT</a>）。两图均以 <code>Stage D → d_completion.json</code> 为边界，再并列进入显式 strict export 与显式 promote 分支，只有 <code>promote → final</code>；完整准入证据图见 <a href="../../docs/diagrams/noise_filter_admission_pipeline.dot">跨目录 Graphviz .dot</a>。另见 <a href="README.md">pipeline README</a> 与 <a href="../../docs/guides/noise_filter/current/Noise_Filter_Pipeline_Guide.md">完整使用指南</a>。</p>
      </div>
    </section>
{parameter_html}
  </main>
  <footer>页面由 <code>scripts/generate_pipeline_html.py</code> 生成 · {html.escape(generated_at)} · 当前文本链与 Graphviz .dot 事实源</footer>
</body>
</html>
"""
    _write_text_atomically(OUT_HTML, page)
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
