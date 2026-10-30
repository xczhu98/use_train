from __future__ import annotations

import math
from numbers import Integral, Real

"""Dynamic mixing simulation parameters (SimulationConfigs).

六层流水线（顺序执行，与 ``DynamicMixingDataset._run`` 一致）：

  Layer A — 干净语音预处理（仅近端 clean / speech）；仿真参数写在 ``simulation_cfg`` 顶层：
    A-1  Speech splice (speech_splice)
    A-2  Speed / pitch / EQ / silence: speed_perturb, pitch_shift, eq_perturb, extra_space
    A-3  duration alignment（长度对齐阶段；内置 silence_pad 静音布局子步骤）：
       生产 SE/AEC/TSE Dynamic Dataset 在 splice 前以 ``silence_pad_prob`` 预判模式。
       模式 B 只加载单 utterance，并在固定 ``max_duration`` 画布中放置一次；长输入只
       连续 crop 一次。模式 A 保留 splice、crop 与 repeat_to_max tile 行为。
       注意：``max_duration_seconds`` / ``max_duration`` / ``repeat_to_max_duration`` /
       ``repeat_tile_silence_gap`` / ``repeat_tile_silence_gap_range`` 是 ``DynamicMixingDataset``
       构造参数，写在 ``data.train`` / ``data.validation`` split，**不属于**
       ``SimulationConfigs`` 字段。
    A-4  High-pass (use_high_pass): ~70 Hz filtfilt

  Layer B — **场景角色抽样**（A 之后、声学混合之前）：
    canonical scene ID 直接使用配置中的 ``prob_*`` 键：
    SE：prob_target_only / prob_noise_only / prob_target_noise。
    AEC：prob_target_only / prob_noise_only / prob_echo_only /
      prob_noise_echo_only / prob_target_noise / prob_target_echo /
      prob_target_noise_echo。
    各分支决定 C-aec（清单回声+SER）是否执行以及最终 noisy/clean 的组装方式。

  Layer C — **声学混合**（reverb + target volume + 加性噪声 + sidechain 风噪 + AEC 回声）：
    C-0  RIR convolution（同时得到 full 与 early target）
    C-0b Post-RIR target volume（以 early target 标定，full/early/target 共用 gain）
    C-1  Additive noise mix_noise（SNR-controlled）；空间观测可由
         ``prob_point_source_background`` 升级为同 room 点源 RIR 背景
    C-2  Wind noise（仅 side-chain）
    C-3  Colored noise（white / pink / brown / equalized）
    C-aec  AEC manifest ref/echo + SER/ERL coupling（far-end 电平由 ``far_db_range`` 与
           ``erl_range`` 耦合）
    C-1b Two-segment additive noise mix（noise_mix_prob）

  Layer D — post-mix amplitude conditioning（固定内部顺序 normalize → guard）：
    D-normalize：可选随机峰值归一化，默认 ``off``；关闭时不消费 RNG。
    D-guard：固定执行的 deterministic peak guard。由全部相关 stems 的共同峰值触发；
    触发后以唯一公共安全增益同比缩放，不抽 RNG、不执行 clipping。音量控制已移至
    Layer C-0b 的 post-RIR early target 标定。

  Layer E — URGENT2026 信道退化（bandwidth / clipping / codec / packet loss；仅作用于 noisy）：
    ``prob_channel_enabled``：**仅**作为 **Layer E 全局** 总开关（false = 整层 E 跳过）。
    各子项仍由 ``num_augmentations`` / ``augmentations`` 等抽样。

  Layer F — observation/device degradation（14 项 F-1~F-14；colored noise 已迁 Layer C-3、不占编号，仅作用于 noisy）：
    ``noisy_post_mix_degradations_enabled``：**仅**作为 **Layer F 全局** 总开关（false = 整层 F 跳过；**默认 false**：当前主线尚未做 Layer F 分支实验；需要时在 YAML 设 ``true``）。
    含 clipping、EQ、lowpass、band-reject、DC offset、DRC、codecs、
    bit-crush、spectral holes、WebRTC NS/AGC、bass boost、spectral leakage、loudness 等。

  Final guard — Layer E/F 与 task-post-EF 之后，只保护最终 observation，不再缩放 target 或其他 stems。

  固定八阶段 post-mix contract：pre-finite → D-normalize → D-guard → E → F →
  task-post-EF → final-guard → post-finite。
  注意：``noisy_post_mix_degradations_enabled`` **不**控制 Layer D/E；``prob_channel_enabled`` **不**控制 Layer D/F。

  ``nontarget`` 只是可由最终 ``noisy - clean`` 派生的 residual，不是 Dataset
  独立提供的 batch 标签。

Each augmentation is controlled by prob_xxx; set to 0 to disable.
"""


from core.data.simulation.rir_config import OnlineRIRFields, ONLINE_RANGE_FIELDS, OnlineRIRConfig
from core.data.speech_scene import TIMELINE_MODES
from core.utils.validation import require_positive_int


class SimulationConfigs(OnlineRIRFields):
    # -----------------------------------------------------------------
    # 需要校验为 [lo, hi] 数值区间的属性白名单。不依赖命名后缀扫描，避免未来新增
    # 非区间语义却碰巧以 _range 结尾的属性被误校验。
    # -----------------------------------------------------------------
    _RANGE_ATTRS: frozenset[str] = ONLINE_RANGE_FIELDS | frozenset({
        "array_rotation_z_deg_range",
        "band_reject_center_freq_range",
        "band_reject_freq_bw_range",
        "bass_boost_highpass_cutoff_range",
        "bit_crush_range",
        "microphone_position_jitter_m_range",
        "clip_pedal_threshold_db_range",
        "clip_pre_drive_db_range",
        "clip_rate_range",
        "clip_sigmoid_slope_range",
        "clip_soft_slope_range",
        "clip_sox_threshold_db_range",
        "clip_threshold_db_range",
        "colored_noise_snr_range",
        "dc_offset_range",
        "drc_attack_ms_range",
        "drc_ratio_range",
        "drc_release_ms_range",
        "drc_threshold_db_range",
        "eq_db_range",
        "erl_range",
        "far_db_range",
        "loudness_factor_range",
        "lowpass_cutoff_range",
        "lowpass_order_range",
        "noise_mix_ratio_range",
        "peak_normalization_scale_range",
        "post_eq_db_range",
        "semitone_range",
        "ser_range",
        "silence_pad_long_range",
        "silence_pad_short_range",
        "silence_pad_total_range",
        "sir_range",
        "source_volume_db_range",
        "spectral_holes_num_range",
        "spectral_holes_width_freq_range",
        "spectral_holes_width_time_range",
        "speech_splice_equal_volume_range",
        "speech_splice_gap_range",
        "speech_splice_seconds_range",
        "speed_rate_range",
        "moving_source_speed_range_mps",
        "overlap_ratio_range",
        "webrtc_agc_target_dbfs_range",
    })

    # =================================================================
    # Layer A — 干净语音预处理（仅作用于 clean / speech）
    # =================================================================

    # -----------------------------------------------------------------
    # A-1  Speech splice (concatenate short utterances to target length)
    # -----------------------------------------------------------------
    speech_splice: bool = False
    speech_splice_seconds: float = 20.0
    speech_splice_seconds_range: list | None = None  # [min_sec, max_sec] for random splice target length;
                                                      # if None, use speech_splice_seconds as fixed target.
    speech_splice_gap_range: list = [0.02, 0.2]  # [min_sec, max_sec] silence between spliced segments
    speech_splice_equal_volume: bool = True
    speech_splice_equal_volume_range: list = [-6, 6]   # dB offset
    speech_splice_max_repeat_times: int = 0          # 0=unlimited; cap repeats for short utterances
    speech_random_start: bool = False

    # -----------------------------------------------------------------
    # A-2  Pre-mix speech augmentations
    # -----------------------------------------------------------------
    prob_speed_perturb: float = 0.0
    speed_rate_range: list = [0.8, 1.2]
    prob_pitch_shift: float = 0.0
    semitone_range: list = [-1.5, 1.5]
    prob_eq_perturb: float = 0.0
    eq_db_range: list = [-5, 5]
    prob_add_extra_space: float = 0.0

    # -----------------------------------------------------------------
    # A-3  Duration alignment + silence padding (subset of Layer A)
    # 生产 Dynamic Dataset 在 splice 前预判：模式 B 只加载单 utterance，并在固定
    # max_duration 画布中放置一次；长输入只连续 crop 一次。模式 A 保持原对齐行为。
    # 注意：max_duration_* / repeat_* 是 data split 的 Dataset lifecycle 参数，不属于 SimulationConfigs；
    #       此处只列出与 silence_pad 相关的 SimulationConfigs 字段。
    # -----------------------------------------------------------------
    silence_pad_prob: float = 0.0
    silence_pad_total_range: list = [1.0, 4.0]  # uniform 模式的静音抽样范围；fixed canvas 下不改变最终总静音量
    silence_pad_position: str = "random"  # "random" | "front" | "back" | "center" | "turn_boundary" | "scene_aware"
    silence_pad_mode: str = "uniform"  # "uniform" | "bimodal"
    silence_pad_short_range: list | None = None  # bimodal 分支抽样范围；fixed canvas 下不控制最终静音总量
    silence_pad_long_range: list | None = None   # bimodal 分支抽样范围；fixed canvas 下不控制最终静音总量
    silence_pad_short_ratio: float = 0.5          # P(short pause) when mode="bimodal"
    silence_pad_beta_a: float = 0.4               # Beta α for scene_aware position
    silence_pad_beta_b: float = 0.4               # Beta β for scene_aware position

    # -----------------------------------------------------------------
    # A-4  High-pass filter (~70 Hz filtfilt)
    # -----------------------------------------------------------------
    use_high_pass: bool = True

    # =================================================================
    # 单人 timeline 统一由上方 splice gap / silence_pad 控制，不另做随机切片。
    # 多源关系：保持第一路时间轴，给其他源规划相对窗口；窗口比例不是 VAD 重叠率。
    prob_timeline: float = 0.0
    overlap_ratio_range: list = [0.1, 0.5]
    # 多源相对窗口模式权重；默认保持 WeSep 的五种模式均匀分布。
    timeline_mode_weights: dict = {mode: 0.2 for mode in TIMELINE_MODES}
    # offline 读取 split-level rir_scp；online 在同 room/mic 下生成全部 source。
    rir_backend: str = "offline"
    snr_reference: str = "all_speech"

    # [共享][空间多通道] Spatial observation contract；spatial_enabled=False 时
    # 保持 C=1 单通道行为。
    # 本组键只能在 shared simulation_cfg 定义一次；任务配置通过 data.mode 复用，
    # 不要在 SE/AEC/TSE/SS/SSL 子段重复声明同名或旧词序键。
    spatial_enabled: bool = False
    # "auto"：奇数阵列使用几何中心；偶数阵列必须显式指定。
    reference_channel: int | str = "auto"
    array_profile: str | None = None
    custom_geometry: dict | None = None
    geometry_provenance: dict | None = None
    validation_recipe: dict | None = None
    spatial_diffuse_coherence_model: str = "habets_spherical"
    prob_array_rotation: float = 0.0
    # 阵列 yaw 旋转采样范围，单位 degree（0–360 为整圈）；runtime 在 RIR 边界转为 rad。
    array_rotation_z_deg_range: list = [0.0, 360.0]
    prob_microphone_position_jitter: float = 0.0
    microphone_position_jitter_m_range: list = [0.0, 0.005]
    spatial_layer_d_policy: str = "shared"
    spatial_layer_e_policy: str = "shared"
    spatial_layer_f_policy: str = "per_channel"
    # [共享][空间多通道][online] 把 C-1 背景从 diffuse_coherence 升级为同 room 点源 RIR。
    # 噪声 actor 与 speech scene 共享 room/array/timebase，位置按 speech 源同一
    # 分布族采样（core/data/simulation/background_resolver.py）；仅支持 online 后端。
    # 0（默认）保持既有 diffuse_coherence/additive_mono 行为，且零随机数消耗。
    prob_point_source_background: float = 0.0
    # [共享][空间多通道] 把 wind 资产作为可分解的加性背景渲染；与
    # ``prob_wind_noise_sidechain`` 的非加性 compressor 路径保持独立。
    # 当前仅支持 spatial_enabled=True，统一复用 diffuse/point-source background 链。
    prob_wind_noise_background: float = 0.0

    # Layer B — 场景角色抽样（A 之后、声学混合之前；概率区间见各数据集 _run）。
    # canonical scene ID：配置字段、运行时 branch 与 trace 使用同一个 prob_* 字符串。
    # =================================================================
    # SE/AEC/TSE/SS 的运行时 branch 均直接使用下面的 prob_* canonical scene ID。
    # AEC 七项之和须 =1；SE 的 prob_target_only + prob_noise_only <= 1，
    # 剩余概率进入 prob_target_noise。
    # 这些字段是四任务共享的 schema；任务 builder 负责提供有效的场景分布。
    # 使用全零中性默认值，避免后续重复声明覆盖不同任务的默认语义。
    prob_target_only: float = 0.0   # target-only：noisy=clean_target，仍先经 Layer C-0 后走 D/E/F
    prob_noise_only: float = 0.0   # P(noise_only)：clean=0，noisy=信噪比之后的 noise 分量（Layer C-1/C-2）
    prob_echo_only: float = 0.0    # AEC only: P(echo-only)
    prob_noise_echo_only: float = 0.0  # AEC only: P(noise+echo)
    # AEC only — target = clean_target (近端监督)；与 prob_noise_only（target=0）不同
    prob_target_noise: float = 0.0       # target+noise：noisy=C-0~C-2 全混合，不叠清单回声
    prob_target_echo: float = 0.0        # target+echo：noisy=clean_target 径，mic≈clean+echo
    prob_target_noise_echo: float = 0.0  # target+noise+echo：noisy=全混合，mic≈近端+回声

    # SS/TSE 继续复用 shared prob_target_only/prob_target_noise/prob_noise_only；
    # 下列字段只属于 TSE，默认 0，需由 TSE 配置显式组成七路分布。
    prob_interference_only: float = 0.0
    prob_interference_noise_only: float = 0.0
    prob_target_interference: float = 0.0
    prob_target_interference_noise: float = 0.0
    num_interferers: dict = {1: 1.0}  # 包含 I 的场景才抽样，key 必须为正整数

    # =================================================================
    # Layer C — 声学混合
    # =================================================================

    # -----------------------------------------------------------------
    # C-0  RIR / Reverb（与 Layer B 分支独立）
    # -----------------------------------------------------------------
    prob_reverberation: float = 0.5
    # 在线 moving RIR 概率（spatial scene 与 mono 路径共用）；room/actor
    # trajectories 在 provider 前固定。仅控制在线生成；离线 moving 资产由
    # manifest/池子组成决定。mono 路径下仅在抽中 prob_reverberation 后生效。
    prob_moving_rir: float = 0.0
    # 物理声源运动速度，独立于 Layer A 的 speech speed perturbation。
    moving_source_speed_range_mps: list = [0.12, 0.4]
    reuse_rir: bool = False
    early_rir_sec: float = 0.05
    # D-01: 训练 target 口径。 "early"(默认)=early-RIR 卷积(PASE/AEC/其他)；
    # "dry"=dry speech(StuPASE 去混响监督)；"rts"=按场景 RT60 缩短 full RIR 后卷积。
    # 仅改变监督 target，不改变声学混合几何或 SNR 参考。
    target_type: str = "early"
    rts_target_t60_sec: float = 0.15
    rts_time_after_max_sec: float = 0.002
    # D-05: RIR 卷积前是否峰值归一化。True(默认)=静态 full/early 共用 full peak，
    # moving 全轨迹及 early 共用 full-sequence peak；False(PASE/StuPASE)=保留静态
    # 跨 RIR 与 moving 跨段原始增益，对齐官方 scaling。
    rir_normalize: bool = True
    # D-06: 离线 moving RIR 的分段时间（秒）。消费侧优先读取 PKL 自带的
    # segment_sec；本字段仅作兜底。它不启用在线 moving gpuRIR。
    segment_sec: float = 0.5

    # -----------------------------------------------------------------
    # C-0b  Post-RIR early-target volume normalization
    # -----------------------------------------------------------------
    # 启用时先以 early target 测量并抽取目标 dBFS，再将同一 gain 同步施加到
    # full、early 与 target，保持三者相对增益；noise / echo 在此步骤之后混入。
    # 字段名不表示在 RIR 前处理 source waveform。
    source_volume_normalization_enabled: bool = False
    source_volume_db_range: list = [-40, -20]
    source_volume_use_rms: bool = True
    # "uniform": every dB in source_volume_db_range equally likely (default,
    # matches historical behavior). "beta": draw from Beta(source_volume_beta_a,
    # source_volume_beta_b) mapped onto the range -- symmetric params (e.g. the
    # default 2.0/2.0) center the distribution at the range's midpoint and
    # taper toward both ends without excluding them. See core/data/simulation_utils.py::rng_uniform_or_beta.
    source_volume_distribution: str = "uniform"
    source_volume_beta_a: float = 2.0
    source_volume_beta_b: float = 2.0

    # -----------------------------------------------------------------
    # C-1  Additive noise / SNR
    # -----------------------------------------------------------------
    sir_range: list | None = None
    # 四任务共享的背景 family 概率；注册音使用自己的 provider。
    music_ratio: float = 0.0


    snr_low: float = -5.0
    snr_high: float = 20.0
    reuse_noise: bool = False

    # -----------------------------------------------------------------
    # C-1b 双段加性噪声混合（noise_mix_prob：是否再混第二段；与 AEC 清单回声 C-aec 不同）
    # -----------------------------------------------------------------
    noise_mix_prob: float = 0.0
    noise_mix_ratio_range: list = [0.1, 1.0]

    # -----------------------------------------------------------------
    # C-2 Wind noise sidechain（非加性；SE/AEC/TSE/SS 四任务共享同一路径）
    # -----------------------------------------------------------------
    # [共享][单通道默认；SSL spatial 实验路径例外] 风噪只通过 side-chain compressor path 施加；普通噪声仍来自 noise_scp、
    # 分类 JSONL 噪声池或 colored noise。
    # 注意：side-chain 压缩是非加性叠加，直接改变 target/observation 波形本身，
    # 不再是一个可分离的噪声分量。基于未失真（undistorted）target 的 loss
    # （SI-SDR、幅度/波形域 loss 等）与 early-target 监督口径会系统性失配；
    # export 下 stems 也因此不可分解。默认 0，由各任务配置显式开启。
    prob_wind_noise_sidechain: float = 0.0
    wind_noise_config: dict = dict(
        threshold=[0.1, 0.3], ratio=[1, 20], attack=[5, 100], release=[5, 100],
        sc_gain=[0.8, 1.2], clipping_threshold=[0.85, 1.0], clipping_chance=0.75,
        wind_noise_snr_range=[-10.0, 15.0],
    )

    # -----------------------------------------------------------------
    # C-aec  AEC — SER（YAML ``simulation_cfg``；SE 可省略，字段保持默认 None）。
    # 另：清单 ref/echo 粗对齐 / GCC / 随机前导延迟（aec_ref_echo_*）可写在同层，由
    # ``expand_simulation_cfg_aec_ref_echo_dataset_keys`` 提到 AEC Dataset 构造参数。
    # -----------------------------------------------------------------
    ser_range: list | None = None

    # =================================================================
    # Layer C-aec — Echo Return Loss (ERL): physical coupling between far-end and echo
    # =================================================================
    # ERL = far_db - echo_db (standard definition: positive = echo quieter than ref).
    # Covers 5-50 dB typical range + negative ERL for hot-echo badcase.
    # SER is the primary control variable (industry standard); ERL provides the
    # physical constraint: far_db = echo_db + ERL.
    #
    # far_db_range: physical bounds for the far-end reference signal (dBFS RMS).
    #   [-40, -12] — aligned with source_volume: both are human speech in the same physical range.
    #   [far_db_lo - echo_db, far_db_hi - echo_db], intersected with erl_range.
    # Distribution: "uniform" or "beta".  Beta(1.5, 1.5) is slightly centre-weighted.
    far_db_range: list = [-40, -12]    # dBFS RMS; physical bounds for reference signal
    erl_range: list = [-20, 40]        # dB; ERL = far-end level - echo level
    erl_distribution: str = "beta"
    erl_beta_a: float = 1.5
    erl_beta_b: float = 1.5

    # =================================================================
    # Layer D — post-mix amplitude conditioning（normalize → guard）
    # =================================================================
    # D-normalize：公共 Layer D 只允许 off/joint。off（默认）不消费 RNG；joint
    # 对相关 bundle 取一个共同 peak 并使用同一 scale，保持线性分解。
    # PASE/StuPASE 的 separate/single 属于模型专属预处理，不再进入公共 schema。
    # D-guard：随后固定执行，仅按全部相关 stems 的共同峰值确定性同比缩放，不抽
    # RNG、不执行 clipping；observation/device clipping 的权威位置是 Layer F。
    peak_normalization_mode: str = "off"
    peak_normalization_scale_range: list = [0.5, 0.95]

    # =================================================================
    # Layer E — URGENT2026 信道退化（全局仅由此开关）
    # =================================================================
    prob_channel_enabled: bool = True  # false = 跳过整层 Layer E；true = 按 num_augmentations 等抽样
    augmentations_name: list = ["bandwidth_limitation", "clipping", "codec", "packet_loss"]
    num_augmentations: dict = {0: 0.25, 1: 0.40, 2: 0.20, 3: 0.15}
    # bandwidth_limitation 可选 target_sr：固定目标采样率（UniPASE/GAP 固定 8000），
    # 未配置时保持随机低采样率行为。
    augmentations: dict = dict(
        bandwidth_limitation=dict(weight=1.0, resample_methods="random"),
        clipping=dict(weight=1.0, clipping_min_quantile=[0.0, 0.1], clipping_max_quantile=[0.9, 1.0]),
        codec=dict(weight=1.0, config=[
            dict(format="mp3", encoder=None, qscale=[1, 10]),
            dict(format="ogg", encoder=["vorbis"], qscale=[1, 10]),
        ]),
        packet_loss=dict(weight=1.0, packet_duration_ms=20, max_continuous_packet_loss=10, packet_loss_rate=[0.05, 0.25]),
    )

    # =================================================================
    # Layer F — observation/device degradation（仅 noisy；全局仅由此开关，与 Layer E 无关）
    # =================================================================
    # Default false：主线未启用 Layer F 实验时保持关闭；启用 observation/device degradation 时在 YAML 设 true。
    noisy_post_mix_degradations_enabled: bool = False

    # -----------------------------------------------------------------
    # F-1  Post-mix: speaker distortion / clipping
    # -----------------------------------------------------------------
    prob_clip: float = 0.0
    # fixed_rail 是 observation/device clipping 的标准变体：先施加 pre-drive，
    # 再固定裁剪到 ±1。其他变体保留既有设备失真模拟能力。
    clip_variants: list = ["fixed_rail"]
    clip_variant_probs: dict = dict(fixed_rail=1.0)
    clip_pre_drive_db_range: list = [0.0, 12.0]
    clip_hard_portion: float = 0.0
    clip_hard_on_rate: bool = False
    clip_rate_range: list = [0.01, 0.2]
    clip_threshold_db_range: list = [-20, -1]
    clip_soft_slope_range: list = [1.0, 5.0]
    clip_sigmoid_slope_range: list = [1.0, 5.0]
    clip_sox_threshold_db_range: list = [-15, -5]
    clip_pedal_threshold_db_range: list = [-15, -5]

    # -----------------------------------------------------------------
    # F-2  Post-mix: EQ (on noisy only)
    # -----------------------------------------------------------------
    prob_post_eq: float = 0.0
    post_eq_db_range: list = [-5, 5]

    # -----------------------------------------------------------------
    # F-3  Post-mix: low-pass filter
    # -----------------------------------------------------------------
    prob_lowpass: float = 0.0
    lowpass_cutoff_range: list = [1000, 8000]
    lowpass_order_range: list = [4, 20]

    # -----------------------------------------------------------------
    # F-4  Post-mix: band reject (notch)
    # -----------------------------------------------------------------
    prob_band_reject: float = 0.0
    band_reject_center_freq_range: list = [100, 8000]
    band_reject_freq_bw_range: list = [20, 500]
    band_reject_use_stft: bool = True
    band_reject_max_n: int = 2

    # -----------------------------------------------------------------
    # C-3  Acoustic mixing: colored noise  (white / pink / brown / equalized)
    #      （自原 Layer F-5 迁入；在 _mix_acoustic 内以 clean_target 功率定 SNR，
    #        从产生起归入 noise stem，仅作用于 need_c12 分支；不占 F 编号）
    # -----------------------------------------------------------------
    prob_colored_noise: float = 0.0
    colored_noise_snr_range: list = [10, 50]
    colored_noise_types: list = ["white", "pink", "brown", "equalized"]

    # -----------------------------------------------------------------
    # F-5  Post-mix: DC offset
    # -----------------------------------------------------------------
    prob_dc_offset: float = 0.0
    dc_offset_range: list = [0.001, 0.2]

    # -----------------------------------------------------------------
    # F-6  Post-mix: DRC  (dynamic range compression)
    # -----------------------------------------------------------------
    prob_drc: float = 0.0
    drc_threshold_db_range: list = [-50, 0]
    drc_ratio_range: list = [1, 20]
    drc_attack_ms_range: list = [0.5, 5.0]
    drc_release_ms_range: list = [50, 1000]

    # -----------------------------------------------------------------
    # F-7  Post-mix: codecs  (MP3 / GSM / OPUS / AAC)
    # -----------------------------------------------------------------
    prob_codecs: float = 0.0
    codecs_types: list = ["mp3", "gsm", "opus"]

    # -----------------------------------------------------------------
    # F-8  Post-mix: bit crush
    # -----------------------------------------------------------------
    prob_bit_crush: float = 0.0
    bit_crush_range: list = [4, 16]

    # -----------------------------------------------------------------
    # F-9  Post-mix: spectral time-frequency holes
    # -----------------------------------------------------------------
    prob_spectral_holes: float = 0.0
    spectral_holes_frame_length: int = 1024
    spectral_holes_hop_length: int = 256
    spectral_holes_num_range: list = [1, 20]
    spectral_holes_width_freq_range: list = [1, 5]
    spectral_holes_width_time_range: list = [1, 5]
    spectral_holes_cutoff_freq: float = 10000

    # -----------------------------------------------------------------
    # F-10/F-11  WebRTC NS/AGC simulation
    # -----------------------------------------------------------------
    prob_webrtc_ns: float = 0.0
    webrtc_ns_levels: list = [0, 1, 2, 3]  # Aggressiveness levels
    prob_webrtc_agc: float = 0.0
    webrtc_agc_target_dbfs_range: list = [-31, -3]

    # -----------------------------------------------------------------
    # F-12  Bass boost
    # -----------------------------------------------------------------
    prob_bass_boost: float = 0.0
    bass_boost_highpass_cutoff_range: list = [500, 2000]
    bass_boost_attenuation_min_db: float = -25

    # -----------------------------------------------------------------
    # F-13  Spectral leakage
    # -----------------------------------------------------------------
    prob_spectral_leakage: float = 0.0
    spectral_leakage_window_lengths: list = [1024, 2048, 4096]
    spectral_leakage_max_time_shift: int = 20

    # -----------------------------------------------------------------
    # F-14  Loudness perturbation
    # -----------------------------------------------------------------
    prob_loudness_perturb: float = 0.0
    loudness_factor_range: list = [0.1, 10.0]
    loudness_max_n_intervals: int = 5

    # =================================================================
    def __init__(self):
        import copy
        # Deep-copy all mutable class-level defaults to avoid shared state
        for attr in dir(self.__class__):
            if attr.startswith("_"):
                continue
            val = getattr(self.__class__, attr)
            if isinstance(val, (dict, list)):
                setattr(self, attr, copy.deepcopy(val))

    @classmethod
    def from_dict(cls, d: dict, strict: bool = False):
        import copy, logging
        d = copy.deepcopy(dict(d))
        # 分层配置只在边界归一化一次；内部仍使用已验证的 flat
        # SimulationConfigs 属性，避免五个任务各自解释同一组 YAML section。
        from core.data.simulation_schema import normalize_simulation_cfg

        d = normalize_simulation_cfg(d)
        removed_scene_keys = {
            "prob_clean_only", "prob_nearend_noise", "prob_nearend_echo",
            "prob_nearend_noise_echo", "prob_speech_only", "prob_speech_noise",
            "prob_interference_noise",
        }
        removed = sorted(removed_scene_keys.intersection(d))
        if removed:
            raise ValueError(f"scene keys are removed (no compatibility): {removed}")

        removed_spatial_keys = {
            "num_channels",
            "spatial_rir_policy",
            "spatial_rir_scp",
            "spatial_geometry",
            "microphone_array",
        }
        removed = sorted(removed_spatial_keys.intersection(d))
        if removed:
            raise ValueError(
                "spatial simulation keys are removed (no compatibility): "
                f"{removed}"
            )

        if "schema_version" in d:
            raise ValueError(
                "SimulationConfigs.schema_version is retired; remove this key. "
                "Training simulation_cfg uses one current strict contract."
            )

        # 退役键集合以 ``simulation_config_contract`` 为单一事实来源，避免多处漂移。
        from core.data.simulation_config_contract import (
            RETIRED_FAREND_VOLUME_KEYS,
            RETIRED_LAYER_D_ADC_KEYS,
        )

        retired_adc_keys = tuple(
            key for key in sorted(RETIRED_LAYER_D_ADC_KEYS) if key in d
        )
        if retired_adc_keys:
            raise ValueError(
                "SimulationConfigs retired Layer D ADC keys "
                f"{list(retired_adc_keys)!r}; migrate observation/device clipping to "
                "Layer F with prob_clip, clip_variants=['fixed_rail'], "
                "clip_variant_probs={'fixed_rail': 1.0}, and "
                "clip_pre_drive_db_range."
            )
        if "prob_wind_noise" in d:
            raise ValueError(
                "SimulationConfigs.prob_wind_noise is no longer supported; use "
                "prob_wind_noise_sidechain instead"
            )
        if "array_rotation_z_rad_range" in d:
            raise ValueError(
                "SimulationConfigs.array_rotation_z_rad_range was renamed to "
                "array_rotation_z_deg_range with degrees (0-360); convert each "
                "bound with x * 180.0 / pi"
            )
        if "prob_background_point_source" in d:
            raise ValueError(
                "SimulationConfigs.prob_background_point_source was renamed to "
                "prob_point_source_background; duplicate keys are not accepted"
            )
        if "layer_a" in d:
            raise ValueError(
                "SimulationConfigs.layer_a is no longer supported; move its child "
                "keys to the top level of simulation_cfg"
            )
        retired = [key for key in sorted(RETIRED_FAREND_VOLUME_KEYS) if key in d]
        if retired:
            raise ValueError(
                "SimulationConfigs retired far-end volume keys "
                f"{retired!r}; migrate to far_db_range and erl_range."
            )
        obj = cls()
        from core.data.simulation.tse_scenarios import REMOVED_TSE_KEYS, REMOVED_SHARED_KEYS

        retired = sorted((REMOVED_TSE_KEYS | REMOVED_SHARED_KEYS).intersection(d))
        if retired:
            raise ValueError(f"simulation keys are removed (no compatibility): {retired}")
        valid_attrs = {a for a in dir(cls) if not a.startswith("_") and not callable(getattr(cls, a))}
        unknown = [k for k in d.keys() if k not in valid_attrs]
        for k, v in d.items():
            if hasattr(obj, k):
                existing = getattr(obj, k)
                if k in {
                    "num_augmentations",
                    "clip_variant_probs",
                    "num_interferers",
                    "timeline_mode_weights",
                }:
                    # 这些字段都是完整概率分布；局部 update 会把显式配置与默认键
                    # 混合，改变 selector 的实际抽样合同，因此必须整体替换。
                    setattr(obj, k, v)
                elif isinstance(existing, dict) and isinstance(v, dict):
                    existing.update(v)
                else:
                    setattr(obj, k, v)
        if unknown:
            log = logging.getLogger(__name__)
            log.warning(
                "SimulationConfigs.from_dict: unknown keys (ignored): %s. "
                "Check spelling or add to SimulationConfigs; use strict=True to raise.",
                unknown,
            )
            if strict:
                raise ValueError(f"SimulationConfigs unknown keys: {unknown}")
        obj._validate()
        return obj

    def _validate(self):
        """Validate invariants that are too easy to get wrong in YAML.

        Called automatically by :meth:`from_dict`. Raises :class:`ValueError`
        on the first problem so training fails fast at config-load time.
        Layer E 的正概率增强数量不得超过所引用增强的正权重支持数，禁用时同样校验。
        """
        def _is_bool_like(value) -> bool:
            value_type = type(value)
            return isinstance(value, bool) or (
                value_type.__module__ == "numpy" and value_type.__name__ == "bool_"
            )

        if (isinstance(self.music_ratio, bool) or not isinstance(self.music_ratio, Real)
                or not math.isfinite(self.music_ratio) or not 0 <= self.music_ratio <= 1):
            raise ValueError("music_ratio must be finite within [0, 1]")
        if self.rir_backend not in {"offline", "online"}:
            raise ValueError("rir_backend must be offline or online")
        if self.prob_moving_rir > 0.0 and self.rir_backend != "online":
            raise ValueError(
                "moving RIR requires rir_backend='online'; offline moving assets "
                "are consumed by manifest/pool composition"
            )
        if self.rir_backend == "online" and self.reuse_rir:
            raise ValueError("online scene RIR cannot use reuse_rir")
        if self.snr_reference not in {"primary", "all_speech"}:
            raise ValueError("snr_reference must be primary or all_speech")
        if (not isinstance(self.num_interferers, dict) or not self.num_interferers
                or any(type(k) is not int or k < 1 for k in self.num_interferers)
                or any(isinstance(v, bool) or not isinstance(v, Real)
                       or not math.isfinite(v) or not 0 <= v <= 1 for v in self.num_interferers.values())
                or not math.isclose(sum(self.num_interferers.values()), 1.0, rel_tol=0, abs_tol=1e-8)):
            raise ValueError("num_interferers must map positive integer counts to probabilities summing to 1")
        overlap = self.overlap_ratio_range
        if (not isinstance(overlap, (list, tuple)) or len(overlap) != 2
                or any(isinstance(v, bool) or not isinstance(v, Real)
                       or not math.isfinite(v) or not 0 <= v <= 1 for v in overlap)
                or overlap[0] > overlap[1]):
            raise ValueError("overlap_ratio_range must be ordered within [0, 1]")
        if (
            not isinstance(self.timeline_mode_weights, dict)
            or set(self.timeline_mode_weights) != set(TIMELINE_MODES)
        ):
            raise ValueError(
                "timeline_mode_weights must contain exactly the modes "
                f"{list(TIMELINE_MODES)!r}"
            )
        timeline_weight_total = 0.0
        for mode in TIMELINE_MODES:
            weight = self.timeline_mode_weights[mode]
            if (
                _is_bool_like(weight)
                or not isinstance(weight, Real)
                or not math.isfinite(float(weight))
                or weight < 0.0
            ):
                raise ValueError(
                    "timeline_mode_weights values must be finite non-negative "
                    f"numbers, got {weight!r} for {mode!r}"
                )
            timeline_weight_total += float(weight)
        if not math.isclose(timeline_weight_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "timeline_mode_weights values must sum to 1, "
                f"got {timeline_weight_total!r}"
            )

        # 按最终类的 MRO 注解统一发现功能开关：继承字段会保留，子类同名注解会
        # 覆盖基类语义。仅 ``bool`` 注解进入此校验，避免把 float 概率误判为开关。
        from typing import get_type_hints

        resolved_annotations = get_type_hints(self.__class__)
        bool_attrs = frozenset(
            attr for attr, annotation in resolved_annotations.items() if annotation is bool
        )
        for attr in bool_attrs:
            val = getattr(self, attr)
            if type(val) is not bool:
                raise ValueError(
                    f"SimulationConfigs.{attr} must be a bool, got {val!r}"
                )

        supported_clip_variants = frozenset(
            {"fixed_rail", "hard", "soft", "sigmoid", "sox", "pedal"}
        )
        if not isinstance(self.clip_variants, (list, tuple)) or not self.clip_variants:
            raise ValueError(
                "SimulationConfigs.clip_variants must be a non-empty list/tuple"
            )
        seen_clip_variants = set()
        for variant in self.clip_variants:
            if not isinstance(variant, str) or not variant:
                raise ValueError(
                    "SimulationConfigs.clip_variants entries must be non-empty strings, "
                    f"got {variant!r}"
                )
            if variant in seen_clip_variants:
                raise ValueError(
                    "SimulationConfigs.clip_variants contains duplicate variant "
                    f"{variant!r}"
                )
            if variant not in supported_clip_variants:
                raise ValueError(
                    "SimulationConfigs.clip_variants contains unsupported variant "
                    f"{variant!r}; expected one of {sorted(supported_clip_variants)!r}"
                )
            seen_clip_variants.add(variant)

        if not isinstance(self.clip_variant_probs, dict) or not self.clip_variant_probs:
            raise ValueError(
                "SimulationConfigs.clip_variant_probs must be a non-empty dict"
            )
        if set(self.clip_variant_probs) != seen_clip_variants:
            raise ValueError(
                "SimulationConfigs.clip_variant_probs keys must exactly match "
                f"clip_variants; got variants={list(self.clip_variants)!r}, "
                f"probability_keys={list(self.clip_variant_probs)!r}"
            )
        clip_probability_total = 0.0
        for variant in self.clip_variants:
            probability = self.clip_variant_probs[variant]
            if (
                _is_bool_like(probability)
                or not isinstance(probability, Real)
                or not math.isfinite(float(probability))
                or probability < 0.0
            ):
                raise ValueError(
                    "SimulationConfigs.clip_variant_probs values must be finite "
                    f"non-negative numbers, got {probability!r} for {variant!r}"
                )
            clip_probability_total += float(probability)
        if not math.isclose(
            clip_probability_total, 1.0, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(
                "SimulationConfigs.clip_variant_probs values must sum to 1, "
                f"got {clip_probability_total!r}"
            )

        # Layer E 的增强数量字典直接传给 rng.choice，必须是完整、归一化的概率分布。
        if not isinstance(self.num_augmentations, dict) or not self.num_augmentations:
            raise ValueError(
                "SimulationConfigs.num_augmentations must be a non-empty dict"
            )
        num_augmentation_total = 0.0
        positive_augmentation_counts = []
        for key, probability in self.num_augmentations.items():
            if _is_bool_like(key) or not isinstance(key, Integral) or key < 0:
                raise ValueError(
                    "SimulationConfigs.num_augmentations keys must be non-negative int "
                    f"values, got {key!r}"
                )
            if (
                _is_bool_like(probability)
                or not isinstance(probability, Real)
                or not math.isfinite(float(probability))
                or probability < 0.0
            ):
                raise ValueError(
                    "SimulationConfigs.num_augmentations probabilities must be finite "
                    f"non-negative numbers, got {probability!r} for key {key!r}"
                )
            num_augmentation_total += float(probability)
            if probability > 0.0:
                positive_augmentation_counts.append(int(key))
        if not math.isclose(num_augmentation_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "SimulationConfigs.num_augmentations probabilities must sum to 1, "
                f"got {num_augmentation_total!r}"
            )

        # Layer E 当前 runtime 仅实现这四个增强；mapping 仍允许保留未引用的扩展键。
        supported_augmentation_names = frozenset(
            {"bandwidth_limitation", "clipping", "codec", "packet_loss"}
        )
        # Dataset 初始化时会按 augmentations_name 读取并归一化 weight；缺项、
        # 非有限权重或全零权重必须在进入 np.array/rng.choice 前失败。
        if not isinstance(self.augmentations, dict) or not self.augmentations:
            raise ValueError(
                "SimulationConfigs.augmentations must be a non-empty dict"
            )
        augmentation_weights = {}
        for name, augmentation_cfg in self.augmentations.items():
            if not isinstance(augmentation_cfg, dict):
                raise ValueError(
                    f"SimulationConfigs.augmentations[{name!r}] must be a dict"
                )
            if "weight" not in augmentation_cfg:
                raise ValueError(
                    f"SimulationConfigs.augmentations[{name!r}].weight is required"
                )
            weight = augmentation_cfg["weight"]
            if (
                _is_bool_like(weight)
                or not isinstance(weight, Real)
                or not math.isfinite(float(weight))
                or weight < 0.0
            ):
                raise ValueError(
                    f"SimulationConfigs.augmentations[{name!r}].weight must be a finite "
                    f"non-negative number, got {weight!r}"
                )
            augmentation_weights[name] = float(weight)

        # UniPASE/GAP 的 Layer E 把 bandwidth_limitation 目标采样率固定为 8000；
        # 显式 target_sr 直接成为 runtime 常量，bool/float/str 必须在配置边界拒绝。
        bandwidth_cfg = self.augmentations.get("bandwidth_limitation")
        if isinstance(bandwidth_cfg, dict) and "target_sr" in bandwidth_cfg:
            require_positive_int(
                bandwidth_cfg["target_sr"],
                "SimulationConfigs.augmentations['bandwidth_limitation'].target_sr",
            )

        if not isinstance(self.augmentations_name, (list, tuple)) or not self.augmentations_name:
            raise ValueError(
                "SimulationConfigs.augmentations_name must be a non-empty list/tuple"
            )
        referenced_weight_total = 0.0
        seen_augmentation_names = set()
        for name in self.augmentations_name:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "SimulationConfigs.augmentations_name entries must be non-empty strings, "
                    f"got {name!r}"
                )
            if name in seen_augmentation_names:
                raise ValueError(
                    "SimulationConfigs.augmentations_name contains duplicate augmentation "
                    f"{name!r}"
                )
            if name not in supported_augmentation_names:
                raise ValueError(
                    "SimulationConfigs.augmentations_name contains unsupported runtime "
                    f"augmentation {name!r}; expected one of "
                    f"{sorted(supported_augmentation_names)!r}"
                )
            seen_augmentation_names.add(name)
            if name not in augmentation_weights:
                raise ValueError(
                    "SimulationConfigs.augmentations_name references unknown augmentation "
                    f"{name!r}"
                )
            referenced_weight_total += augmentation_weights[name]
        if referenced_weight_total <= 0.0:
            raise ValueError(
                "SimulationConfigs referenced augmentation weights must have a sum greater than 0"
            )
        max_positive_count = max(positive_augmentation_counts)
        unique_augmentation_count = len(seen_augmentation_names)
        if max_positive_count > unique_augmentation_count:
            raise ValueError(
                "SimulationConfigs.num_augmentations positive-probability count "
                f"{max_positive_count} exceeds {unique_augmentation_count} unique "
                "augmentations_name entries"
            )
        positive_weight_count = sum(
            augmentation_weights[name] > 0.0
            for name in seen_augmentation_names
        )
        if max_positive_count > positive_weight_count:
            raise ValueError(
                "SimulationConfigs.num_augmentations positive-probability count "
                f"{max_positive_count} exceeds {positive_weight_count} positive-weight "
                "augmentations_name entries"
            )

        def _layer_e_range(name, key, *, integer=False, lower=None, upper=None):
            value = self.augmentations[name].get(key)
            path = f"SimulationConfigs.augmentations[{name!r}].{key}"
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    f"{path} must be a 2-element list/tuple [lo, hi], got {value!r}"
                )
            lo, hi = value
            valid_type = (
                all(type(item) is int for item in value)
                if integer
                else all(
                    not _is_bool_like(item) and isinstance(item, Real)
                    for item in value
                )
            )
            if not valid_type or not all(
                math.isfinite(float(item)) for item in value
            ):
                kind = "exact integers" if integer else "finite numbers"
                raise ValueError(f"{path} elements must be {kind}, got {value!r}")
            if (
                lo > hi
                or (lower is not None and lo < lower)
                or (upper is not None and hi > upper)
            ):
                raise ValueError(f"{path} has invalid bounds {value!r}")

        # Layer E runtime 直接解包这些参数；在抽到对应操作前完成结构与 DSP 域校验。
        if "clipping" in seen_augmentation_names:
            _layer_e_range(
                "clipping", "clipping_min_quantile", lower=0.0, upper=1.0
            )
            _layer_e_range(
                "clipping", "clipping_max_quantile", lower=0.0, upper=1.0
            )
        if "codec" in seen_augmentation_names:
            codec_config = self.augmentations["codec"].get("config")
            if not isinstance(codec_config, (list, tuple)) or not codec_config:
                raise ValueError(
                    "SimulationConfigs.augmentations['codec'].config must be a "
                    "non-empty list/tuple"
                )
            for index, codec in enumerate(codec_config):
                path = f"SimulationConfigs.augmentations['codec'].config[{index}]"
                if not isinstance(codec, dict) or codec.get("format") not in {
                    "mp3",
                    "ogg",
                }:
                    raise ValueError(f"{path}.format must be 'mp3' or 'ogg'")
                encoder = codec.get("encoder")
                encoders = encoder if isinstance(encoder, list) else [encoder]
                if (
                    isinstance(encoder, tuple)
                    or not encoders
                    or any(
                        value not in {None, "None", "vorbis", "opus"}
                        for value in encoders
                    )
                ):
                    raise ValueError(
                        f"{path}.encoder must be a supported scalar or non-empty list"
                    )
                qscale = codec.get("qscale")
                if type(qscale) is int:
                    qscale = [qscale, qscale]
                if (
                    not isinstance(qscale, (list, tuple))
                    or len(qscale) != 2
                    or any(type(value) is not int for value in qscale)
                    or qscale[0] > qscale[1]
                    or qscale[0] < -1
                ):
                    raise ValueError(
                        f"{path}.qscale must be an exact int or ordered exact-int "
                        "range with lo >= -1"
                    )
        if "packet_loss" in seen_augmentation_names:
            packet_cfg = self.augmentations["packet_loss"]
            for key in ("packet_duration_ms", "max_continuous_packet_loss"):
                value = packet_cfg.get(key)
                if type(value) is not int or value <= 0:
                    raise ValueError(
                        "SimulationConfigs.augmentations['packet_loss']."
                        f"{key} must be an exact int > 0, got {value!r}"
                    )
            _layer_e_range(
                "packet_loss", "packet_loss_rate", lower=0.0, upper=1.0
            )

        # 所有白名单 range 必须是有限数值区间；bool / np.bool_ 不是数值端点。
        for attr in self.__class__._RANGE_ATTRS:
            val = getattr(self, attr)
            if val is None:
                continue
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(
                    f"SimulationConfigs.{attr} must be a 2-element list/tuple [lo, hi], got {val!r}"
                )
            lo, hi = val
            if any(_is_bool_like(item) or not isinstance(item, Real) for item in val):
                raise ValueError(
                    f"SimulationConfigs.{attr} elements must be numeric, got {val!r}"
                )
            try:
                lo_f, hi_f = float(lo), float(hi)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"SimulationConfigs.{attr} elements must be numeric, got {val!r}"
                ) from exc
            if not math.isfinite(lo_f) or not math.isfinite(hi_f):
                raise ValueError(
                    f"SimulationConfigs.{attr} elements must be finite, got {val!r}"
                )
            if lo_f > hi_f:
                raise ValueError(
                    f"SimulationConfigs.{attr} must satisfy lo <= hi, got [{lo_f}, {hi_f}]"
                )

        # 下列 Layer-F 频率范围会在 runtime 与目标采样率的 Nyquist 相交；配置
        # 本身不得含负频率，具体采样率下没有交集时由算子显式跳过。
        for attr in (
            "lowpass_cutoff_range",
            "band_reject_center_freq_range",
            "band_reject_freq_bw_range",
            "bass_boost_highpass_cutoff_range",
        ):
            value = getattr(self, attr)
            if float(value[0]) < 0.0:
                raise ValueError(
                    f"SimulationConfigs.{attr} must contain non-negative frequencies, "
                    f"got {value!r}"
                )
        # randint / FFT 等离散 DSP 参数不得接受 bool、float 或数值字符串。
        exact_int_ranges = {
            "lowpass_order_range": 1,
            "bit_crush_range": 1,
            "spectral_holes_num_range": 1,
            "spectral_holes_width_freq_range": 1,
            "spectral_holes_width_time_range": 1,
        }
        for attr, minimum in exact_int_ranges.items():
            value = getattr(self, attr)
            if any(type(item) is not int for item in value) or value[0] < minimum:
                raise ValueError(
                    f"SimulationConfigs.{attr} must contain exact integers >= "
                    f"{minimum}, got {value!r}"
                )

        exact_int_scalars = {
            "speech_splice_max_repeat_times": 0,
            "band_reject_max_n": 1,
            "spectral_holes_frame_length": 1,
            "spectral_holes_hop_length": 1,
            "spectral_leakage_max_time_shift": 0,
            "loudness_max_n_intervals": 1,
        }
        for attr, minimum in exact_int_scalars.items():
            value = getattr(self, attr)
            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"SimulationConfigs.{attr} must be an exact int >= {minimum}, "
                    f"got {value!r}"
                )
        if self.spectral_holes_hop_length > self.spectral_holes_frame_length:
            raise ValueError(
                "SimulationConfigs.spectral_holes_hop_length must be <= "
                "spectral_holes_frame_length"
            )

        duration_domains = {
            "speech_splice_seconds": False,
            "early_rir_sec": True,
            "segment_sec": False,
        }
        for attr, allow_zero in duration_domains.items():
            value = getattr(self, attr)
            valid_lower = value >= 0.0 if isinstance(value, Real) else False
            if not allow_zero and valid_lower:
                valid_lower = value > 0.0
            if (
                _is_bool_like(value)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not valid_lower
            ):
                relation = ">= 0" if allow_zero else "> 0"
                raise ValueError(
                    f"SimulationConfigs.{attr} must be a finite duration {relation}, "
                    f"got {value!r}"
                )

        discrete_selectors = {
            "colored_noise_types": (
                frozenset({"white", "pink", "brown", "equalized"}),
                str,
            ),
            "codecs_types": (frozenset({"mp3", "gsm", "opus", "aac"}), str),
            "webrtc_ns_levels": (frozenset({0, 1, 2, 3}), int),
        }
        for attr, (allowed, exact_type) in discrete_selectors.items():
            value = getattr(self, attr)
            valid_items = isinstance(value, (list, tuple)) and bool(value)
            if valid_items:
                valid_items = all(
                    type(item) is exact_type and item in allowed for item in value
                )
            if not valid_items or len(set(value)) != len(value):
                raise ValueError(
                    f"SimulationConfigs.{attr} must be a non-empty unique selector "
                    f"list from {sorted(allowed)!r}, got {value!r}"
                )
        window_lengths = self.spectral_leakage_window_lengths
        valid_window_lengths = isinstance(window_lengths, (list, tuple)) and bool(
            window_lengths
        )
        if valid_window_lengths:
            valid_window_lengths = all(
                type(value) is int and value > 0 for value in window_lengths
            )
        if (
            not valid_window_lengths
            or len(set(window_lengths)) != len(window_lengths)
        ):
            raise ValueError(
                "SimulationConfigs.spectral_leakage_window_lengths must be a non-empty "
                f"unique list of exact ints > 0, got {window_lengths!r}"
            )

        peak_scale_range = self.peak_normalization_scale_range
        if not (
            0.0
            < float(peak_scale_range[0])
            <= float(peak_scale_range[1])
            <= 1.0
        ):
            raise ValueError(
                "SimulationConfigs.peak_normalization_scale_range must satisfy "
                f"0 < lo <= hi <= 1, got {peak_scale_range!r}"
            )

        for attr in (
            "speech_splice_seconds_range",
            "speech_splice_gap_range",
            "silence_pad_total_range",
            "silence_pad_short_range",
            "silence_pad_long_range",
        ):
            val = getattr(self, attr)
            if val is not None and float(val[0]) < 0.0:
                raise ValueError(
                    f"SimulationConfigs.{attr} durations must be non-negative, got {val!r}"
                )
        for attr in ("drc_attack_ms_range", "drc_release_ms_range"):
            val = getattr(self, attr)
            if float(val[0]) <= 0.0:
                raise ValueError(
                    f"SimulationConfigs.{attr} durations must be positive, got {val!r}"
                )

        # 主加噪 SNR 是两个独立标量，采用严格实数策略并校验顺序。
        snr_values = {}
        for attr in ("snr_low", "snr_high"):
            val = getattr(self, attr)
            if _is_bool_like(val) or not isinstance(val, Real):
                raise ValueError(
                    f"SimulationConfigs.{attr} must be a real number, got {val!r}"
                )
            value = float(val)
            if not math.isfinite(value):
                raise ValueError(
                    f"SimulationConfigs.{attr} must be finite, got {val!r}"
                )
            snr_values[attr] = value
        if snr_values["snr_low"] > snr_values["snr_high"]:
            raise ValueError(
                "SimulationConfigs.snr_low must be <= snr_high, "
                f"got {self.snr_low!r} > {self.snr_high!r}"
            )

        # wind_noise_config 内的 range 不属于顶层白名单，需显式校验类型、有限性和 DSP 领域。
        if not isinstance(self.wind_noise_config, dict):
            raise ValueError(
                "SimulationConfigs.wind_noise_config must be a dict, "
                f"got {self.wind_noise_config!r}"
            )
        wind_ranges = {}
        for attr in (
            "threshold",
            "ratio",
            "attack",
            "release",
            "sc_gain",
            "clipping_threshold",
            "wind_noise_snr_range",
        ):
            val = self.wind_noise_config.get(attr)
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(
                    f"SimulationConfigs.wind_noise_config.{attr} must be a 2-element list/tuple [lo, hi], got {val!r}"
                )
            if any(_is_bool_like(item) or not isinstance(item, Real) for item in val):
                raise ValueError(
                    f"SimulationConfigs.wind_noise_config.{attr} elements must be numeric, got {val!r}"
                )
            lo_f, hi_f = float(val[0]), float(val[1])
            if not math.isfinite(lo_f) or not math.isfinite(hi_f):
                raise ValueError(
                    f"SimulationConfigs.wind_noise_config.{attr} elements must be finite, got {val!r}"
                )
            if lo_f > hi_f:
                raise ValueError(
                    f"SimulationConfigs.wind_noise_config.{attr} must satisfy lo <= hi, got [{lo_f}, {hi_f}]"
                )
            wind_ranges[attr] = (lo_f, hi_f)

        wind_domains = {
            "threshold": lambda lo, hi: 0.0 < lo and hi <= 1.0,
            "ratio": lambda lo, hi: lo >= 1.0,
            "attack": lambda lo, hi: lo > 0.0,
            "release": lambda lo, hi: lo > 0.0,
            "sc_gain": lambda lo, hi: lo > 0.0,
            "clipping_threshold": lambda lo, hi: 0.0 <= lo and hi <= 1.0,
        }
        for attr, is_valid in wind_domains.items():
            lo_f, hi_f = wind_ranges[attr]
            if not is_valid(lo_f, hi_f):
                raise ValueError(
                    f"SimulationConfigs.wind_noise_config.{attr} has invalid bounds "
                    f"{self.wind_noise_config[attr]!r}"
                )

        clipping_chance = self.wind_noise_config.get("clipping_chance")
        if _is_bool_like(clipping_chance) or not isinstance(clipping_chance, Real):
            raise ValueError(
                "SimulationConfigs.wind_noise_config.clipping_chance must be a real number, "
                f"got {clipping_chance!r}"
            )
        if not math.isfinite(float(clipping_chance)) or not 0.0 <= clipping_chance <= 1.0:
            raise ValueError(
                "SimulationConfigs.wind_noise_config.clipping_chance must be a finite "
                f"probability in [0.0, 1.0], got {clipping_chance!r}"
            )

        # Spatial observation 使用 canonical array_profile 或显式 custom geometry；
        # profile 只描述物理拓扑，不携带任务/模型命名。
        if type(self.spatial_enabled) is not bool:
            raise ValueError("SimulationConfigs.spatial_enabled must be a bool")
        from core.data.simulation.array_profiles import (
            normalize_geometry_provenance,
            normalize_validation_recipe,
            resolve_microphone_positions,
            validate_reference_channel,
        )
        custom_geometry = self.custom_geometry
        if self.spatial_enabled and self.array_profile is None and custom_geometry is None:
            raise ValueError(
                "spatial_enabled=True requires array_profile or custom_geometry; "
                "microphone count is derived from the resolved geometry"
            )
        geometry_probe = resolve_microphone_positions(
            array_profile=self.array_profile,
            custom_geometry=custom_geometry,
            reference_channel=0,
        )
        channels = len(geometry_probe) if geometry_probe is not None else 1
        if self.reference_channel == "auto":
            if channels == 1 or not self.spatial_enabled:
                self.reference_channel = 0
            elif channels % 2:
                self.reference_channel = channels // 2
            else:
                raise ValueError(
                    "reference_channel must be explicit for an even-channel array; "
                    "only odd-channel arrays have an unambiguous center microphone"
                )
        if type(self.reference_channel) is not int or not 0 <= self.reference_channel < channels:
            raise ValueError(
                "SimulationConfigs.reference_channel must be an exact int in [0, resolved geometry channels)"
            )
        microphone_positions = resolve_microphone_positions(
            array_profile=self.array_profile,
            custom_geometry=custom_geometry,
            reference_channel=self.reference_channel,
        )
        if microphone_positions is not None and len(microphone_positions) != channels:
            raise ValueError("resolved microphone geometry changed channel count")
        validate_reference_channel(self.reference_channel, channels)
        OnlineRIRConfig.from_simulation(self)
        if not self.spatial_enabled and (
            self.array_profile is not None
            or self.custom_geometry is not None
        ):
            raise ValueError(
                "array_profile and custom_geometry require spatial_enabled=True"
            )
        self.geometry_provenance = normalize_geometry_provenance(self.geometry_provenance)
        self.validation_recipe = normalize_validation_recipe(self.validation_recipe)
        if self.rir_backend == "online" and self.spatial_enabled:
            if microphone_positions is None:
                raise ValueError(
                    "spatial online RIR requires array_profile or microphone geometry"
                )
        spatial_policy_defaults = {
            "spatial_layer_d_policy": "shared",
            "spatial_layer_e_policy": "shared",
            "spatial_layer_f_policy": "per_channel",
        }
        for attr, default in spatial_policy_defaults.items():
            if getattr(self, attr) not in {"shared", "per_channel"}:
                raise ValueError(f"SimulationConfigs.{attr} must be shared or per_channel")
            if not self.spatial_enabled and getattr(self, attr) != default:
                raise ValueError(f"SimulationConfigs.{attr} requires spatial_enabled=True")
        selector_values = {
            "target_type": frozenset({"early", "dry", "rts"}),
            "spatial_diffuse_coherence_model": frozenset({"habets_spherical"}),
            "silence_pad_position": frozenset(
                {"random", "front", "back", "center", "turn_boundary", "scene_aware"}
            ),
            "silence_pad_mode": frozenset({"uniform", "bimodal"}),
            "peak_normalization_mode": frozenset({"off", "joint"}),
        }
        for attr, allowed in selector_values.items():
            val = getattr(self, attr)
            if not isinstance(val, str) or val not in allowed:
                allowed_text = "off/joint" if attr == "peak_normalization_mode" else sorted(allowed)
                raise ValueError(
                    f"SimulationConfigs.{attr} must be one of {allowed_text!r}, got {val!r}"
                )
        for attr in ("prob_array_rotation", "prob_microphone_position_jitter"):
            value = getattr(self, attr)
            if (
                _is_bool_like(value)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"SimulationConfigs.{attr} must be a numeric probability in [0, 1]")
        if self.target_type == "rts" and not self.spatial_enabled:
            raise ValueError("SimulationConfigs.target_type='rts' requires spatial_enabled=True")
        if self.prob_point_source_background > 0.0:
            if not self.spatial_enabled:
                raise ValueError(
                    "SimulationConfigs.prob_point_source_background requires spatial_enabled=True"
                )
            if self.rir_backend != "online":
                raise ValueError(
                    "SimulationConfigs.prob_point_source_background requires rir_backend='online'"
                )
        if self.prob_wind_noise_background > 0.0 and not self.spatial_enabled:
            raise ValueError(
                "SimulationConfigs.prob_wind_noise_background requires spatial_enabled=True"
            )
        if not self.spatial_enabled and (
            self.prob_array_rotation > 0.0
            or self.prob_microphone_position_jitter > 0.0
        ):
            raise ValueError("array rotation and microphone position jitter require spatial_enabled=True")
        for attr, allow_zero in (
            ("moving_source_speed_range_mps", False),
            ("microphone_position_jitter_m_range", True),
        ):
            lo, hi = (float(value) for value in getattr(self, attr))
            if lo < 0.0 or (not allow_zero and lo == 0.0):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"SimulationConfigs.{attr} must be {qualifier}")
        for attr, allow_zero in (
            ("rts_target_t60_sec", False),
            ("rts_time_after_max_sec", True),
        ):
            value = getattr(self, attr)
            if (
                _is_bool_like(value)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or (value < 0.0 if allow_zero else value <= 0.0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"SimulationConfigs.{attr} must be finite and {qualifier}")

        short_ratio = self.silence_pad_short_ratio
        if _is_bool_like(short_ratio) or not isinstance(short_ratio, Real):
            raise ValueError(
                "SimulationConfigs.silence_pad_short_ratio must be a real number, "
                f"got {short_ratio!r}"
            )
        if not math.isfinite(float(short_ratio)) or not 0.0 <= short_ratio <= 1.0:
            raise ValueError(
                "SimulationConfigs.silence_pad_short_ratio must be a finite ratio in "
                f"[0.0, 1.0], got {short_ratio!r}"
            )

        # Distribution selectors must be one of the supported values.
        for attr in (
            "source_volume_distribution",
            "erl_distribution",
        ):
            val = getattr(self, attr)
            if val not in ("uniform", "beta"):
                raise ValueError(
                    f"SimulationConfigs.{attr} must be 'uniform' or 'beta', got {val!r}"
                )

        # 所有共享 Beta shape 参数必须是有限正数，禁止布尔值或 NaN 绕过校验。
        for attr in (
            "source_volume_beta_a",
            "source_volume_beta_b",
            "erl_beta_a",
            "erl_beta_b",
            "silence_pad_beta_a",
            "silence_pad_beta_b",
        ):
            val = getattr(self, attr)
            if _is_bool_like(val) or not isinstance(val, Real):
                raise ValueError(
                    f"SimulationConfigs.{attr} must be a finite positive number, got {val!r}"
                )
            if not math.isfinite(float(val)):
                raise ValueError(
                    f"SimulationConfigs.{attr} must be finite, got {val!r}"
                )
            if val <= 0.0:
                raise ValueError(
                    f"SimulationConfigs.{attr} must be a positive number, got {val!r}"
                )

        for attr, upper in (
            ("spectral_holes_cutoff_freq", None),
            ("bass_boost_attenuation_min_db", -5.0),
        ):
            val = getattr(self, attr)
            if (
                _is_bool_like(val)
                or not isinstance(val, Real)
                or not math.isfinite(float(val))
                or (upper is None and float(val) < 0.0)
                or (upper is not None and float(val) > upper)
            ):
                relation = "non-negative" if upper is None else f"<= {upper:g} dB"
                raise ValueError(
                    f"SimulationConfigs.{attr} must be a finite {relation} value, got {val!r}"
                )

        # ``clip_hard_portion`` 是 hard 分支 selector 的 Bernoulli 概率，但其名称
        # 不匹配通用 prob_* 约定，必须显式校验。
        clip_hard_portion = self.clip_hard_portion
        if (
            _is_bool_like(clip_hard_portion)
            or not isinstance(clip_hard_portion, Real)
            or not math.isfinite(float(clip_hard_portion))
            or not 0.0 <= float(clip_hard_portion) <= 1.0
        ):
            raise ValueError(
                "SimulationConfigs.clip_hard_portion must be a finite probability in "
                f"[0.0, 1.0], got {clip_hard_portion!r}"
            )

        # 概率字段必须是 [0, 1] 内的数值；bool 注解开关已由上方统一校验。
        for attr in dir(self.__class__):
            if attr.startswith("_") or not (attr.startswith("prob_") or attr.endswith("_prob")):
                continue
            val = getattr(self, attr)
            if attr in bool_attrs:
                continue
            if (
                _is_bool_like(val)
                or not isinstance(val, Real)
                or not math.isfinite(float(val))
                or not (0.0 <= val <= 1.0)
            ):
                raise ValueError(
                    f"SimulationConfigs.{attr} must be a numeric probability in [0.0, 1.0], got {val!r}"
                )

    # ------------------------------------------------------------------
    # 所有范围型参数统一使用 *_range 命名。
    # ------------------------------------------------------------------
