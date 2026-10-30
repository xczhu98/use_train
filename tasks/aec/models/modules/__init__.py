"""AEC 模型内部实现模块库（对称 ``tasks/se/models/modules``）。

仅存放被 ``tasks/aec/models`` 顶层模型复用的实现细节，不注册 registry、不感知
STFT/训练框架契约；顶层薄封装负责注册与接口适配。
"""
