"""TSE (Target Speaker Extraction，目标说话人提取) 任务包。

数据仿真复用 core A–F 六层管线，新增干扰人声支路与 Layer C 音乐噪声源扩展；
enrollment 独立处理（Layer A-enroll + Layer C-enroll），见 ``tasks/tse/data/dynamic_dataset.py``。
"""
