"""SSL 配置字段解析，供训练与离线验证入口共享。"""

from collections.abc import Mapping


def resolve_ssl_num_tracks(config: Mapping) -> int:
    """解析 canonical 声源数并检查兼容字段与 PIT 容量。

    ``ssl.num_tracks`` 是首选入口。未声明时从既有的模型或 train/validation
    Dataset 字段推导，以保持旧配置可用。重复声明必须一致；PIT loss 的
    ``max_tracks`` 是容量上限，省略时沿用 loss 默认值 8。
    """

    if not isinstance(config, Mapping):
        raise TypeError("SSL config must be a mapping")

    ssl_cfg = config.get("ssl")
    has_canonical = isinstance(ssl_cfg, Mapping) and "num_tracks" in ssl_cfg
    canonical = None
    if has_canonical:
        canonical = ssl_cfg["num_tracks"]
        if type(canonical) is not int or canonical < 1:
            raise ValueError("ssl.num_tracks must be an exact positive integer")

    declared: list[tuple[str, int]] = []
    raw_model_cfg = config.get("model")
    model_cfg = {} if raw_model_cfg is None else raw_model_cfg
    if not isinstance(model_cfg, Mapping):
        raise TypeError("model must be a mapping")
    model_args = model_cfg.get("args", {})
    if not isinstance(model_args, Mapping):
        raise TypeError("model.args must be a mapping")
    if "max_track" in model_args:
        value = model_args["max_track"]
        if type(value) is not int or value < 1:
            raise ValueError("model.args.max_track must be an exact positive integer")
        declared.append(("model.args.max_track", value))

    raw_loss_cfg = config.get("loss")
    loss_cfg = {} if raw_loss_cfg is None else raw_loss_cfg
    if not isinstance(loss_cfg, Mapping):
        raise TypeError("loss must be a mapping")
    raw_loss_args = loss_cfg.get("args")
    loss_args = {} if raw_loss_args is None else raw_loss_args
    if not isinstance(loss_args, Mapping):
        raise TypeError("loss.args must be a mapping")
    loss_type = loss_cfg.get("type")
    loss_max_tracks = loss_args.get("max_tracks")
    if "max_tracks" in loss_args:
        if type(loss_max_tracks) is not int or loss_max_tracks < 1:
            raise ValueError("loss.args.max_tracks must be an exact positive integer")

    raw_data = config.get("data")
    data = {} if raw_data is None else raw_data
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")
    for split in ("train", "validation"):
        section = data.get(split)
        if not isinstance(section, Mapping):
            continue
        split_values: list[tuple[str, int]] = []
        for field in ("tracks", "max_track"):
            if field not in section:
                continue
            value = section[field]
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"data.{split}.{field} must be an exact positive integer"
                )
            split_values.append((field, value))
        if len({value for _, value in split_values}) > 1:
            raise ValueError(
                f"data.{split}.tracks and data.{split}.max_track must be equal"
            )
        declared.extend(
            (f"data.{split}.{field}", value)
            for field, value in split_values
        )

    if has_canonical:
        conflicts = [name for name, value in declared if value != canonical]
        if conflicts:
            raise ValueError(
                "ssl.num_tracks must match existing source-count fields: "
                f"{', '.join(conflicts)}"
            )
        resolved = canonical
    else:
        if not declared:
            raise ValueError(
                "SSL source count is missing; set ssl.num_tracks or an existing "
                "data.{train,validation}.tracks/model.args.max_track field"
            )
        resolved = declared[0][1]
        conflicts = [name for name, value in declared if value != resolved]
        if conflicts:
            raise ValueError(
                "SSL source-count fields must all be equal; conflicts: "
                f"{', '.join(conflicts)}"
            )

    if loss_type in ("SSLDPIPDPITLoss", "SSLDOACompositeLoss"):
        loss_capacity = 8 if "max_tracks" not in loss_args else loss_max_tracks
        if loss_capacity < resolved:
            raise ValueError(
                "SSL PIT loss max_tracks capacity must be >= canonical source count "
                f"ssl.num_tracks={resolved}"
            )
    return resolved
