"""VAD 路径/环境引导工具（从 processors/vad.py 拆出）。

注意：本文件位于 processors/vad/ 下，比原 processors/vad.py 深一层，
所有基于 ``__file__`` 的 use_train 根 / third_party 路径推导均已多上一层
``os.path.dirname``（相对路径候选也相应多一级 ``..``）。
"""
import contextlib
import logging
import os
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)

# 并发首次下载互斥参数：锁等待超时与轮询间隔（测试可经 monkeypatch 调小）
_DOWNLOAD_LOCK_TIMEOUT_SEC = 1800.0
_DOWNLOAD_LOCK_POLL_SEC = 1.0


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """临时抑制 stdout/stderr（含 fd 级别），屏蔽第三方 VAD 的 rtf_avg/tqdm 等输出。"""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        devnull_fd = devnull.fileno()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        saved_fds = []  # (saved_fd, target_fd) 列表，finally 中逐个恢复
        try:
            sys.stdout, sys.stderr = devnull, devnull
            # tqdm 等库可能直接写 fd，需在 fd 层面重定向；逐个保存+重定向，
            # 任一步失败只影响已记录的 fd，finally 会精确恢复原状——旧实现
            # 若 dup2(devnull, 2) 失败会把两个 save fd 都置 None，导致 stdout
            # 永久指向 devnull 且 fd 泄漏
            for target_fd in (1, 2):
                try:
                    saved_fds.append((os.dup(target_fd), target_fd))
                    os.dup2(devnull_fd, target_fd)
                except (OSError, AttributeError):
                    break
            try:
                yield
            finally:
                # 先恢复 stderr 再恢复 stdout
                for saved_fd, target_fd in reversed(saved_fds):
                    try:
                        os.dup2(saved_fd, target_fd)
                        os.close(saved_fd)
                    except OSError:
                        pass
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def _ensure_fireredvad_path():
    """Ensure fireredvad is importable: prefer pip package, else add repo to path.
    Official: https://github.com/FireRedTeam/FireRedVAD
    pip install fireredvad  # recommended; then no path setup needed.
    """
    try:
        import fireredvad
        return
    except ImportError:
        pass

    current_dir = os.path.dirname(os.path.abspath(__file__))
    processors_dir = os.path.dirname(current_dir)
    pipeline_root = os.path.dirname(processors_dir)
    tools_dir = os.path.dirname(pipeline_root)
    use_train_root = os.path.dirname(tools_dir)
    third_party = os.environ.get("USE_TRAIN_THIRD_PARTY_DIR") or os.path.join(use_train_root, "third_party_scripts")
    # Repo name: FireRedVAD (clone) or FireRedVAD-main (GitHub zip)
    potential_paths = [
        os.path.join(third_party, "FireRedVAD"),
        os.path.join(current_dir, "../../../../FireRedVAD"),
        os.path.join(current_dir, "../../../../FireRedVAD-main"),
        os.path.join(current_dir, "../../../../../FireRedVAD"),
        os.path.join(current_dir, "../../../../../FireRedVAD-main"),
        os.path.abspath(os.path.join(current_dir, "../../../../FireRedVAD")),
    ]
    for path in potential_paths:
        path = os.path.abspath(path) if os.path.isabs(path) else os.path.normpath(path)
        if os.path.exists(path) and os.path.isdir(path):
            inserted = path not in sys.path
            if inserted:
                sys.path.insert(0, path)
            try:
                import fireredvad
                logger.debug("Successfully added FireRedVAD from %s", path)
                return
            except ImportError:
                # import 失败必须回滚 sys.path：无效候选路径残留会污染后续
                # import 解析（同名包遮蔽），且每个候选都会累积一条脏路径
                if inserted:
                    try:
                        sys.path.remove(path)
                    except ValueError:
                        pass
                continue

_ensure_fireredvad_path()


_fireredvad_root_cache: Optional[str] = None


def _lock_holder_alive(lock_path: str) -> bool:
    """读取下载锁中的持有 pid，判断持有进程是否仍存活。

    锁内容为持锁进程写入的 pid；读取失败/无内容时保守视为存活
    （不主动破坏未知状态的锁）。pid 复用会导致误判存活，后果仅是
    继续等待到超时（保守方向，可接受）。
    """
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # PermissionError（存在但无权限）等：视为存活
        return True
    return True


def _get_or_download_fireredvad_root() -> Optional[str]:
    """返回 FireRedVAD 权重根目录（其下含 VAD/、AED/）。若不存在则尝试自动下载到缓存。"""
    global _fireredvad_root_cache
    if _fireredvad_root_cache is not None:
        return _fireredvad_root_cache
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processors_dir = os.path.dirname(current_dir)
    pipeline_root = os.path.dirname(processors_dir)
    tools_dir = os.path.dirname(pipeline_root)
    use_train_root = os.path.dirname(tools_dir)
    third_party = os.environ.get("USE_TRAIN_THIRD_PARTY_DIR") or os.path.join(use_train_root, "third_party_scripts")
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    cache_root = os.path.join(cache_home, "use_train", "FireRedVAD")
    # 完成标记定义前移：step-2 缓存预检与第 4 步下载互斥都要用到它
    done_marker = os.path.join(cache_root, ".download_complete")

    def _valid_root(path: str) -> bool:
        if not path or not os.path.isdir(path):
            return False
        vad_dir = os.path.join(path, "VAD")
        return os.path.isdir(vad_dir)

    # 1) third_party_scripts/FireRedVAD
    candidate = os.path.join(third_party, "FireRedVAD")
    if _valid_root(candidate):
        _fireredvad_root_cache = candidate
        return candidate
    # 2) 用户缓存 ~/.cache/use_train/FireRedVAD
    # 只查 _valid_root（VAD/ 目录存在）不够：snapshot_download 中途崩溃会留下
    # 有 VAD/ 却缺权重的半成品，若被当有效缓存返回，永远走不到第 4 步的锁与
    # 下载。必须同时要求完成标记。对升级前已完整下载、无 marker 的旧缓存，
    # VAD/ 目录非空即视为完整并补写 marker 后返回（向后兼容启发式）。
    if _valid_root(cache_root):
        if os.path.exists(done_marker):
            _fireredvad_root_cache = cache_root
            return cache_root
        vad_dir = os.path.join(cache_root, "VAD")
        vad_file_count = sum(
            1 for f in os.listdir(vad_dir) if os.path.isfile(os.path.join(vad_dir, f))
        )
        if vad_file_count > 0:
            try:
                with open(done_marker, "w", encoding="utf-8") as f:
                    f.write("ok\n")
            except OSError:
                logger.warning("FireRedVAD 缓存补写完成标记失败: %s", done_marker)
            _fireredvad_root_cache = cache_root
            return cache_root
        # VAD/ 存在但为空：视为半成品，落入第 4 步的锁与下载
        logger.warning(
            "FireRedVAD 用户缓存 %s 的 VAD/ 目录为空，视为半成品，重新下载",
            cache_root,
        )
    # 3) 当前目录 pretrained_models（processors/ 一级，与拆分前 processors/vad.py 口径一致）
    for base in (use_train_root, processors_dir, os.getcwd()):
        candidate = os.path.join(base, "pretrained_models", "FireRedVAD")
        if _valid_root(candidate):
            _fireredvad_root_cache = candidate
            return candidate

    # 4) 自动下载到 ~/.cache/use_train/FireRedVAD
    # 并发首次下载互斥：VAD worker 等多进程同时进入下载会互相写出半成品
    # 权重。用 O_CREAT|O_EXCL 原子创建锁文件，拿到锁的进程下载并写完成
    # 标记；其余进程等锁释放（或看到完成标记）后复用结果，不再重复下载。
    os.makedirs(cache_root, exist_ok=True)
    lock_path = cache_root + ".download.lock"

    def _cache_ready() -> bool:
        # 完成标记区分"完整下载"与"半成品"：仅 VAD/ 目录存在不足以证明
        # 下载完成（snapshot_download 可能中途崩溃留下部分文件）
        return _valid_root(cache_root) and os.path.exists(done_marker)

    lock_fd: Optional[int] = None
    wait_deadline = time.monotonic() + _DOWNLOAD_LOCK_TIMEOUT_SEC
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            # 锁内写持有 pid：供等待方做 stale 锁自愈（持有进程被 SIGKILL/
            # OOM 后锁文件残留，否则后续每次运行都干等满 timeout）
            os.write(lock_fd, f"{os.getpid()}\n".encode())
        except FileExistsError:
            if _cache_ready():
                # 另一个进程已完成下载，直接复用
                _fireredvad_root_cache = cache_root
                return cache_root
            if not _lock_holder_alive(lock_path):
                # 持有进程已死：清理残留锁重新竞争，避免"下一次运行重新
                # 竞争"的注释承诺失效（锁文件无人清理时每轮白等 1800s）
                logger.warning("FireRedVAD 下载锁持有进程已退出，清理残留锁重抢")
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
                continue
            if time.monotonic() > wait_deadline:
                # 锁文件残留（持有进程存活但下载过久/卡死）：放弃本次下载
                # 走无模型 fallback，避免死锁；下一次运行重新竞争
                logger.warning(
                    "FireRedVAD 下载锁等待超时（%s 残留？），跳过自动下载", lock_path
                )
                return None
            time.sleep(_DOWNLOAD_LOCK_POLL_SEC)
    try:
        # 拿锁后重新校验：等待期间可能已有进程完成下载
        if _cache_ready():
            _fireredvad_root_cache = cache_root
            return cache_root
        backend = "auto"
        try:
            import modelscope  # noqa: F401
            backend = "modelscope"
        except ImportError:
            backend = "huggingface"
        if backend == "modelscope":
            try:
                from modelscope import snapshot_download
                snapshot_download("xukaituo/FireRedVAD", local_dir=cache_root)
                if _valid_root(cache_root):
                    with open(done_marker, "w", encoding="utf-8") as f:
                        f.write("ok\n")
                    _fireredvad_root_cache = cache_root
                    logger.debug("FireRedVAD 已自动下载到 %s", cache_root)
                    return cache_root
            except Exception as e:
                logger.debug("FireRedVAD ModelScope 自动下载失败: %s，尝试 HuggingFace", e)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="FireRedTeam/FireRedVAD", local_dir=cache_root)
            if _valid_root(cache_root):
                with open(done_marker, "w", encoding="utf-8") as f:
                    f.write("ok\n")
                _fireredvad_root_cache = cache_root
                logger.debug("FireRedVAD 已自动下载到 %s", cache_root)
                return cache_root
        except Exception as e:
            logger.warning(
                "FireRedVAD 自动下载失败: %s。请手动下载 FireRedTeam/FireRedVAD "
                "到 USE_TRAIN_THIRD_PARTY_DIR（或 use_train/third_party_scripts）下的 FireRedVAD 目录",
                e,
            )
        return None
    finally:
        os.close(lock_fd)
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def _get_default_nemo_onnx_path() -> Optional[str]:
    """返回 third_party_scripts 下 NeMo VAD ONNX 默认路径（若文件存在）。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processors_dir = os.path.dirname(current_dir)
    pipeline_root = os.path.dirname(processors_dir)
    tools_dir = os.path.dirname(pipeline_root)
    use_train_root = os.path.dirname(tools_dir)
    third_party = os.environ.get("USE_TRAIN_THIRD_PARTY_DIR") or os.path.join(use_train_root, "third_party_scripts")
    for name in ("vad_marblenet.onnx", "vad.onnx"):
        path = os.path.join(third_party, "NeMo-VAD-onnx", name)
        if os.path.isfile(path):
            return path
    return None
