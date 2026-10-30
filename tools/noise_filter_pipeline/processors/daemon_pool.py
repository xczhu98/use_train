"""
多 GPU 长驻 daemon 进程池公共基类。

_ApolloDaemonPool（D1）与 _AudioTagDaemonPool（C2）同构：每 GPU 一个长驻
daemon 子进程，ping 握手确认就绪，per-daemon 锁串行化访问，round-robin 分发，
daemon 死亡后原位重启，重启失败移出轮询，全部不可用时中止 stage。
本模块把该生命周期管理下沉为基类，协议差异通过钩子方法注入，避免两份拷贝
修复不同步（历史上 AudioTag 侧曾缺少 kill 后 wait 与槽位身份校验，
导致僵尸进程与并发误删健康 daemon）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import select
import shutil
import stat
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DaemonPoolExhaustedError(RuntimeError):
    """所有 daemon 不可用或连续失败次数超限。

    上层（_flush_pending）遇到此错误必须原样上抛中止 stage，
    而不是整批 reject 后静默继续。
    """


class _DaemonSlotIndex(int):
    """兼容整数下标，并携带 `_pick` 时绑定的稳定进程身份。"""

    def __new__(cls, value: int, expected_proc: subprocess.Popen):
        instance = super().__new__(cls, value)
        instance._expected_proc = expected_proc
        return instance


class BaseDaemonPool:
    """多 GPU 长驻 daemon 进程池的公共基类。

    子类需实现的协议钩子：
      - ``_build_cmd(gpu_id)``: daemon 启动参数（不含解释器与脚本路径）。
      - ``_ping_payload()`` / ``_check_pong(resp)``: ping 握手协议。
      - ``_stop_payload()``: 通知 daemon 退出的命令。
      - ``_validate_required_files()``: start 前的额外文件校验（可选，默认无）。
    """

    def __init__(
        self,
        python_path: str,
        daemon_script_path: str,
        checkpoint: str,
        gpu_ids: List[int],
        num_gpus: int,
        timeout: int,
        startup_timeout: int = 600,
        stage_name: str = "?",
    ):
        self.python_path = ""
        self._python_executable_identity: Tuple[int, int, int, int, int, str]
        self.bind_python_executable(python_path)
        self.daemon_script_path = daemon_script_path
        self.checkpoint = checkpoint
        self.gpu_ids = list(gpu_ids) if gpu_ids else list(range(max(1, num_gpus)))
        self.timeout = timeout
        # ping 握手（含模型加载）专用的启动超时，与正常推理响应超时分开：
        # 推理超时通常为小时级，直接用于 ping 会让死掉的启动流程挂住太久
        self.startup_timeout = max(1, int(startup_timeout))
        self.stage_name = stage_name

        self._daemon_procs: List[subprocess.Popen] = []
        self._daemon_locks: List[threading.Lock] = []
        # 与 _daemon_procs 平行的 GPU id 列表，供 daemon 死亡后原位重启使用
        self._daemon_gpu_ids: List[int] = []
        self._daemon_round_robin_lock = threading.Lock()
        self._daemon_round_robin_index = 0
        # 单调递增的启动序号，拼进 daemon 日志文件名：同一 gpu_id 起多个
        # daemon（小模型同卡多实例）或多次重启时日志互不覆盖。
        self._launch_seq = 0
        self._daemon_log_dir: Optional[str] = None
        self._daemon_started = False
        # _recv 按行切分时 chunk 末尾不完整行的残留字节（按 proc 分别保存），
        # 避免一次多读把下一条响应的字节吞掉导致后续响应错位
        self._recv_leftovers: Dict[subprocess.Popen, bytearray] = {}

    @staticmethod
    def _executable_identity(path: str) -> Tuple[int, int, int, int, int, str]:
        """读取 executable 的稳定身份；内容哈希防止元数据被恢复后绕过。"""
        try:
            before = os.stat(path)
            if not stat.S_ISREG(before.st_mode) or not os.access(path, os.X_OK):
                raise RuntimeError("不是可执行的普通文件")
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                opened = os.fstat(fh.fileno())
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise RuntimeError("读取 executable 时文件身份发生变化")
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
                after_open = os.fstat(fh.fileno())
            after_path = os.stat(path)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"无法绑定 daemon Python executable {path!r}: {exc}") from exc
        metadata = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if metadata(before) != metadata(opened) or metadata(before) != metadata(after_open):
            raise RuntimeError(f"读取 daemon Python executable 时内容发生变化: {path}")
        if metadata(before) != metadata(after_path):
            raise RuntimeError(f"读取 daemon Python executable 后路径身份发生变化: {path}")
        return (*metadata(before), digest.hexdigest())

    def bind_python_executable(self, python_path: str) -> None:
        """按当前 PATH 解析一次并冻结绝对 executable 与其身份。"""
        resolved = shutil.which(os.path.expanduser(os.path.expandvars(python_path)))
        if resolved is None:
            raise RuntimeError(f"daemon Python executable 不存在或不可执行: {python_path!r}")
        resolved = os.path.realpath(os.path.abspath(resolved))
        identity = self._executable_identity(resolved)
        self.python_path = resolved
        self._python_executable_identity = identity

    def _verify_python_executable(self) -> None:
        """每次 launch/restart 前复验冻结路径，禁止重新按 PATH 解析。"""
        current = self._executable_identity(self.python_path)
        if current != self._python_executable_identity:
            raise RuntimeError(
                f"{self.stage_name} daemon Python executable 身份已变化，拒绝启动: "
                f"{self.python_path}"
            )

    # ------------------------------------------------------------------
    # 子类协议钩子
    # ------------------------------------------------------------------
    def _build_cmd(self, gpu_id: int) -> List[str]:
        """daemon 启动参数（不含解释器与脚本路径）。"""
        raise NotImplementedError

    def _ping_payload(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _check_pong(self, resp: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def _stop_payload(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _validate_required_files(self) -> None:
        """start 前的额外文件校验（如 label_dict）。默认无。"""

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------
    def start(self, log_dir: str) -> None:
        """按需启动所有 GPU 对应的 daemon（每个 GPU 一个独立子进程）。"""
        if self._daemon_started:
            return
        if not self.daemon_script_path or not os.path.isfile(self.daemon_script_path):
            raise FileNotFoundError(
                f"{self.stage_name} daemon 模式需要 daemon_script_path: {self.daemon_script_path}"
            )
        if not self.checkpoint or not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(
                f"{self.stage_name} daemon 模式需要 checkpoint: {self.checkpoint}"
            )
        self._validate_required_files()

        self._daemon_log_dir = log_dir
        logger.info(
            "%s 启动 daemon，共 %d 个 GPU: %s",
            self.stage_name, len(self.gpu_ids), self.gpu_ids,
        )

        try:
            for gpu_id in self.gpu_ids:
                proc = self._launch_daemon(gpu_id, log_dir)
                self._daemon_procs.append(proc)
                self._daemon_locks.append(threading.Lock())
                self._daemon_gpu_ids.append(gpu_id)
        except Exception:
            # 任一 daemon 启动失败时回收已启动的 daemon，避免泄漏占 GPU
            self.stop()
            raise

        self._daemon_started = True

    def _launch_daemon(self, gpu_id: int, log_dir: str) -> subprocess.Popen:
        """启动单个 GPU 对应的 daemon 子进程并 ping 确认就绪。

        启动或 ping 失败时回收子进程并抛异常，由调用方决定中止或降级。
        """
        self._verify_python_executable()
        cmd = [self.python_path, self.daemon_script_path] + self._build_cmd(gpu_id)
        launch_seq = self._launch_seq
        self._launch_seq += 1
        log_path = os.path.join(
            log_dir, f"{self.stage_name.lower()}_daemon_gpu_{gpu_id}_{launch_seq}.log"
        )
        logger.info(
            "%s 启动 daemon gpu=%s: %s, log=%s",
            self.stage_name, gpu_id, " ".join(cmd), log_path,
        )
        stderr_fh = None
        try:
            # 每次启动使用带序号的独立日志文件：重启/同卡多实例的历史 stderr
            # 证据保留在各自的旧文件中，不会被后续实例覆盖
            stderr_fh = open(log_path, "a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_fh,
                text=True,
                bufsize=1,
                env=env,
            )
            stderr_fh.close()
            stderr_fh = None
        except Exception as e:
            if stderr_fh is not None:
                stderr_fh.close()
            raise RuntimeError(
                f"{self.stage_name} 启动 daemon (gpu={gpu_id}) 失败: {e}"
            ) from e

        # ping 确认就绪（含模型加载）：用独立的启动超时，而不是推理用的
        # 长超时，避免启动失败时挂住整个小时级 timeout
        try:
            self._send(proc, self._ping_payload())
            resp = self._recv(proc, timeout=self.startup_timeout)
        except Exception:
            # ping 失败时必须回收当前进程，避免泄漏占 GPU；kill 后 wait 回收僵尸
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass
            raise
        if not self._check_pong(resp):
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass
            raise RuntimeError(
                f"{self.stage_name} daemon (gpu={gpu_id}) ping 失败: {resp}"
            )
        return proc

    def _restart_daemon(
        self,
        idx: int,
        expected_proc: Optional[subprocess.Popen] = None,
    ) -> bool:
        """按进程身份重启故障 daemon；重启失败则将其移出轮询。

        ``idx`` 仅是调用方观测到故障时的槽位提示，列表删除会使它漂移；
        真正操作前必须在 round-robin 锁内按 ``expected_proc`` 重新定位。
        ``_pick`` 返回的下标携带等价稳定身份，供尚未显式传参的子类兼容使用。
        """
        token_proc = getattr(idx, "_expected_proc", None)
        if expected_proc is None:
            expected_proc = token_proc
        elif token_proc is not None and expected_proc is not token_proc:
            raise ValueError("daemon 槽位 token 与 expected_proc 身份不一致")
        if expected_proc is None:
            raise ValueError("重启 daemon 必须提供 expected_proc 或稳定槽位 token")

        with self._daemon_round_robin_lock:
            current_idx = next(
                (
                    slot_idx
                    for slot_idx, proc in enumerate(self._daemon_procs)
                    if proc is expected_proc
                ),
                None,
            )
            # 故障进程已被另一线程删除或替换，不得按旧 idx 操作当前进程。
            if current_idx is None:
                return bool(self._daemon_procs)
            gpu_id = self._daemon_gpu_ids[current_idx]
        try:
            expected_proc.kill()
            # kill 后必须 wait：回收僵尸进程，并等同 GPU 的 CUDA context 释放
            # 后再在同一 GPU 上拉起替代进程，避免同卡 CUDA 上下文竞争
            expected_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # 30s 仍未退出：CUDA context 可能尚未释放，二次强杀并记 error，
            # 否则紧接着在同 GPU 拉起的新 daemon 加载模型时可能 OOM
            logger.error(
                "%s daemon (gpu=%s) kill 后 30s 未退出，二次强杀",
                self.stage_name, gpu_id,
            )
            try:
                expected_proc.kill()
                expected_proc.wait(timeout=10)
            except Exception:
                pass
        except Exception:
            pass
        # 故障 daemon 的 _recv 残留字节一并丢弃，避免污染替代进程的响应流
        self._recv_leftovers.pop(expected_proc, None)
        logger.warning(
            "%s daemon (gpu=%s) 故障，尝试重启替代进程",
            self.stage_name, gpu_id,
        )
        try:
            new_proc = self._launch_daemon(gpu_id, self._daemon_log_dir or ".")
        except Exception as e:
            logger.error(
                "%s daemon (gpu=%s) 重启失败，将其移出轮询: %s",
                self.stage_name, gpu_id, e,
            )
            with self._daemon_round_robin_lock:
                current_idx = next(
                    (
                        slot_idx
                        for slot_idx, proc in enumerate(self._daemon_procs)
                        if proc is expected_proc
                    ),
                    None,
                )
                if current_idx is not None:
                    del self._daemon_procs[current_idx]
                    del self._daemon_locks[current_idx]
                    del self._daemon_gpu_ids[current_idx]
                    if self._daemon_procs:
                        self._daemon_round_robin_index %= len(self._daemon_procs)
                    else:
                        self._daemon_round_robin_index = 0
            return False
        with self._daemon_round_robin_lock:
            current_idx = next(
                (
                    slot_idx
                    for slot_idx, proc in enumerate(self._daemon_procs)
                    if proc is expected_proc
                ),
                None,
            )
            if current_idx is not None:
                self._daemon_procs[current_idx] = new_proc
                return True
            pool_available = bool(self._daemon_procs)
        # 槽位已被其他线程处理：本次启动的替代进程是多余的，
        # 必须回收，否则泄漏的 daemon 会一直占着 GPU
        logger.warning(
            "%s daemon (gpu=%s) 槽位已被并发线程处理，回收多余的替代进程",
            self.stage_name, gpu_id,
        )
        try:
            new_proc.kill()
            new_proc.wait(timeout=30)
        except Exception:
            pass
        return pool_available

    def stop(self) -> None:
        """通知所有 daemon 退出并等待其结束。"""
        procs = self._daemon_procs
        self._daemon_procs = []
        self._daemon_locks = []
        self._daemon_gpu_ids = []
        self._daemon_round_robin_index = 0
        self._daemon_started = False
        self._recv_leftovers.clear()
        for proc in procs:
            try:
                self._send(proc, self._stop_payload())
                proc.wait(timeout=10)
            except Exception as e:
                logger.warning("%s daemon 退出异常: %s", self.stage_name, e)
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 协议 I/O
    # ------------------------------------------------------------------
    def _send(self, proc: subprocess.Popen, cmd: Dict[str, Any]) -> None:
        """向指定 daemon 发送一条 JSON 命令。"""
        if proc is None or proc.stdin is None:
            raise RuntimeError(f"{self.stage_name} daemon 未启动")
        line = json.dumps(cmd, ensure_ascii=False)
        proc.stdin.write(line + "\n")
        proc.stdin.flush()

    def _recv(self, proc: subprocess.Popen, timeout: Optional[int] = None) -> Dict[str, Any]:
        """从指定 daemon 读取一条 JSON 响应，带超时避免永久阻塞。

        timeout 为 None 时使用 self.timeout（推理响应超时）；
        ping 握手等场景可传入更短的启动超时。

        直接读底层 fd（os.read）并按 b"\\n" 切分：select 只保证"有数据"，
        buffered readline 在 daemon 半行挂起时会越过超时预算永久阻塞；
        这里每轮读取都扣减剩余超时预算，预算耗尽即报错。
        """
        if proc is None or proc.stdout is None:
            raise RuntimeError(f"{self.stage_name} daemon 未启动")
        # 当前所有调用方均传 int 超时；浮点值会被 int() 向下截断放大
        # （0.5s -> 1s），如需亚秒精度改用 int(math.ceil(timeout))
        timeout = max(1, int(timeout if timeout is not None else self.timeout))
        # 绕过 TextIOWrapper 直接用底层 fd：wrapper 的预读缓冲对 select 不可见，
        # 混用两种读法会导致数据丢失或超时失效
        fd = proc.stdout.fileno()
        buf = self._recv_leftovers.setdefault(proc, bytearray())
        deadline = time.monotonic() + timeout
        while True:
            newline = buf.find(b"\n")
            if newline >= 0:
                line = bytes(buf[:newline]).decode("utf-8", errors="replace").strip()
                del buf[: newline + 1]
                if not buf:
                    self._recv_leftovers.pop(proc, None)
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"{self.stage_name} daemon 响应超时（>{timeout}s）"
                )
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise RuntimeError(
                    f"{self.stage_name} daemon 响应超时（>{timeout}s）"
                )
            chunk = os.read(fd, 65536)
            if not chunk:
                raise RuntimeError(f"{self.stage_name} daemon 意外结束输出")
            buf.extend(chunk)
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{self.stage_name} daemon 返回非法 JSON: {line!r} ({e})"
            )

    def _pick(self) -> Tuple[int, subprocess.Popen, threading.Lock]:
        """按 round-robin 选择一个 daemon。"""
        with self._daemon_round_robin_lock:
            if not self._daemon_procs:
                raise DaemonPoolExhaustedError(
                    f"{self.stage_name} 无可用 daemon，中止 stage"
                )
            idx = self._daemon_round_robin_index
            self._daemon_round_robin_index = (idx + 1) % len(self._daemon_procs)
            # 下标仍可用于日志/文件名，但额外绑定进程身份；即使低槽被删除，
            # 旧调用点只传 idx，重启也不会误杀漂移到该下标的健康进程。
            proc = self._daemon_procs[idx]
            return _DaemonSlotIndex(idx, proc), proc, self._daemon_locks[idx]
