# 修复 YAM 双臂 teleop 中 follower 卡死

面向**新搭的一套 YAM leader/follower teleop 系统**。症状是：teleop 到一半 follower 停住不动，
leader 还能自由拖动，两边进程都活着，终端里没有任何异常和栈。这个症状有 **4 个互相独立的成因**，
只修其中一个仍然会复发。参考实现见 [`examples/minimum_gello/minimum_gello.py`](../../examples/minimum_gello/minimum_gello.py)。

> 适用前提：follower 通过 `portal`（本仓库锁定 `portal==3.7.3`）暴露 RPC，leader 侧用
> `portal.Client` 发 setpoint；follower 侧由 `MotorChainRobot` / `DMChainCanInterface` 驱动 CAN 电机链。

---

## 0. 先做 5 分钟分诊

卡住的当下，**不要重启**，按顺序取这三样信息，能直接把成因缩到一个：

```bash
# 1) follower 的 RPC 还活着吗？（另开一个终端）
python - <<'PY'
import portal, time
c = portal.Client("127.0.0.1:6001")          # 换成你的 follower host:port
print("connected:", c.connect(timeout=2.0))
print("pos:", c.get_joint_pos().result(timeout=2.0))
time.sleep(0.5)
print("pos again:", c.get_joint_pos().result(timeout=2.0))
PY

# 2) leader 进程卡在哪一行
py-spy dump --pid <leader_pid>

# 3) follower 进程里还有几个线程、哪个死了
py-spy dump --pid <follower_pid>
ls /proc/<follower_pid>/task | wc -l
```

| 观察到 | 成因 |
| --- | --- |
| RPC 有应答，但两次 `get_joint_pos` 读到**完全相同**的值（臂被手推也不变） | [C](#c-电机链控制线程静默死亡) — 控制线程死了 |
| RPC 完全不应答 / `TimeoutError` | [C](#c-电机链控制线程静默死亡) 或进程真的挂了 |
| RPC 正常、数值在变，但 leader 的 `py-spy dump` 停在 `Future.result` / `Client.call` 的 `cond.wait` / `isconn.wait` | [A](#a-portal-的三条无限阻塞路径) — leader 侧被 portal 锁死 |
| 之前超时过、之后再也没恢复 | [B](#b-超时的-future-不会被回收连接不自愈) |
| 只在长时间跑之后偶发，`stats()` 里 `inflight` 一直是 16 | [A](#a-portal-的三条无限阻塞路径) + [D](#d-io-循环无节流) |

---

## A. portal 的三条无限阻塞路径

**为什么会卡：** `portal` 3.7.3 有三个默认永不超时、也不打日志的等待点。任何一个命中，
leader 的 IO 线程就永久静默停住，follower 从此收不到新的 setpoint —— 而 MIT 模式下电机会**保持最后一个
setpoint**，所以表现为"臂硬邦邦地冻在原地"，而不是掉电软下来。

| 位置 | 默认行为 |
| --- | --- |
| `Future.result(timeout=None)` | 无限等响应 |
| `Client.call`：`while len(self.futures) >= self.maxinflight:` (默认 16) | 在 `cond.wait(0.2)` 里**无限循环**，直到有 future 被回收 |
| `Client.call` → `socket.send()` → `require_connection(None)` → `isconn.wait(None)` | 连接断了就无限等重连 |
| `Client.close(timeout=None)` → `thread.join(None)` | 关闭本身也能卡死（影响 teardown） |

最阴的是第二条：如果你像大多数示例那样把命令写成 fire-and-forget

```python
self._client.command_joint_pos(joint_pos)   # ← 返回的 future 被丢弃，从不消费
```

那么这些 future 只有**收到响应时**才会被 `_recv` 从 `self.futures` 里 pop 掉。**丢一次响应，
未决计数就永久 +1**；攒够 16 个之后，下一次 `call` 直接卡死在 `cond.wait` 里，无声无息。

### 修法：把每一个 RPC 都变成有界的，并且消费掉每一个 future

```python
_RPC_TIMEOUT_S = 2.0
"""每个 follower RPC 的上限。相对亚毫秒级的 loopback 往返非常宽松：
这是一个存活性检查，不是延迟预算。"""


class ClientRobot(Robot):
    def __init__(self, port: int, host: str = "127.0.0.1", timeout: float = _RPC_TIMEOUT_S) -> None:
        self._addr = f"{host}:{port}"
        self._timeout = timeout
        self._client = portal.Client(self._addr)

    def _connected(self) -> None:
        """在 socket 断开时抛异常，而不是阻塞。
        `Client.call` 在建 future 之前就要发送，而那次发送等连接是完全无超时的；
        `connect` 是同一个等待的有界形式。"""
        if not self._client.connect(timeout=self._timeout):
            raise TimeoutError(f"not connected to {self._addr}")

    def get_joint_pos(self) -> np.ndarray:
        self._connected()
        return self._client.get_joint_pos().result(timeout=self._timeout)

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._connected()
        # 关键：命令类调用也要 .result()。等响应意味着同时在途最多 1 个请求，
        # maxinflight(16) 那个无超时的等待点永远够不到。
        self._client.command_joint_pos(joint_pos).result(timeout=self._timeout)

    def close(self) -> None:
        self._client.close(timeout=1.0)      # 连 close 都要有界
```

**规则：这个类里不允许出现任何一个不带 `timeout=` 的 portal 调用。** 包括 `connect`、`result`、`close`。

> 有人会担心"等响应会不会降低带宽"。不会：follower 那边 `command_joint_pos` 只是把 setpoint 塞进一个
> `maxsize=1` 的队列就返回，往返是亚毫秒级的，而控制 worker 产出 setpoint 的速率只有几百 Hz。

---

## B. 超时的 future 不会被回收，连接不自愈

**为什么会卡：** 一次 `result(timeout=...)` 超时后，那个 future **仍然留在 `Client.futures` 里**
（只有接收路径 `_recv` 会 pop 它）。所以一条"已经不再应答"的连接永远不会自己恢复，只会慢慢把
`maxinflight` 填满，最后退化成 [A](#a-portal-的三条无限阻塞路径)。

### 修法：连续失败到阈值就整个换掉 Client

```python
_RPC_ERRORS_BEFORE_RECONNECT = 20
"""连续失败多少次后重拨（约 4 s）。高到足以忍过一次抖动，
低到操作者手还握着 leader 时连接就已经被换掉。"""


def reconnect(self) -> None:
    """丢掉 socket 重拨，用一个全新的 Client。
    `close` 会把所有 pending future 置错并清空那个 dict,所以替换者从干净状态开始。"""
    try:
        self.close()
    except Exception as e:            # 卡死的 socket 不能阻止我们重拨
        logging.warning(f"[{self._addr}] closing the stale client failed: {e}")
    self._client = portal.Client(self._addr)
```

在 leader 的 IO 循环里计数：

```python
consecutive_errors = 0
while not stop_event.is_set():
    try:
        ...  # command_joint_pos + get_joint_pos
        consecutive_errors = 0
    except Exception as e:
        consecutive_errors += 1
        logging.error(f"[leader io] error ({consecutive_errors} in a row): {e}")
        if consecutive_errors >= _RPC_ERRORS_BEFORE_RECONNECT:
            logging.error("[leader io] follower unresponsive, reconnecting")
            client_robot.reconnect()
            consecutive_errors = 0
        time.sleep(0.1)
```

**同时必须做的两件事：**

1. **异常绝不能逃逸出 IO 循环。** 逃出去会走到 `finally: cleanup()`，那会在 leader 臂**带电做 bilateral PD**
   的时候 `proc.kill()` 掉控制 worker —— 一条通电的臂突然失控。
2. **启动时给 follower 一个重试窗口。** RPC 现在是有界的了，而 follower 可能还在标定夹爪，
   这比一个 timeout 长得多：

```python
_FOLLOWER_STARTUP_TIMEOUT_S = 60.0

deadline = time.time() + _FOLLOWER_STARTUP_TIMEOUT_S
while True:
    try:
        initial_follower_pos = client_robot.get_joint_pos()
        break
    except TimeoutError:
        if time.time() > deadline:
            raise TimeoutError(f"follower did not answer within {_FOLLOWER_STARTUP_TIMEOUT_S:.0f}s") from None
        logging.info("waiting for the follower to answer...")
```

---

## C. 电机链控制线程静默死亡

**这是 follower 侧真正的硬件成因，也是最容易漏的一个。**

CAN 帧**只**从 `DMChainCanInterface` 自己的控制循环线程里发出。`enable_auto_recovery=False`（默认）时，
**第一个电机 error 就让那个线程抛异常退出**，而承载它的进程照常活着：继续通过 RPC 返回**缓存的**
`joint_pos`、继续接收谁也不再消费的命令。于是：

- 外面看起来一切正常，RPC 全部成功；
- 但臂保持最后的 MIT setpoint，冻住；
- 日志里那条异常常常被 worker 进程吞掉，或者混在几万行 rate report 里。

### 修法：follower 侧开启自动恢复

```python
yam = get_yam_robot(
    channel=channel,
    gripper_type=gripper_type,
    sim=args.sim,
    # follower 出错不能 fail-fast：CAN 帧只从电机链自己的控制线程发出，
    # 而那个线程在第一个电机 error 上就会死掉 —— 而且是静默的，因为这个进程
    # 还在继续用缓存的 joint_pos 应答 RPC。臂于是保持最后的 MIT setpoint 冻在原地。
    # 让电机链去 clean + re-enable 出错的电机。
    enable_auto_recovery=True,
)
```

实现见 [`i2rt/motor_drivers/dm_driver.py:585`](../../i2rt/motor_drivers/dm_driver.py)：
它在控制循环里捕获 `Motor error detected`，调 `_try_recover_motors()` 成功就 `continue`，而不是
`self.running = False; raise`。

**注意分寸：**
- **只对 teleop 的 follower 臂开。** Flow Base 明确保持 `enable_auto_recovery=False`（fail-fast，
  底盘不自愈，见 `i2rt/flow_base/flow_base_controller.py:163`）。
- 自动恢复**掩盖**真实故障。恢复时会打 `WARNING`，把这条日志接到告警上；如果一台臂反复恢复，
  那是硬件/供电/线束问题，不要靠 auto-recovery 撑着跑。

### 补一个显式的存活检查（强烈建议）

auto-recovery 只覆盖"恢复得回来"的情况。再加一个"控制线程还活着吗"的哨兵，让静默死亡变成可见故障：
在 follower 的 polling worker 里每次更新共享位置时打一个时间戳，RPC 侧顺带把它返回；leader 发现
时间戳停止前进就报警/停机。参考实现里 `get_last_command` 就是同一类做法（返回最后一条命令及其
接收时刻），用于分辨"leader 没发"和"follower 没动"。

---

## D. IO 循环无节流

**为什么会卡：** 没有限速时 leader 的 IO 循环实测跑到 **~9.5 kHz**，也就是约 **19k RPC/s** 打到一个
单 worker 线程的 follower server 上 —— 比控制 worker 产出 setpoint 的 ~410 Hz 高 20 倍。
这对 portal 的流控压力大到"一次 stall 就永久 wedge"，直接把 [A](#a-portal-的三条无限阻塞路径) 的概率
从"偶发"变成"必然"。

### 修法一：给循环限速

```python
_WORKER_LOOP_PERIOD_S = 0.002   # 500 Hz 上限；YAM 控制线程本身约 250 Hz，不丢数据
...
        time.sleep(_WORKER_LOOP_PERIOD_S)   # 循环末尾
```

### 修法二：setpoint 流是 latest-wins，不是工作队列

每一跳都只保留最新的一条，丢掉积压：

```python
# leader 侧：排空到最新的一条命令
cmd = None
while True:
    try:
        cmd = cmd_queue.get_nowait()
    except queue.Empty:
        break
if cmd is not None:
    client_robot.command_joint_pos(cmd)

# follower 侧：maxsize=1 的队列，溢出就丢旧的
cmd_queue = portal.mp.Queue(maxsize=1)
```

否则控制 worker 入队比每次阻塞 RPC 出队更快，重放整个积压会让 follower 追一串**过期的**目标位置。

---

## E. 新系统上线检查清单

代码：

- [ ] `ClientRobot` 里**没有任何一个**不带 `timeout=` 的 portal 调用（`connect` / `result` / `close`）
- [ ] 命令类 RPC 也 `.result(timeout=...)`，不存在被丢弃的 future
- [ ] 连续失败 N 次 → `reconnect()`（新建 `portal.Client`，不是复用）
- [ ] IO 循环里的异常**不会**逃逸到会 kill 控制 worker 的 `finally`
- [ ] leader 启动时对 follower 首读有 60 s 重试窗口
- [ ] follower 的 `get_yam_robot(..., enable_auto_recovery=True)`
- [ ] IO 循环有 `time.sleep(_WORKER_LOOP_PERIOD_S)`
- [ ] 命令队列 latest-wins（leader 排空取最新 + follower `maxsize=1`）
- [ ] teardown 里没有任何一步可能 hang 在 gello 子进程（以及它们驱动的臂）被 kill 之前

硬件 / CAN（同样会让臂"看起来半死"，别误诊成软件）：

- [ ] 每个双通道 USB-CAN 适配器**只有一路 netdev 是 UP** —— 未接线的那一路会抢走部分电机应答。
      见 `scripts/fix_can_links.sh`，启动脚本应在发现兄弟通道 UP 时拒绝启动。
- [ ] 臂↔CAN 通道的映射只写在一个地方（本仓库是 `scripts/can_map.conf`），不要用 udev 持久名：
      udev 只按 serial 匹配，会绑到未接线的那一路。
- [ ] `python scripts/check_arms.py` 能看到 leader 6 个电机 / follower 7 个（含夹爪）

跑一遍验收：

- [ ] 连续 teleop ≥ 30 min 不冻结
- [ ] 中途 `kill -STOP` follower 进程 10 s 再 `-CONT`：leader 应打超时日志、随后自动重连恢复，
      而不是永久静默
- [ ] 中途拔掉 follower 的 CAN：应看到 auto-recovery WARNING 或明确的错误，不应是"RPC 一切正常但臂不动"

---

## 附：一个相关但不同的 portal 坑

`portal` 3.7.3 的 `ClientSocket` 收发线程只要有过流量就会陷入忙等（`client_socket.py` 里 `writing`
只在 `if self.sendq:` 分支内被清掉）。它**不会**让臂卡住，但会各占 ~65% CPU 并在 CUDA kernel launch
之间抢走 GIL —— 同一份策略推理，打桩掉 RPC 是 209 ms，开着 RPC 是 550 ms（真机上到过 1.45 s），
和发送频率无关。

**所以：任何要和 teleop RPC 同进程跑 GPU 推理的场景，把推理放到独立子进程**（参考
`scripts/box_folding_policy.py` 的 `PolicyProcess`：spawn + Pipe，一次只允许一个请求在途）。
排查手段：采样 `/proc/<pid>/task/*/stat` 的每线程 CPU，忙等线程一眼可见。
