# box_folding_dagger — HG-DAgger 数据说明与训练指南

`yam_data/box_folding_dagger/`：YAM 双臂 `box_folding` 任务的 **HG-DAgger 第一轮**数据，2026-09-03 采集，
49 条 episode（`episode_0002` – `episode_0050`，中间编号是当场丢弃的），每条一个目录，共 4.3 GB。

采集脚本：`i2rt/scripts/box_folding_dagger.py`。驱动策略是 `box_folding_policy/e2e/checkpoints/goal_step_150000.pt`
（55 条 demo 训到 150k 步的 e2e policy），每 0.8 s 重规划一次（`--replan-every 8`，脚本默认值）。
操作员从 leader 手柄按钮接管：policy 驱动某臂时其 leader PD 镜像该臂，按钮一按该臂切给人（leader 变自由、
follower 跟 leader），再按一次交回 policy。两臂独立切换。指令固定为
`fold the flat cardboard blank into a box and close the lid`。

## 1. 数据格式

和 `yam_data/box_folding`（`scripts/bimanual_teleop_record.py` 录的 demo）**完全一致**，30 fps，
外加几个 DAgger 专属键：

```
episode_NNNN/
    top.mp4  left_wrist.mp4  right_wrist.mp4   # 1920x1080 / 640x480 / 640x480，每 tick 一帧
    low_dim.npz                                # 行与视频帧 1:1
    meta.json                                  # 最后写；source = "dagger"，含每臂接管统计
```

`low_dim.npz`，每臂（`<side>` = left / right）：

| 键 | 含义 |
|---|---|
| `joint_pos_<side>` (T,7) | 实测 6 关节 + 归一化夹爪（0-1，1 = 开） |
| `eef_pos_<side>` (T,3) / `eef_quat_<side>` (T,4) | 实测关节 FK 的末端位姿，各臂自身基座系，四元数 **wxyz** |
| `gripper_<side>` (T,) | 实测夹爪开度 |
| `action_joint_pos_<side>` (T,7) | **本 tick 真正驱动该臂的命令**（绝对关节 + 夹爪）：人接管时 = leader 位姿 + trigger，policy 驱动时 = chunk 插值目标，无人驱动时 = 实测值 |
| `action_eef_pos/quat_<side>`、`action_eef_delta_<side>` | `action_joint_pos` 的 FK，及相对实测位姿的 dpos(3)+axis-angle(3) |
| `action_gripper_<side>` (T,) | 命令夹爪 |
| `engaged_<side>` (T,) bool | 本 tick 有人（policy 或人）在驱动 |
| **`control_mode_<side>`** (T,) int8 | **0 = hold（切换瞬间没人驱动），1 = policy，2 = human** |
| `policy_joint_pos_<side>` (T,7) | policy 本 tick 想要的目标，**即使被人覆盖也记录**（首个 plan 之前 NaN） |
| `human_joint_pos_<side>` (T,7) | leader 位姿 + trigger，不管谁在驱动都记录 |
| `command_joint_pos_<side>` (T,7) | 过关节限位 / 1.5 rad/s 限速后真正发给 follower 的值 |

全局：`plan_age`、`plan_latency`（当前 chunk 的年龄与推理耗时）、`t_mono`、`t_wall`、`cam_t_<role>`。

`meta.json["interventions"]`：每臂 `takeovers`（接管次数）、`human_frames`、`frames`。

## 2. 数据量（10 fps 时间轴，即训练用的 stride=3）

| | 步数 |
|---|---|
| 105 条 demo（`yam_data/box_folding`） | 78,639 |
| 49 条 DAgger 全部帧 | 36,617 |
| 其中 human 帧：左 / 右 / 任一臂 / 双臂 | 13,206 / 20,520 / 20,970 / 12,756 |
| hold 帧 | 428 |
| human 片段 | 251 段，中位 12 s（1.4–48 s） |

右臂 human 帧明显多于左臂：右臂是当前 policy 失败的主因。

对照 55 集的 `norm_stats.json`：state 只有 2.0% 的行超出旧 min/max，human action 只有 1.05% 的维度
超出 p01/p99 —— **续训可以沿用旧 norm_stats**。

## 3. 训练：只用 human 帧做 label，观测可以用全部帧

- `control_mode == 2`：`action_joint_pos_*` 是 leader 位姿 = 专家纠正，**这是 DAgger 的全部信号**。
- `control_mode == 1`：`action_joint_pos_*` 是 policy 自己的输出。拿它当 label 是自蒸馏，会把犯错前的
  行为固化下来。**不要进 loss**。
- 但 policy 帧的图像 / state 是 on-policy 状态分布，正是 DAgger 要的：4 帧历史可以跨进 policy 帧，
  只是 loss 不算它们。
- `control_mode == 0` 直接丢。
- **按臂做 mask**：同一时刻可能左 = human、右 = policy，mask 作用在 14 维 action 的各 7 维上，
  不是整帧。

### 3.1 接进 svl_project 的三处改动

**(a) `build_box_folding_policy_hdf5.py`**

- `eid = "box_folding_" + src_dir.name.split("_")[-1]` 会让 DAgger 的 `episode_0002` 与 demo 的
  `box_folding_0002` **同名覆盖**，改成 `box_folding_dagger_0002`。
- 多写一个 `action_mask (T, 14) bool`；demo 全 True，DAgger：

```python
cm_l = z["control_mode_left"][idx] == 2
cm_r = z["control_mode_right"][idx] == 2
mask = np.concatenate([np.repeat(cm_l[:, None], 7, 1), np.repeat(cm_r[:, None], 7, 1)], 1)
f.create_dataset("action_mask", data=mask)
```

- `--src-root` 指到 `yam_data/box_folding_dagger`，输出到同一个 `frames/`，再照常跑
  `extract_box_folding_radio_features.py`（只处理新 h5）。

**(b) `planning/dataset/box_folding_chunk_goal_radio_dataset.py`**

- 采样 chunk 起点 t 时只在 `action_mask[t].any()` 的位置采（否则 32 步里可能一帧 human 都没有）；
- `__getitem__` 多返回 `action_mask[t : t + 32]`，越界按现有 padding 方式补 False。

**(c) loss**：`(loss * mask).sum() / mask.sum().clamp(min=1)`。demo 样本 mask 全 1，行为不变。

### 3.2 怎么训

- **推荐续训**：从 `e2e/checkpoints/goal_step_150000.pt` `--resume`（它本来就是 300k 跑到一半的
  ckpt），数据 = 105 demo + 49 DAgger，跑到 300k。optimizer / schedule 直接接上，正好是"DAgger 第二轮"。
- **norm_stats 用原来的**（55 集版）：ckpt 的动作缩放绑死在这套 p01/p99 上；min-max 归一化对越界值线性
  外推，不需要重算。若从头训 300k，再用全集重算。
- **采样比例**：human 步只有 demo 的 ~27%，均匀采样贡献偏小。用 `WeightedRandomSampler` 把 DAgger
  episode 上采 2–3×，或 demo : dagger = 1 : 1 采。

### 3.3 迭代

训完 → `python scripts/box_folding_dagger.py --checkpoint <新 ckpt>`（norm_stats 同一个包）→
采下一轮 `--task box_folding_dagger_r2` → **聚合所有轮次**的 human 帧再训（经典 DAgger 每轮都聚合，
不是只用最新一轮）。`meta.json["interventions"]` 的 human_frames 占比逐轮下降就是进步的指标。

### 3.4 可选进阶

- Sirius 式加权：接管后头 1–2 s 的 human 帧是"从错误状态恢复"的样本，最稀缺，可以加权。
- `policy_joint_pos_*` 记录了被覆盖时 policy 想干什么：离线算 human 与 policy 的偏差分布，看错在哪个
  关节 / 阶段，决定下一轮采集重点。
- 接管后夹爪有 `--grip-sync-tol` 的对齐期（label 是 trigger，follower 夹爪暂时不动），只影响每段头几帧，
  可忽略。

## 4. 采集（复现或下一轮）

```bash
python scripts/check_arms.py                 # 四臂都在
python scripts/check_leader_triggers.py      # trigger 能从 1.0 扫到 0.0
python scripts/box_folding_dagger.py --task box_folding_dagger_r2 --checkpoint <ckpt>
```

| 输入 | 作用 |
|---|---|
| 空格 / 鼠标左键 | IDLE → 开始 episode（policy 接管、开始录）；RUNNING → 停止并保存 |
| r / 鼠标右键 | RUNNING → 丢弃；IDLE → 撤回上一条 |
| 手柄按钮（每臂） | 该臂在 policy ↔ human 间切换；IDLE 时也能用（普通 teleop，摆回起始姿态） |
| i | 键盘同时切换双臂 |
| h | IDLE 时归位：开夹爪，经 working pose 折叠 |
| q / Esc | 归位后退出 |

上传：`python scripts/upload_yam_data_hf.py --task box_folding_dagger_r2 --all`。
