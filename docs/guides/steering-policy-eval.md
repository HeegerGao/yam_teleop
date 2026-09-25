# SteeringPolicyReal 本地推理与评估

入口：`scripts/steering_policy_eval.py`。在 `/home/gck/i2rt` 执行下面的命令。
机器人端使用已有 `.venv`（Python 3.11），模型端自动启动
`~/steering_policy_real/pi05_env/bin/python`（Python 3.12），退出时释放模型进程和显存。

## 模型选择

```bash
# 不发送动作的评估；仍会启动 follower、读取关节和相机并录制。
uv run --no-sync python scripts/steering_policy_eval.py --task cup_pingpong --model pi

# 真机执行：--task 和 --model 可独立切换。
uv run --no-sync python scripts/steering_policy_eval.py --task cup_pingpong --model steeract_enc --execute
uv run --no-sync python scripts/steering_policy_eval.py --task cup_pingpong --model steeract_dec --execute
uv run --no-sync python scripts/steering_policy_eval.py --task cup_pingpong --model drvla --execute
uv run --no-sync python scripts/steering_policy_eval.py --task cup_pingpong --model coast --execute
uv run --no-sync python scripts/steering_policy_eval.py --task cloth --model pi --execute
uv run --no-sync python scripts/steering_policy_eval.py --task push_cube --model steeract_enc --execute
uv run --no-sync python scripts/steering_policy_eval.py --task cloth --model drvla --execute
uv run --no-sync python scripts/steering_policy_eval.py --task cloth --model coast --execute
uv run --no-sync python scripts/steering_policy_eval.py --task push_cube --model drvla --execute
uv run --no-sync python scripts/steering_policy_eval.py --task push_cube --model coast --execute
```

任务：`cup_pingpong`、`cloth`、`push_cube`。方法对应如下：

| 参数 | 发布的方法 | 默认 checkpoint | hook |
| --- | --- | --- | --- |
| `pi` | 原始 π₀.₅ baseline | `<task>/020000/pretrained_model` | 无 |
| `steeract_enc` | D5 + Maturity encoder | `<task>/finetuned_ckpt/pretrained_model` | PaliGemma 14/15/16 |
| `steeract_dec` | D5 + Maturity decoder | `<task>/finetuned_ckpt/pretrained_model` | expert 14/15/16，10 个独立 step controller |
| `drvla` | DrVLA，固定 alpha=100 | `<task>/020000/pretrained_model` | PaliGemma 14/15/16 |
| `coast` | COAST（原请求中的 COSAT），beta=1 | `<task>/020000/pretrained_model` | expert 14/15/16 |

旧参数 `--steer` 等价于 `--model steeract_enc`。`--model-root` 可指定另一份完整资产目录；
`--checkpoint` 只覆盖基础权重路径，不替换该任务的 SAE、bank 或 selection。
不要跨任务混用。发布的 decoder 仅支持 10 个 denoising steps。

当前固定下载版本：`b42b04972a24d47b6266606050bd9ea1cdf56f0d`，来源为
[SteeringPolicyReal](https://huggingface.co/lzy001Yuki/SteeringPolicyReal/tree/b42b04972a24d47b6266606050bd9ea1cdf56f0d)，
参考 [PI05_STEERING.md](https://huggingface.co/lzy001Yuki/SteeringPolicyReal/blob/b42b04972a24d47b6266606050bd9ea1cdf56f0d/PI05_STEERING.md)
和 [STEERING.md](https://huggingface.co/lzy001Yuki/SteeringPolicyReal/blob/b42b04972a24d47b6266606050bd9ea1cdf56f0d/STEERING.md)。

2026-09-24 已补齐 cloth、push_cube 的 DrVLA / COAST，现在三个任务的五种方法共 **15 个组合**
均有本地资产。新增 baseline 各自使用该任务的 `020000/pretrained_model`、`sae/` 与
`drvla/selection.json` 或 `coast/selection.json`，prompt 分别为 `cloth`、`push_cube`。
资产缺失仍会在启动机器人前报错，不会换成 pi 或套用其他任务的资产。

新版本位于 `~/steering_policy_real/releases/b42b04972a24d47b6266606050bd9ea1cdf56f0d/`，
基础权重与旧版本相同；下载器新增了从已有 release 校验 SHA256 后复用大文件的能力。
旧版本 `d13ee888cd915e89d3a755960240efa8226da985` 保留，可通过 `--model-root` 显式使用。
本次仅补资产和升级默认资产路径，不改变动作限制、q/Esc、s/f 或 episode 保存目录规则。

## 轨迹保存与操作

每轮 episode 直接保存在 model 下面，按已有最大编号递增（包括按 q 后重新开始、切回同一模型）：

```text
~/yam_eval/steering_<task>/<model>/
  episode_0000/
    top.mp4
    left_wrist.mp4
    right_wrist.mp4
    low_dim.npz
    meta.json
  episode_0001/
  episode_0002/
```

`--save-root` 改根目录，`--run-name` 仅作为标签写入 `meta.json`，不再改变目录名称。
每个 task/model 独立编号，失败 episode 也占用原编号，重新启动不会覆盖已有轨迹。
旧的时间戳目录保留原状；新录制不再创建时间戳层。
`meta.json` 保存 task、model、实际 checkpoint、资产根目录、seed、运行参数、每次决策 telemetry、结果标签。
decoder 与 COAST 的 telemetry 保留一次决策全部 10 个 step。视频与关节轨迹沿用原评估格式。
失败标签会把 episode 改为 `episode_0000_failed`。

COAST 支持 `--coast-beta`（范围 0–1，默认仍是发布值 1.0）。
cup_pingpong 的一次启动失败记录 `coast/episode_0001` 显示：机器人已到准备姿态
（最大偏差约 0.021 rad），但首个预测动作偏差 1.138 rad，触发了 0.8 rad 的启动保护。
该观测的离线重放（seed 0/1/2、检查动作块前 10 步所有臂关节）结果为：

| COAST beta | 相对当前姿态的最大偏差范围 |
| --- | --- |
| 0（关闭干预） | 0.160–0.168 rad |
| 0.1 | 0.173–0.187 rad |
| 1（发布值） | 2.413–3.260 rad |

减小 beta 是改变 COAST 干预强度，不代表发布版 COAST 已修复或成功率提高。
`args.coast_beta`、`policy_info.coast_beta` 和 telemetry 中的 beta 会随轨迹保存。
不要通过放宽 `--max-start-error` 来消除该错误；较小 beta 的现场验证可先用：

```bash
uv run --no-sync python scripts/steering_policy_eval.py \
  --task cup_pingpong --model coast --coast-beta 0.1 --max-seconds 10
```

该命令不发送策略动作，但会启动 follower。确认观测与输出后再加 `--execute`。
离线检查脚本也支持相同的 `--coast-beta` 参数。以上复现使用保存的视频，有压缩损失，
且原失败运行未固定 seed，因此不应要求重放数值与原日志完全相等。

启动后 SPACE / Enter 开始。**q** 结束并保存当前轮，打开夹爪，经工作姿态回到折叠姿态，
然后自动移动到当前 task 的准备姿态，等待下一次 **Enter**。模型、相机和 follower 保持运行，
旧动作和 steering history / dose 清空；回位过程不记入评估轨迹。
默认回位后先在终端和预览窗口显示上一轮的结果标记提示：在**启动终端或预览窗口**按
**s（成功）/ f（失败）**，无需回车；标记保存后再按 **Enter** 开始下一轮。
尚未标记时按 Enter 不会启动。回位途中在上述窗口按下的 s / f 会保留并在回位后处理。
`--no-label-outcome` 可关闭必填标记步骤。全局按键监听仍只处理 q / Esc。
**Esc** 执行原先 q 的保存、释放夹爪、收臂和退出流程；回位中按 Esc 也会转入退出。
Ctrl-C 沿用原有紧急停止流程。`--max-seconds 10` 到时也结束当前轮并回位等待 Enter。
即使设置 `--no-wait-for-start`，q / 超时后仍必须按 Enter 才能再开始。
`--park-pose` / `--no-park-on-exit` 控制退出动作；q 的轮间回位始终先折叠再到 task 准备姿态。
已经在另一终端启动 follower 时，加 `--no-launch`。

push_cube 沿用已有控制约定：左臂执行，右臂保持演示初始姿态；
cloth 的起始前移及 cup_pingpong 的人工起始姿态也保留。
不指定 `--execute` 不发送策略动作，但真实 follower 仍会启动；完全不连接硬件的验证请用下面的离线命令。

## 完全离线验证

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python scripts/steering_policy_offline_check.py \
  --task cup_pingpong --model steeract_dec --episodes 0 --stride 10 --max-decisions 12
```

输入来自 `~/yam_data/<task>/episode_0000`（push_cube 对应 `push_cube_shovel_correct`）；
无需相机或机器人。每次启动把动作块、原始帧编号、误差和 telemetry 保存到
`~/yam_eval/offline/steering_<task>/<model>/episode_NNNN/`，编号自动递增。
输入演示的路径另存为 `meta.json` 中的 `source_episode`，输出编号不要求与输入编号相同。
不同 episode 开始时会重置 D5 history、maturity 和 dose。离线重放是推理链路检查，不是任务成功率评估。

2026-09-24 新增四个组合的验证：使用各任务 episode_0000，seed=0、stride=10，各运行 12 次决策。
每个输出均为有限值 `(12, 50, 14)`；DrVLA 每层调用且施加干预 12 次，COAST 每层 120 次，
从第一个决策开始生效。COAST 保留每次决策的全部 10 步 trace；DrVLA 发布 hook 只提供计数。

| 任务 / 模型 | 前 10 步对演示命令的臂关节 MAE（rad） | 首块前 10 步相对初始实测姿态的最大偏差（rad） |
| --- | ---: | ---: |
| cloth / drvla | 0.0267 | 0.0477 |
| cloth / coast | 0.7517 | 3.4545 |
| push_cube / drvla | 0.2565 | 2.1466 |
| push_cube / coast | 3.4525 | 13.2323 |

MAE 平均了 12 次决策、每次前 10 步、双臂 12 个关节；最大偏差仅统计实际执行侧
（cloth 双臂，push_cube 左臂），不含夹爪。这是离线原始输出，未经 eval 限幅，
也不是考虑实时推理延迟后的首条实际指令。cloth COAST、push_cube 两个 baseline 有明显偏差，
**配置/推理链路可用不代表动作可安全执行**；先离线检查和现场 dry-run，不要放宽启动保护来跑通。
保持发布的 DrVLA alpha=100、COAST beta=1，没有自动调整模型强度或现有安全限制。

四次验证分别保存在 `~/yam_eval/offline/steering_<task>/<model>/episode_0000/` 的
`actions.npz`、`meta.json`，其中记录了新版本的完整资产路径和 telemetry。
未启动 follower、相机或机械臂；19 项资产路由 / 目录 / 连续评估测试和 Ruff 检查通过。

此前验证：旧版本的 11 个有资产组合均在 RTX 5090 上，以各任务 episode_0000 的
前 12 次决策（stride=10、seed=0）完成离线推理，每次输出有限值 `(50, 14)`。
encoder / DrVLA 每层调用 12 次，decoder / COAST 每层调用 120 次；DrVLA 与 COAST
从第一次决策实际施加干预。该片段的 D5 gate 没有开启，因此不构成 D5 干预效果验证。

需注意实测异常：cup_pingpong 的 COAST 平均关节误差约 **0.739 rad**（pi 为 0.020 rad）；
push_cube 的发布版 pi 约 **0.257 rad**（steeract 两版约 0.027 rad）。这里比较的是前 10 步
动作与本地演示命令，不是成功率。COAST 保持发布的 beta=1，未自动改参数。
新版 `push_cube/020000` 权重与此前本机使用的 `push_cube_shovel_correct` 权重内容不同，
旧文件仍保留在 `~/steering_policy_real/push_cube/020000/pretrained_model`。
这两项应先检查 dry-run 轨迹，再决定是否实机执行；现有首动作检查、速度限制和关节限位均保留。

另外完成了 `pi → steeract_dec → pi` 的 MuJoCo follower 存档检查：每次 3 秒、90 帧，
三路视频均能解码，视频帧数与 `low_dim.npz` 一致，重复 run-name 也产生三个独立 session。
该检查使用黑色测试图像、`execute=False`，仅在仿真测试参数里放宽首动作距离；
没有连接或驱动真机，命令行默认距离限制仍为 0.8 rad。缺失资产拦截、目录隔离及 decoder
十步 telemetry 的 6 项测试通过，修改文件的 Ruff 检查通过。

## 下载与环境

```bash
~/steering_policy_real/pi05_env/bin/python scripts/steering_policy_download.py
```

下载到 `~/steering_policy_real/releases/<revision>/`，复用 SHA256 一致的旧大文件（硬链接），
支持重复运行和续传，跳过训练 optimizer state。旧目录和其中的自定义配置保留。
下载结果见版本目录下 `download_report.json`。大权重作为不可变资产使用；不要原地改写硬链接文件。

本机推理环境主要版本：Python 3.12、LeRobot commit
`71a11efe77f55e61f3ab2ce45b40da8cf626afa9`、torch `2.11.0+cu128`、
torchvision `0.26.0+cu128`、transformers `5.5.4`、numpy `2.2.6`、
safetensors `0.8.0`、numba `0.61.2`、llvmlite `0.44.0`。
numba 是远端 decoder hook 的额外依赖，已在独立推理环境安装。
不要把这些依赖安装到机器人环境中。

所有方法使用训练时的 checkpoint pre/postprocessor，传入 raw 14 维 state 与三路原分辨率 RGB，
不重复归一化。适配器复用发布者的 steering hook，避免其独立 infer 函数把 state 提前补成 32 维的差异。
发布代码缺失的 `version_b_descriptors` 分支沿用原适配器的 fail-closed 占位实现；D5 不使用该分支。
checkpoint 中新版 `min_range` 字段沿用已有兼容处理；远端原始文件不改写。
