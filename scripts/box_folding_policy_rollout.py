"""Closed-loop rollout of the box_folding flow policies on the bimanual YAM.

Brings up the two follower arms (no leaders -- the policy replaces the operator), opens the
three RealSense cameras, and runs the policy as a receding-horizon controller: an observation
every 100 ms, a fresh 32-step action chunk every ``--replan-every`` ticks, and interpolated
absolute joint commands streamed to both arms at ``--send-hz``.

    observe  10 Hz     ~5 ms  grab 3 frames, resize to 240x320, buffer them as uint8
    plan    0.4 Hz   ~105 ms  RADIO + prefix + 10 Euler steps, in a separate policy *process*
    send     30 Hz     <1 ms  interpolate the active chunk at (now - t0) / 0.1 s

``--policy-mode`` picks which of the two trained policies drives it, and with it the bundle
``--policy-root`` defaults to -- they were not released together, so the default differs:

    e2e   the goal-free policy. Cameras and proprioception, nothing above it.
          ``~/box_folding_policy/e2e``, i.e. the 55-episode 150k checkpoint. demo_full's newer
          105-episode 300k e2e checkpoint scores the same offline but asks for about half the
          joint velocity on the real arms; see box_folding_policy.DEFAULT_POLICY_ROOT for the
          measurement. ``--policy-root ~/box_folding_policy/demo_full`` runs it.
    goal  the hierarchical one, from ``~/box_folding_policy/demo_full`` (the only bundle with a
          goal checkpoint and the Wan LoRA). Before the loop starts, Wan2.2-TI2V-5B + that LoRA turns
          the current top view into 15 subgoal images, and the policy is conditioned on a sliding
          6-subgoal window that ``--goal-advance`` walks forward. The generator runs in a child
          process, started *before* the policy process so the two model loads overlap.

          The upper level then **replans online**: every ``--wan-replan-chunks`` executed action
          chunks (8 by default, ~19 s) Wan generates a fresh 15-subgoal set from the top view as
          it is *then*, and the policy is re-conditioned on it with its cursor back at 0. The
          generation runs in the same child while the control loop keeps streaming the current
          chunk, so the 10 Hz tick never slips; the costs are VRAM and plan time. The child now
          stays resident for the whole run instead of exiting after the first set, so every
          generation peaks at 31.4 GB of a 32 GB card, and the third of plans that overlap one
          take ~200 ms instead of ~105 ms -- both measured, see --wan-replan-chunks.
          ``--wan-replan-chunks 0`` restores the old one-shot behaviour -- which is what the LoRA
          was actually trained for, since it only ever saw first-frame conditioning.

``--continuation`` picks how a new chunk joins the one being executed (see
box_folding_continuation.py):

    none    independent samples, the old behaviour -- the arm can jump or switch modes at a replan.
    rtc     Real-Time Chunking: PiGDM-guided inpainting of the overlap with the previous chunk,
            frozen over the inference delay. One backward pass per Euler step.
    bid     Bidirectional Decoding: --bid-samples candidates, keep the one most coherent with the
            previous chunk and most like the strong policy / least like a weak early checkpoint.
    legato  Legato's per-step guidance schedule. Native only on a Legato-trained checkpoint; on the
            released ones it runs a guidance-only approximation and says so in the [policy] line.

All three need the previous chunk to overlap the new one, i.e. --replan-every well under 32; at the
default 24 only 8 steps overlap, so try them with --replan-every 8-16. The inference delay is
estimated from the measured plan ages (--continuation-delay-steps overrides it), and every switch
logs ``jump``: how far the command target moved at the instant the new chunk took over.

Both cost the same per plan (the goal prefix is 3638 tokens against 3614) and both default to
half precision for the RADIO tower and the DiT trunk, which halved a plan to ~105 ms; see
box_folding_policy.py for the measured breakdown.

The policy runs in its own process, not a thread: portal's client socket threads busy-spin while
they have traffic, and in-process that inflated a plan from ~210 ms to ~550 ms (and, on the real
robot, up to 1.45 s) by holding the GIL between CUDA launches. See PolicyProcess.

Planning runs off the control loop precisely because it is longer than one tick; the send loop
keeps streaming the previous chunk meanwhile. A chunk is never extrapolated past its horizon --
once it runs out the arms hold its last step, and only ``--max-plan-age`` of that ends the run.

**Safety.** ``--execute`` is required to actually move the arms; without it everything runs and
prints but no command is sent. Commands are clamped to the arm's URDF joint limits, the gripper
to 0-1, and slew-rate-limited by ``--max-joint-vel``. The run refuses to engage if the policy's
first target is more than ``--max-start-error`` from the measured pose (that means the policy
does not recognise the scene -- start from a pose like the demonstrations began in).

**Stopping.** ``q`` (or Esc) ends the run, from the terminal it was started in, from anywhere on
the desktop (pynput/X11, ``--no-global-keys`` to disable) or from the preview window -- see
QuitWatcher. That, and reaching ``--max-seconds``, runs the operator's stop sequence:

    1. both grippers open where they stand, so the workpiece is let go and not dragged
    2. both arms fold up: to the working pose, then down to the folded rest pose (--park-pose)
    3. teardown: cameras, policy process, follower clients, then the followers themselves

Step 2 ends folded rather than back where the run started, because a run does not necessarily
*start* folded -- and step 3 zeroes every motor torque, so an arm left in mid-air drops out of it.
It goes via the working pose because folding straight down from wherever the rollout stopped would
sweep the arm across the table. ``--park-pose start`` restores the old behaviour.

``--no-release-grippers`` / ``--no-park-on-exit`` drop step 1 / step 2. **Ctrl-C does neither**:
an interrupt is the operator saying stop now, not "make two more moves", so the arms simply hold
their last command. Same after an error.

Every run is recorded by default to ``<--save-root>/<--run-name>/`` (``~/yam_eval/eval_<stamp>/``)
in the recorder's own layout -- one high-quality H.264 mp4 per view at the camera's native rate
(1080p30 for the top camera), ``low_dim.npz`` at the 10 Hz policy tick, and ``meta.json`` written
last -- so an eval and a demonstration can be inspected with the same tools.
The npz carries the measured joints and EEF poses, the chunk target at each tick and the command
that was actually sent (they differ exactly where the slew limit bit), plus plan age and latency.
``--record-policy-input`` stores the 240x320 frames the policy consumed instead of the camera's
own resolution, at 10 Hz; ``--no-record`` writes nothing. It is *not* ~/yam_data: a rollout is not
a demo.

Prerequisites: the arms powered and on the CAN buses named in ``scripts/can_map.conf``, the three
cameras plugged in, and the policy bundle plus RADIO weights on disk (see box_folding_policy.py).

When this script launches the followers itself it also tears them down, and that sets every motor
torque to zero -- so the arms go limp at the end of *every* run, including a clean one. It folds
them up first (--no-park-on-exit to skip), which is why the folded pose is the default park: it
is where an unpowered arm rests, so there is nowhere left to fall. For repeated runs prefer
keeping the followers alive in their own terminal:

    scripts/run_policy_followers.sh                                    # terminal 1, stays up
    python scripts/box_folding_policy_rollout.py --no-launch --execute # terminal 2, repeatable

Usage:
    python scripts/box_folding_policy_rollout.py                       # dry run, nothing moves
    python scripts/box_folding_policy_rollout.py --execute             # e2e 150k, arms move
    python scripts/box_folding_policy_rollout.py --execute \
        --policy-root ~/box_folding_policy/demo_full                   # e2e 300k instead
    python scripts/box_folding_policy_rollout.py --policy-mode goal --execute       # hierarchical
    python scripts/box_folding_policy_rollout.py --policy-mode goal --execute \
        --wan-replan-chunks 4                           # replan twice as often
    python scripts/box_folding_policy_rollout.py --policy-mode goal --execute \
        --wan-replan-chunks 0                           # one-shot subgoals, the old behaviour
    python scripts/box_folding_policy_rollout.py --policy-mode goal --subgoal-dir ~/subgoals \
        --execute                            # reuse a fixed set, no Wan load and no replanning
    python scripts/box_folding_policy_rollout.py --execute --max-seconds 60 --replan-every 2
    python scripts/box_folding_policy_rollout.py --checkpoint-preset dagger-base-150k \
        --execute --replan-every 12 --continuation rtc
    python scripts/box_folding_policy_rollout.py --checkpoint-preset dagger-ft-250k \
        --execute --replan-every 12 --continuation rtc
    python scripts/box_folding_policy_rollout.py --execute --replan-every 12 --continuation rtc
    python scripts/box_folding_policy_rollout.py --execute --replan-every 12 --continuation bid
    python scripts/box_folding_policy_rollout.py --execute --replan-every 12 --continuation legato
    python scripts/box_folding_policy_rollout.py --sim --allow-missing-cameras  # plumbing only
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import portal
import tyro

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import bimanual_teleop_record as rec  # cameras, FK, gello launch/teardown
from box_folding_continuation import ContinuationConfig
from box_folding_policy import (
    CONTROL_DT,
    DEFAULT_TORCH_HOME,
    FOLDED_Q,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    IMAGE_HW,
    VIEW_ORDER,
    WORKING_Q,
    PolicyProcess,
    build_state,
    default_checkpoint,
    default_policy_root,
    resize_to_policy,
    split_action,
)
from box_folding_wan_subgoals import (
    DEFAULT_WAN_MODELS,
    WanSubgoalProcess,
    load_subgoals,
    save_subgoals,
)
from can_channels import channel_for

SIDES = ("left", "right")
_RPC_TIMEOUT_S = 2.0
_CHECKPOINT_PRESETS = {
    "dagger-base-150k": (
        "~/box_folding_policy/dagger_base_150k",
        "checkpoints/dagger_base_150k.pt",
    ),
    "dagger-ft-250k": (
        "~/box_folding_policy/dagger_ft_250k",
        "checkpoints/dagger_ft_250k.pt",
    ),
}


@dataclass
class Args:
    # --- policy ---
    policy_mode: Literal["e2e", "goal"] = "e2e"
    """e2e = the goal-free policy. goal = the hierarchical one: Wan2.2 generates 15 subgoals from
    the top view and the policy is conditioned on a 6-subgoal window that advances as the task
    progresses, with the whole set regenerated every --wan-replan-chunks chunks. The checkpoint is
    checked against this, so the two cannot be swapped."""
    policy_root: Optional[str] = None
    """The downloaded bundle (checkpoints, norm_stats, planning package). Default depends on
    --policy-mode: ~/box_folding_policy/e2e for e2e, ~/box_folding_policy/demo_full for goal --
    the two policies were not released together. See box_folding_policy.DEFAULT_POLICY_ROOT."""
    checkpoint: Optional[str] = None
    """Explicit checkpoint. Default: the newest one in <policy_root>/<policy_mode>/."""
    checkpoint_preset: Literal["default", "dagger-base-150k", "dagger-ft-250k"] = "default"
    """Named checkpoint bundle. default preserves --policy-mode/--policy-root/--checkpoint;
    dagger-base-150k selects the original 55-episode e2e model used as the DAgger base, and
    dagger-ft-250k selects the DAgger-fine-tuned e2e model. Each preset also selects its matching
    norm_stats.json; do not combine a preset with --policy-root or --checkpoint."""
    device: str = "cuda:0"
    torch_home: str = DEFAULT_TORCH_HOME
    """Torch Hub cache holding the c-radio_v3-h weights."""
    flow_steps: int = 10
    """Euler steps for the flow integration. 5/10/20 score the same on the replay check, and the
    cost is linear in the count -- 5 takes 30 ms off a plan."""
    radio_dtype: str = "float16"
    """float32 | float16 | bfloat16 for the frozen RADIO tower. fp16 is 3.6x faster (110 -> 31 ms
    per plan) and scores the same; training itself consumed fp16 features."""
    trunk_dtype: str = "float16"
    """float32 | float16 autocast for the DiT trunk (prefix + Euler steps): 97 -> 72 ms."""
    seed: Optional[int] = 0
    """Seed for the flow noise. Fixed by default: it makes a run reproducible and consecutive
    chunks agree slightly better in their overlap (0.0164 vs 0.0174 rad), at no accuracy cost."""

    # --- continuation between consecutive chunks ---
    continuation: Literal["none", "rtc", "bid", "legato"] = "none"
    """How a new chunk continues the one being executed. none = independent samples (the old
    behaviour); rtc = Real-Time Chunking guided inpainting; bid = Bidirectional Decoding best-of-N;
    legato = Legato per-step guidance (native only on a Legato-trained checkpoint, a guidance-only
    approximation otherwise). Only matters where chunks overlap -- pair it with a --replan-every
    well under the 32-step chunk. See box_folding_continuation.py."""
    continuation_delay_steps: Optional[int] = None
    """Inference delay d in 10 fps steps: how much of the new chunk the old one still executes
    before the new one lands, frozen (rtc/legato) to the old chunk. Default: estimated per plan as
    ceil(max of the last 5 measured plan ages / 100 ms) -- ~2 at the ~105 ms default plan."""
    rtc_beta: float = 5.0
    """--continuation rtc: clip on the guidance weight (the paper's default)."""
    bid_samples: int = 16
    """--continuation bid: candidates per plan, and as many again from the weak checkpoint. The
    denoise runs batched, so this costs far less than N plans, but it is not free -- check plan=."""
    bid_mode_size: int = 3
    """--continuation bid: K, the nearest positive / negative samples forward contrast averages."""
    bid_decay: float = 0.9
    """--continuation bid: rho, the backward-coherence weight rho**tau over the overlap."""
    bid_weak_checkpoint: str = "auto"
    """--continuation bid: weak policy for the negative samples. auto = the lowest-step checkpoint
    beside the strong one (the e2e bundle's goal_step_050000.pt; demo_full has none, so goal mode
    runs positive-only), none = positive-only, or a path to a checkpoint of the same mode."""
    legato_ramp: Optional[int] = None
    """--continuation legato: ramp length r after the frozen delay. Default 32 - offset - delay."""

    # --- subgoals (--policy-mode goal only) ---
    subgoal_dir: Optional[str] = None
    """Use these subgoal_NN.png instead of generating: a repeatable run, and no 20 GB Wan load.
    Generate a set offline with scripts/box_folding_wan_subgoals.py. Turns off online replanning
    -- a fixed set has no generator to re-run."""
    wan_replan_chunks: int = 8
    """Online replanning: re-run Wan every N executed action chunks, conditioned on the top view
    as it is *then*, and re-condition the policy on the fresh set. 0 = one generation for the
    whole episode (the old behaviour).

    The new set's cursor starts at 0, which is what a set generated from the current frame means:
    its first stage is the next thing to do, not the start of the task. At the default
    --replan-every 24 a chunk is 2.4 s, so 8 chunks is a generation every ~19 s against ~8 s to
    generate one at --wan-steps 40 -- it lands well inside the window, and the loop never waits
    for it: the request is fired from the control loop and the policy keeps planning against the
    current set until the new one is encoded.

    Measured over a 120 s run (5 generations, sim followers, RTX 5090, everything else default).
    The 10 Hz tick never slipped: median period 100.0 ms, worst 145 ms -- the one tick that
    encodes the fresh set and writes its PNGs. Plans got dearer, because a generation shares the
    GPU with them: 36% of plans overlapped one and cost ~200 ms against ~105 ms, dragging the
    median from 107 to 116 ms. Both are far under the 2.4 s replan interval and the 3.2 s chunk
    horizon, so nothing went stale.

    VRAM is the real limit. The child now holds its ~20 GB for the whole run instead of exiting
    after the first set, so every generation peaks at 31.4 GB of this card's 32.6 GB (one-shot
    peaks at 30.9 GB, but only once, while the two model loads overlap at startup). It fits with
    ~1 GB to spare and nothing else on the GPU; if something else is, run --wan-replan-chunks 0.

    Finally: the LoRA was trained on first-frame conditioning only, so a mid-episode conditioning
    frame is outside its distribution. This is the experiment, not a proven improvement."""
    wan_models: str = DEFAULT_WAN_MODELS
    """DiffSynth model base path holding Wan-AI/Wan2.2-TI2V-5B."""
    wan_steps: int = 40
    """Wan denoising steps per generation, and the only real handle on how long one takes.

    Measured on an idle 5090 (scripts/box_folding_wan_bench.py): 1.24 s fixed + 162 ms/step, so
    40 steps = 7.7 s and the denoiser is 84% of it. The fixed part is the T5 encode plus the VAE
    decode of all 61 frames, charged whatever the step count; tiled or untiled decoding makes no
    difference at 512x288. 162 ms/step is roughly what the hardware allows -- a 5 B DiT over 2304
    latent tokens is ~23 TFLOP a step, i.e. ~145 TFLOPS achieved against the card's ~210 TFLOPS
    bf16 peak -- so this is not a knob a faster kernel will move much.

    Fewer steps is a real saving: 8 steps = 2.6 s, 4 = 1.9 s. What it costs is smaller than the
    numbers suggest. Against the 40-step set at the same seed, 8 steps scores 22.8 dB PSNR and 16
    steps is no closer (22.6 dB) -- the schedules do not converge on the same pixels -- but side
    by side the 8-, 16- and 40-step sets stage the task identically and differ in texture and
    sharpness. 4 steps is where it becomes visible: the late subgoals soften. Nothing here has
    been tested on the arms, and the bundle's own finding that subgoal *content* barely moves the
    action chunk (reversed subgoals: 0.054 rad) suggests it would not show up there either."""
    wan_seed: int = 0
    goal_advance: Literal["similarity", "time"] = "similarity"
    """How the 6-subgoal window walks forward. similarity: cosine-match the live top view against
    the subgoal RADIO summaries (bounded to +2 stages a plan, never rewinds). time: one stage per
    --goal-stage-seconds, ignoring what the scene actually looks like."""
    goal_stage_seconds: float = 6.0
    """Seconds per stage for --goal-advance time. The demos average T/15 ~= 4-9 s."""

    # --- control ---
    execute: bool = False
    """Send commands to the arms. Off by default: the run is a dry run that only prints."""
    send_hz: float = 30.0
    """Command stream rate. The chunk is on a 10 fps timeline and is interpolated up to this."""
    replan_every: int = 24
    """Minimum ticks between plans, which is how much of each chunk is executed open-loop.

    The chunk is 32 steps = 3.2 s, so 24 (2.4 s) runs three quarters of it and leaves 0.8 s of
    headroom before the horizon runs out and the arms would start holding its last step. The
    trade-off is the whole point of the knob: a small value replans often but then only ever
    executes the first few steps of each chunk, which are the ones closest to the current pose
    and so the slowest -- the policy keeps re-deciding instead of following through. A large
    value lets the policy commit to its plan, at the price of acting on an observation up to
    2.4 s old, so it cannot react to anything that moves under it.

    Was 3 (0.3 s). Nothing else needs changing at 24: a plan is at most ~2.5 s old, still under
    the 3.2 s horizon and far under --max-plan-age."""
    max_seconds: float = 120.0
    """Hard stop for the rollout."""
    max_plan_age: float = 8.0
    """Abort if the active chunk gets this old (s) -- planning has died. Between the chunk's own
    3.2 s horizon and this, the arms simply hold the chunk's last step instead of stopping: an
    abort tears down the follower processes, which disables the motors and drops the arms."""
    max_joint_vel: float = 1.5
    """Slew limit per arm joint, rad/s, applied to the streamed command."""
    max_gripper_vel: float = 3.0
    """Slew limit for the normalised gripper command, 1/s."""
    max_start_error: float = 0.8
    """Refuse to engage if the first predicted target is farther than this (rad) from the
    measured pose. Raise it only if you know why the policy is asking for a big first move."""
    goto_start: bool = False
    """With --execute, slew to --start-pose-file before cameras and policy startup. Disabled for
    box folding; task-specific wrappers may enable it when they ship a recorded start pose."""
    start_pose_file: Optional[str] = None
    """JSON file containing joint_pos.left/right arrays for --goto-start."""
    start_speed_scale: float = 0.25
    """Fraction of --max-joint-vel / --max-gripper-vel used for the start-pose move."""
    start_seconds: float = 15.0
    """Time budget for the start-pose move."""
    release_grippers: bool = True
    """On a normal finish (q or --max-seconds), open both grippers before the arms move anywhere,
    so whatever is being held is let go on the spot rather than dragged back across the table.
    Skipped after Ctrl-C and after an error, for the same reason --park-on-exit is."""
    release_seconds: float = 3.0
    """Time budget for that release. At the default limits it takes ~0.7 s from fully closed."""
    park_on_exit: bool = True
    """On a normal finish (--max-seconds or q), walk both arms to --park-pose before the followers
    are torn down. Runs after --release-grippers and keeps the gripper wherever that left it.
    Skipped after Ctrl-C and after an error: an interrupt is the operator saying stop, not "make
    one more move"."""
    park_pose: Literal["folded", "working", "start"] = "folded"
    """Where --park-on-exit ends.

    folded   the folded rest pose, by way of the working pose (two legs -- straight down from
             wherever the rollout stopped would sweep the arm across the table). This is roughly
             where an unpowered arm rests, and teardown zeroes every motor torque, so an arm left
             anywhere else drops out of it. Use this unless you have a reason not to.
    working  stop at the working pose: clear of the table, still up.
    start    the pose the run began in -- the old behaviour. Only safe when the run itself
             started from a folded pose; starting mid-air means ending mid-air."""
    park_speed_scale: float = 0.5
    """Fraction of --max-joint-vel / --max-gripper-vel used for that return move. Half speed by
    default: nothing is driving the arm there and the operator is usually standing next to it."""
    park_seconds: float = 8.0
    """Time budget for *each leg* of the park (--park-pose folded has two). Scales with
    --park-speed-scale: at half speed the same distance takes twice as long, and a budget that
    runs out just leaves the arm part-way there. The longest leg is ~1.9 rad on joint 3, which
    needs ~2.5 s at the default 0.75 rad/s."""
    warmup_ticks: int = 0
    """Extra observation ticks before the first plan; the history needs 4 either way."""

    # --- recording ---
    record: bool = True
    """Save the run: one mp4 per view at native camera resolution/rate, plus 10 Hz low_dim.npz
    and meta.json. Same layout as bimanual_teleop_record.py writes, so the same tools read both."""
    save_root: str = "~/yam_eval"
    """Where runs land, as <save_root>/<run_name>/. Deliberately not the recorder's ~/yam_data:
    an eval rollout is not a demonstration and must never be uploaded as one."""
    run_name: Optional[str] = None
    """Directory name for this run. Default: eval_<YYYYmmdd>_<HHMMSS>."""
    record_policy_input: bool = False
    """Write the 240x320 frames the policy actually consumed instead of the full-resolution
    camera frames. Pixel-exact for debugging what the policy saw; worse to watch and intentionally
    limited to the policy's 10 Hz observation rate. The default raw-camera recording runs at each
    camera's native resolution and frame rate."""
    video_crf: int = 10
    """H.264 constant-rate-factor for raw-camera eval videos. 10 is visually near-lossless;
    smaller values increase quality and file size (0 is lossless, 51 is worst)."""
    video_preset: Literal["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"] = "veryfast"
    """libx264 speed/compression trade-off. veryfast comfortably keeps up with three live cameras
    on the eval workstation while CRF controls the image quality."""
    require_top_video_profile: bool = True
    """Refuse to start a raw-camera recording if the top camera did not open at the requested
    --top-width/--top-height/--top-fps. This prevents a USB/profile fallback from silently turning
    an eval into a lower-resolution or lower-frame-rate recording."""

    # --- robot ---
    arm: str = "yam"
    version: int = 1
    follower_gripper: str = "linear_4310"
    can_follower_left: str = channel_for("follower_left")
    """LEFT follower CAN netdev (from scripts/can_map.conf)."""
    can_follower_right: str = channel_for("follower_right")
    """RIGHT follower CAN netdev (from scripts/can_map.conf)."""
    port_left: int = 1235
    port_right: int = 1234
    launch: bool = True
    """Spawn the two follower processes. --no-launch attaches to already-running ones."""
    sim: bool = False
    """Followers run in MuJoCo. Plumbing test only -- the policy needs the real cameras."""

    # --- cameras (same fields and defaults as the recorder, so the rig comes up identically) ---
    top_serial: Optional[str] = None
    left_wrist_serial: Optional[str] = None
    right_wrist_serial: Optional[str] = None
    swap_wrists: bool = False
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30
    top_width: int = 1920
    top_height: int = 1080
    top_fps: int = 30
    allow_missing_cameras: bool = False
    """Continue with fewer than three cameras. The policy is blind on the missing view -- for
    smoke-testing the loop, never for a real rollout."""

    # --- display ---
    display: bool = True
    """Preview window with the policy-resolution views; q/Esc there stops the run."""
    global_keys: bool = True
    """Also listen for q/Esc anywhere on the desktop (pynput, needs X11), so the run can be
    stopped without focusing the terminal or the preview window. The terminal this run was
    started from always works and is not affected by this flag."""
    display_width: int = 1200
    log_every: int = 10
    """Print a status line every N ticks."""


@dataclass
class Plan:
    """One action chunk and the observation time it was predicted from."""

    chunk: np.ndarray  # [32, 14] absolute leader joint commands
    t0: float  # monotonic time of the observation that produced it
    latency_s: float

    def target(self, now: float) -> Tuple[np.ndarray, float]:
        """Linearly interpolated command for ``now``, plus how far into the chunk that is."""
        u = float(np.clip((now - self.t0) / CONTROL_DT, 0.0, self.chunk.shape[0] - 1))
        lo = int(np.floor(u))
        hi = min(lo + 1, self.chunk.shape[0] - 1)
        w = u - lo
        return (1.0 - w) * self.chunk[lo] + w * self.chunk[hi], u


class FollowerArm:
    """One follower server: read the measured joint pos, write absolute joint commands."""

    def __init__(self, name: str, port: int) -> None:
        self.name = name
        self._client = portal.Client(f"127.0.0.1:{port}")
        self.num_dofs = int(self._client.num_dofs().result(timeout=10.0))

    def joint_pos(self) -> np.ndarray:
        return np.asarray(self._client.get_joint_pos().result(timeout=_RPC_TIMEOUT_S), dtype=np.float64)

    def command(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(np.asarray(joint_pos, dtype=np.float64)).result(timeout=_RPC_TIMEOUT_S)

    def close(self) -> None:
        self._client.close(timeout=2.0)


class Ema:
    """Exponential moving average of a millisecond timing, for the status line."""

    def __init__(self, alpha: float = 0.2) -> None:
        self._alpha = float(alpha)
        self.value = 0.0

    def add(self, seconds: float) -> None:
        ms = seconds * 1e3
        self.value = ms if self.value == 0.0 else (1 - self._alpha) * self.value + self._alpha * ms


class NativeVideoRecorder:
    """Record every new camera frame without coupling video rate to the 10 Hz policy loop.

    One thread and one ffmpeg process are used per camera. The threads ask ``CameraRig`` only for
    frames newer than the last device timestamp, so the full-resolution copy is made once rather
    than on every poll. If scheduling misses a camera frame, its device timestamp determines how
    many frames belong on the fixed-rate timeline; duplicating the newest frame preserves the
    episode duration instead of making the video run fast.
    """

    def __init__(self, rig: rec.CameraRig, ep_dir: Path, *, crf: int, preset: str) -> None:
        self._rig = rig
        self._ep_dir = ep_dir
        self._crf = int(crf)
        self._preset = preset
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._processes: Dict[str, "subprocess.Popen[bytes]"] = {}
        self._errors: Dict[str, BaseException] = {}
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._closed = False

    def start(self) -> None:
        for role in self._rig.roles:
            cam = self._rig.cams_by_role[role]
            profile = cam.get("color")
            if profile is None:
                continue
            width, height, fps = (int(v) for v in profile)
            self._stats[role] = {
                "width": width,
                "height": height,
                "fps": fps,
                "codec": "libx264",
                "crf": self._crf,
                "preset": self._preset,
                "source_frames": 0,
                "encoded_frames": 0,
                "filled_frames": 0,
            }
            thread = threading.Thread(
                target=self._record_role,
                args=(role, width, height, fps),
                name=f"eval-video-{role}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
            print(f"[record] {role}.mp4: {width}x{height}@{fps} H.264 CRF {self._crf}")

    def _open_ffmpeg(self, role: str, width: int, height: int, fps: int) -> "subprocess.Popen[bytes]":
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            self._preset,
            "-crf",
            str(self._crf),
            "-pix_fmt",
            "yuv420p",
            str(self._ep_dir / f"{role}.mp4"),
        ]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._processes[role] = proc
        return proc

    def _record_role(self, role: str, width: int, height: int, fps: int) -> None:
        proc: Optional["subprocess.Popen[bytes]"] = None
        last_timestamp: Optional[float] = None
        first_timestamp: Optional[float] = None
        encoded = 0
        try:
            while not self._stop.is_set():
                item = self._rig.snapshot_role(role, after_timestamp=last_timestamp)
                if item is None:
                    time.sleep(0.001)
                    continue
                image, timestamp = item
                if image.shape[:2] != (height, width):
                    raise RuntimeError(
                        f"{role} changed resolution from {width}x{height} to "
                        f"{image.shape[1]}x{image.shape[0]}"
                    )
                if proc is None:
                    proc = self._open_ffmpeg(role, width, height, fps)
                    first_timestamp = timestamp
                assert proc.stdin is not None
                assert first_timestamp is not None
                desired_total = max(encoded + 1, round((timestamp - first_timestamp) * fps / 1000.0) + 1)
                repeats = min(desired_total - encoded, fps * 2)
                pixels = memoryview(np.ascontiguousarray(image)).cast("B")
                for _ in range(repeats):
                    proc.stdin.write(pixels)
                encoded += repeats
                stats = self._stats[role]
                stats["source_frames"] += 1
                stats["encoded_frames"] = encoded
                stats["filled_frames"] += repeats - 1
                last_timestamp = timestamp
        except BaseException as e:
            self._errors[role] = e
            self._stop.set()
        finally:
            if proc is not None:
                assert proc.stdin is not None
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    returncode = proc.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    returncode = proc.wait(timeout=2.0)
                stderr = b"" if proc.stderr is None else proc.stderr.read()
                if returncode and role not in self._errors:
                    detail = stderr.decode(errors="replace").strip()
                    self._errors[role] = RuntimeError(
                        f"ffmpeg for {role} exited with {returncode}" + (f": {detail}" if detail else "")
                    )

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=20.0)
        stuck = [thread.name for thread in self._threads if thread.is_alive()]
        if stuck:
            for proc in self._processes.values():
                if proc.poll() is None:
                    proc.kill()
            for thread in self._threads:
                thread.join(timeout=2.0)
            self._errors["shutdown"] = RuntimeError(f"video encoder threads did not stop: {stuck}")
        if self._errors:
            details = "; ".join(f"{role}: {error}" for role, error in self._errors.items())
            raise RuntimeError(f"native video recording failed ({details})")

        for role, stats in self._stats.items():
            print(
                f"[record] {role}.mp4 finalized: {stats['source_frames']} camera frames, "
                f"{stats['encoded_frames']} encoded"
                + (f" ({stats['filled_frames']} timeline fills)" if stats["filled_frames"] else "")
            )

    def raise_if_failed(self) -> None:
        if self._errors:
            details = "; ".join(f"{role}: {error}" for role, error in self._errors.items())
            raise RuntimeError(f"native video recording failed ({details})")

    def summary(self) -> Dict[str, Dict[str, Any]]:
        return {role: dict(stats) for role, stats in self._stats.items()}


class FrameHistory:
    """The last ``length`` ticks of camera frames, as the policy process wants them.

    Frames are kept as uint8 here rather than encoded on arrival: only the 4 frames a plan
    actually reads need RADIO features, and on the real robot a plan runs about once a second.
    Each tick gets an id so the policy process can reuse features for frames it already saw --
    including the repeated first frame, which is how the training set left-clamped episode starts.
    """

    def __init__(self, length: int) -> None:
        self._buf: Deque[Tuple[int, np.ndarray]] = deque(maxlen=int(length))
        self._length = int(length)
        self._next_id = 0

    def push(self, frames_rgb: Dict[str, np.ndarray]) -> None:
        stack = np.stack([frames_rgb[v] for v in VIEW_ORDER])  # [V,H,W,3] uint8
        self._next_id += 1
        for _ in range(self._length if not self._buf else 1):
            self._buf.append((self._next_id, stack))

    @property
    def ready(self) -> bool:
        return len(self._buf) == self._length

    def snapshot(self) -> Tuple[List[int], np.ndarray]:
        return [fid for fid, _ in self._buf], np.stack([img for _, img in self._buf])


class QuitWatcher:
    """``q`` (or Esc) from wherever the operator's hands are, not only the preview window.

    Three independent sources set one flag, and any of them is enough:

    * the **terminal** this run was started from, put in cbreak so a keypress arrives without
      Enter. This is the one that works over SSH and with ``--no-display``. ``ISIG`` is left
      alone, so Ctrl-C still interrupts.
    * a **global** pynput listener, so the key is picked up while the focus is on another window.
      Needs X11; missing or refused, it is simply one fewer source.
    * the **preview window**, via :meth:`feed` from the existing ``cv2.waitKey`` handling.

    All three are best-effort at construction; :attr:`sources` says which came up, and the caller
    warns if none did. The terminal mode is restored by :meth:`close`, which the rollout runs in
    its teardown -- leaving a shell without echo would be a nasty parting gift.
    """

    KEYS = ("q", "\x1b")

    def __init__(self, global_keys: bool = True) -> None:
        self._flag = threading.Event()
        self._stopping = threading.Event()
        self._fd: Optional[int] = None
        self._saved: Optional[Any] = None
        self._listener: Optional[Any] = None
        self.sources: List[str] = []
        self._start_terminal()
        if global_keys:
            self._start_global()

    def _start_terminal(self) -> None:
        try:
            if not sys.stdin.isatty():
                return
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # clears ECHO|ICANON only -- Ctrl-C keeps working
        except Exception as e:  # not a tty, no termios, a weird terminal: just skip this source
            logging.debug(f"terminal key source unavailable: {e}")
            self._fd, self._saved = None, None
            return
        threading.Thread(target=self._read_terminal, name="quit-stdin", daemon=True).start()
        self.sources.append("terminal")

    def _read_terminal(self) -> None:
        import select

        while not self._stopping.is_set():
            try:
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                if sys.stdin.read(1) in self.KEYS:
                    self._flag.set()
                    return
            except Exception:
                return

    def _start_global(self) -> None:
        try:
            from pynput import keyboard

            def on_press(key: Any) -> None:
                char = getattr(key, "char", None)
                if (char and char.lower() in self.KEYS) or key == keyboard.Key.esc:
                    self._flag.set()

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()
            self.sources.append("global")
        except Exception as e:
            logging.debug(f"global key source unavailable: {e}")
            self._listener = None

    def feed(self, key: int) -> None:
        """A key code from ``cv2.waitKey``."""
        if key in (ord("q"), 27):
            self._flag.set()

    @property
    def triggered(self) -> bool:
        return self._flag.is_set()

    def close(self) -> None:
        self._stopping.set()
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._fd is not None and self._saved is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._fd, self._saved = None, None


class SafetyLimiter:
    """Joint-limit clamp + slew-rate limit on the outgoing command of one arm."""

    def __init__(self, joint_range: np.ndarray, max_joint_vel: float, max_gripper_vel: float) -> None:
        self._range = np.asarray(joint_range, dtype=np.float64)
        self._max_joint_vel = float(max_joint_vel)
        self._max_gripper_vel = float(max_gripper_vel)
        self._lo = np.concatenate([joint_range[:, 0], [GRIPPER_CLOSED]])
        self._hi = np.concatenate([joint_range[:, 1], [GRIPPER_OPEN]])
        self._step = np.array([max_joint_vel] * joint_range.shape[0] + [max_gripper_vel], dtype=np.float64)
        self.n_dofs = int(self._lo.size)
        """Arm joints + the gripper -- the width of a command, with the gripper last."""
        self.last: Optional[np.ndarray] = None

    def scaled(self, factor: float) -> "SafetyLimiter":
        """A copy with the slew limit scaled, carrying over the last command it issued.

        Carrying ``last`` over is what keeps the handover continuous: the slower limiter starts
        from the command the arm is already tracking, not from a fresh reading.
        """
        clone = SafetyLimiter(self._range, self._max_joint_vel * factor, self._max_gripper_vel * factor)
        if self.last is not None:
            clone.last = self.last.copy()
        return clone

    def reset(self, measured: np.ndarray) -> None:
        self.last = np.asarray(measured, dtype=np.float64)[: self._lo.size].copy()

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        t = np.clip(np.asarray(target, dtype=np.float64)[: self._lo.size], self._lo, self._hi)
        if self.last is None:
            self.last = t.copy()
            return t
        delta = np.clip(t - self.last, -self._step * dt, self._step * dt)
        self.last = self.last + delta
        return self.last.copy()


def _joint_range(arm: str, version: int) -> np.ndarray:
    """``[n_arm_joints, 2]`` limits from the arm-only MJCF -- the same model the FK uses."""
    import mujoco

    from i2rt.robots.utils import ArmType

    model = mujoco.MjModel.from_xml_path(ArmType.from_string_name(arm).get_xml_path(version))
    return np.asarray(model.jnt_range[: model.nq].copy(), dtype=np.float64)


def _listening_pids(port: int) -> List[int]:
    """PIDs of this user's processes listening on ``port``, straight from /proc.

    No ``ss``/``lsof`` dependency: read the listening socket's inode out of /proc/net/tcp*, then
    find which process has it open. Processes owned by another user are invisible here, which is
    fine -- the followers always run as the caller.
    """
    inodes = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(path).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":  # 0A = TCP_LISTEN
                continue
            try:
                if int(fields[1].rsplit(":", 1)[1], 16) == int(port):
                    inodes.add(fields[9])
            except ValueError:
                continue
    if not inodes:
        return []
    pids: List[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for fd in (entry / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    pids.append(int(entry.name))
                    break
        except OSError:
            continue
    return pids


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").replace("\0", " ").strip()
    except OSError:
        return "<gone>"


def _describe_port(port: int) -> str:
    return " | ".join(f"pid {pid}: {_cmdline(pid)}" for pid in _listening_pids(port)) or "<unknown process>"


def _check_ports_free(ports: Dict[str, int]) -> None:
    """Refuse to launch onto a port something else already serves.

    Otherwise the follower this script spawns dies with ``Address already in use`` while the
    *stale* server keeps the port -- and since the readiness check is only "is the port
    connectable", the rollout then happily drives whatever that old process is attached to
    (a MuJoCo sim, or nothing at all) while the real arms sit unpowered.
    """
    busy = {side: port for side, port in ports.items() if _listening_pids(port)}
    if not busy:
        return
    detail = "\n".join(f"  {side} port {port} -- {_describe_port(port)}" for side, port in busy.items())
    raise SystemExit(
        "[error] follower port(s) already in use, so the followers this run launches would die on bind:\n"
        f"{detail}\n"
        "Kill the process(es) above, or attach to them instead with --no-launch (it checks that "
        "they match this run)."
    )


def _check_attached_followers(args: Args, ports: Dict[str, int]) -> None:
    """--no-launch: make sure the servers already on those ports are the ones we mean to drive."""
    channels = {"left": args.can_follower_left, "right": args.can_follower_right}
    for side, port in ports.items():
        pids = _listening_pids(port)
        if not pids:
            raise SystemExit(
                f"[error] nothing is listening on the {side} follower port {port}; start the followers "
                "(scripts/run_policy_followers.sh) or drop --no-launch"
            )
        line = _cmdline(pids[0])
        if "minimum_gello" not in line:
            raise SystemExit(f"[error] {side} port {port} is served by something else -- {line}")
        if ("--sim" in line.split()) != bool(args.sim):
            kind = "a --sim follower" if "--sim" in line.split() else "a real-hardware follower"
            raise SystemExit(
                f"[error] {side} port {port} is served by {kind}, which does not match this run "
                f"({'--sim' if args.sim else 'real hardware'}) -- {line}"
            )
        if not args.sim and channels[side] not in line.split():
            raise SystemExit(
                f"[error] {side} port {port} is served by a follower on a different CAN bus (expected "
                f"{channels[side]}) -- {line}"
            )
        print(f"[attach] {side} follower on port {port}: pid {pids[0]}")


def _launch_followers(args: Args, procs: List["subprocess.Popen[bytes]"], shutdown: rec.ShutdownRequest) -> None:
    """The follower half of the recorder's launch: two gello processes, no leaders."""
    ports = {"left": args.port_left, "right": args.port_right}
    _check_ports_free(ports)
    if not args.sim:
        for interface in (args.can_follower_left, args.can_follower_right):
            rec._check_can_interface(interface)
        print(f"[can] follower-left {args.can_follower_left}, follower-right {args.can_follower_right}")

    common = ["--arm", args.arm, "--version", str(args.version)]
    started: Dict[int, subprocess.Popen[bytes]] = {}
    for can, port in ((args.can_follower_right, args.port_right), (args.can_follower_left, args.port_left)):
        cmd = [*common, "--can_channel", can, "--gripper", args.follower_gripper, "--server_port", str(port)]
        if args.sim:
            cmd.append("--sim")
        started[port] = rec._spawn_gello(cmd)
        procs.append(started[port])
    for port, proc in started.items():
        rec._wait_for_port(port, proc=proc, shutdown=shutdown)
        # Belt and braces: the port being connectable only proves *someone* serves it.
        if proc.poll() is not None:
            raise RuntimeError(f"follower for port {port} exited with code {proc.returncode} right after binding")
        if proc.pid not in _listening_pids(port):
            raise RuntimeError(
                f"port {port} is served by another process ({_describe_port(port)}), not the follower "
                f"we launched (pid {proc.pid})"
            )
    print("[launch] follower servers up")


def _read_state(arms: Dict[str, FollowerArm], kin: rec.EEFKinematics) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """The 28-D policy state and the per-side measured joint vectors behind it."""
    pos = {side: arms[side].joint_pos() for side in SIDES}
    fk = {side: kin.fk(pos[side]) for side in SIDES}
    state = build_state(pos["left"], pos["right"], fk["left"][0], fk["left"][1], fk["right"][0], fk["right"][1])
    return state, pos


def _grab_frames(
    rig: rec.CameraRig, last: Dict[str, Tuple[np.ndarray, float]], allow_missing: bool
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, float]]]:
    """``(policy frames, raw frames)`` for one tick.

    Policy frames are 240x320 RGB, one per view; raw frames are the camera's own resolution in
    BGR with its device timestamp, which is what gets recorded. A view that has not delivered
    yet repeats its last frame.
    """
    snap = rig.snapshot()
    raw: Dict[str, Tuple[np.ndarray, float]] = {}
    for view in VIEW_ORDER:
        if view in snap:
            raw[view] = snap[view]
            last[view] = snap[view]
        elif view in last:
            raw[view] = last[view]
        elif allow_missing:
            raw[view] = (np.zeros((*IMAGE_HW, 3), np.uint8), 0.0)
        else:
            raise RuntimeError(f"camera '{view}' has delivered no frame")
    return {view: resize_to_policy(img, bgr=True) for view, (img, _) in raw.items()}, raw


def _raw_top_frame(rig: rec.CameraRig, *, allow_missing: bool) -> np.ndarray:
    """The top camera's newest frame as full-resolution RGB -- what Wan is conditioned on.

    Deliberately not the 240x320 policy frame: Wan wants 512x288, and downscaling to 320x240 and
    back up would throw away detail the generator can use.
    """
    snap = rig.snapshot()
    if "top" not in snap:
        if not allow_missing:
            raise RuntimeError("the top camera has delivered no frame; cannot condition Wan")
        print("[warn] no top frame -- conditioning Wan on a black image (plumbing test only)")
        return np.zeros((*IMAGE_HW, 3), np.uint8)
    return cv2.cvtColor(snap["top"][0], cv2.COLOR_BGR2RGB)


def _continuation_config(args: Args) -> ContinuationConfig:
    return ContinuationConfig(
        method=args.continuation,
        rtc_beta=args.rtc_beta,
        bid_samples=args.bid_samples,
        bid_mode_size=args.bid_mode_size,
        bid_decay=args.bid_decay,
        bid_weak_checkpoint=args.bid_weak_checkpoint,
        legato_ramp=args.legato_ramp,
    )


def _delay_steps(args: Args, plan_ages: Deque[float], horizon: int) -> int:
    """The inference delay d for the next plan, in chunk steps.

    A chunk becomes active ``age`` seconds after its observation, and from then on the executor
    reads it at ``age / 100 ms`` -- so its steps below ``ceil(age / 100 ms)`` were executed by the old
    chunk. RTC's rule is to estimate this conservatively from recent history, hence the max.
    """
    if args.continuation_delay_steps is not None:
        return int(np.clip(args.continuation_delay_steps, 0, horizon))
    age = max(plan_ages) if plan_ages else 0.15
    return int(np.clip(math.ceil(age / CONTROL_DT), 1, horizon))


def _condition_on_subgoals(policy: PolicyProcess, subgoals: np.ndarray) -> List[np.ndarray]:
    """Encode a subgoal set into the policy and hand back the policy-resolution frames.

    ``set_goals`` restarts the cursor at 0, which is exactly right for every set this script
    ships: each one is generated from the top view at that moment, so its first stage is the next
    thing to do rather than the start of the episode.
    """
    frames = [resize_to_policy(img, bgr=False) for img in subgoals]
    policy.set_goals(np.stack(frames))
    return frames


def _save_generation(run_dir: Path, index: int, subgoals: np.ndarray) -> Path:
    """Write one Wan generation's PNGs next to the recording.

    Generation 0 keeps the plain ``subgoals/`` name a one-shot run has always written; the
    replans land in ``subgoals_01/``, ``subgoals_02/``, ... Each directory is a complete set, so
    any of them reads back with ``load_subgoals`` and can be fed to ``--subgoal-dir``.
    """
    out = run_dir / ("subgoals" if index == 0 else f"subgoals_{index:02d}")
    save_subgoals(np.asarray(subgoals), out)
    return out


_WINDOW = "box_folding policy"
_window_size: Optional[Tuple[int, int]] = None
"""(w, h) the window was last sized to, so a changed layout resizes it instead of being clipped."""


def _preview(frames: Dict[str, np.ndarray], lines: List[str], width: int, goal: Optional[np.ndarray] = None) -> int:
    """Draw the policy-resolution views with a status overlay; returns the pressed key.

    These are the frames the **policy** consumes, not the camera's own: all three are squeezed to
    320x240, so the 16:9 top view looks horizontally compressed next to the 4:3 wrists. That
    squeeze is part of the training contract (see ``resize_to_policy``) and showing it is the
    point -- a swapped view or a BGR flip is visible here and nowhere else. The recorder's window
    is the one that shows each camera at its own aspect ratio.

    ``goal`` is the subgoal the window currently starts at, drawn as a fourth tile so the operator
    can see what the upper level is asking the policy to reach.
    """
    global _window_size

    tiles = [cv2.cvtColor(frames[v], cv2.COLOR_RGB2BGR) for v in VIEW_ORDER]
    labels = list(VIEW_ORDER)
    if goal is not None:
        tiles.append(cv2.cvtColor(goal, cv2.COLOR_RGB2BGR))
        labels.append("subgoal")
    strip = np.concatenate(tiles, axis=1)
    scale = width / strip.shape[1]
    strip = cv2.resize(strip, (width, int(strip.shape[0] * scale)))
    for i, view in enumerate(labels):
        x = int(i * width / len(labels)) + 8
        # A dark plate under the label: yellow-on-white is unreadable on a bright wrist frame.
        cv2.rectangle(strip, (x - 6, 4), (x + 12 * len(view), 30), (0, 0, 0), -1)
        cv2.putText(strip, view, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    panel = np.zeros((26 * len(lines) + 12, width, 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    canvas = np.concatenate([strip, panel], axis=0)
    size = (int(canvas.shape[1]), int(canvas.shape[0]))
    if size != _window_size:
        # Create the window explicitly. Letting imshow do it takes OpenCV's default flags, which
        # are WINDOW_AUTOSIZE | WINDOW_GUI_EXPANDED -- and this build has only the Qt backend
        # (GUI: QT5, GTK: NO), where "expanded" means a toolbar, a status bar and mouse-wheel
        # zoom/pan over the image. A stray scroll then leaves the view zoomed and panned, which
        # reads as misaligned, overlapping tiles, and the toolbar eats height the layout did not
        # budget for. GUI_NORMAL is the plain window: no toolbar, no zoom. NORMAL (rather than
        # AUTOSIZE) also lets the operator and the window manager resize it, so a canvas wider
        # than the screen is fitted instead of clipped.
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(_WINDOW, *size)
        _window_size = size
    cv2.imshow(_WINDOW, canvas)
    return cv2.waitKey(1) & 0xFF


def _tick_sample(
    state: np.ndarray,
    limiters: Dict[str, "SafetyLimiter"],
    target: Optional[np.ndarray],
    active: Optional["Plan"],
    now: float,
    engaged: bool,
    goal_cursor: int = -1,
    goal_gen: int = 0,
) -> Dict[str, Any]:
    """One row of low_dim.npz: what the policy saw, what it asked for, what was sent.

    Key names follow bimanual_teleop_record.py where the quantity is the same one, so a rollout
    and a demonstration can be plotted against each other without a translation table.
    """
    nan7 = np.full(7, np.nan)
    sample: Dict[str, Any] = {
        "joint_pos_left": state[0:7].copy(),
        "joint_pos_right": state[7:14].copy(),
        "eef_pos_left": state[14:17].copy(),
        "eef_quat_left": state[17:21].copy(),
        "eef_pos_right": state[21:24].copy(),
        "eef_quat_right": state[24:28].copy(),
        "engaged": bool(engaged),
        "plan_age": float(now - active.t0) if active is not None else np.nan,
        "plan_latency": float(active.latency_s) if active is not None else np.nan,
        # Which subgoal the 6-image window starts at, so a run can be replayed against the
        # subgoal PNGs saved next to it. -1 for the goal-free policy.
        "goal_cursor": int(goal_cursor),
        # Which Wan generation those PNGs are: goal_cursor indexes into subgoals/ at generation 0
        # and into subgoals_NN/ after each online replan. Always 0 without replanning.
        "goal_gen": int(goal_gen),
    }
    parts = split_action(target) if target is not None else (nan7, nan7)
    for side, part in zip(SIDES, parts, strict=True):
        # target_*: the chunk interpolated at this instant. command_*: what the limiter let
        # through and the follower actually received. They differ exactly where clamping bit.
        sample[f"target_joint_pos_{side}"] = np.asarray(part, dtype=np.float64)
        last = limiters[side].last
        sample[f"command_joint_pos_{side}"] = nan7 if last is None else last.copy()
    return sample


def _slew_to(
    arms: Dict[str, "FollowerArm"],
    limiters: Dict[str, "SafetyLimiter"],
    targets: Dict[str, np.ndarray],
    *,
    seconds: float,
    send_hz: float,
) -> bool:
    """Stream ``targets`` to both arms until they arrive or the budget runs out. True if arrived.

    The limiters are the caller's, so consecutive calls continue from the command already in
    flight instead of stepping discontinuously between phases.
    """
    dt = 1.0 / float(send_hz)
    deadline = time.monotonic() + float(seconds)
    previous = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        elapsed = dt if previous is None else min(now - previous, dt)
        previous = now
        done = True
        for side, arm in arms.items():
            cmd = limiters[side].step(targets[side], elapsed)
            arm.command(cmd)
            if float(np.max(np.abs(cmd - targets[side][: cmd.size]))) > 1e-3:
                done = False
        if done:
            return True
        time.sleep(dt)
    return False


def _load_start_pose(path: Path, arms: Dict[str, "FollowerArm"]) -> Optional[Dict[str, np.ndarray]]:
    """Load a recorded bimanual pose, or return None when the task has no recording yet."""
    if not path.is_file():
        print(f"[start] no recorded pose at {path} -- starting from the current pose")
        return None
    try:
        payload = json.loads(path.read_text())
        saved = payload["joint_pos"]
        targets = {side: np.asarray(saved[side], dtype=np.float64) for side in SIDES}
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise RuntimeError(f"invalid start pose file {path}: {e}") from e
    for side, arm in arms.items():
        if targets[side].shape != (arm.num_dofs,):
            raise RuntimeError(
                f"invalid start pose file {path}: {side} has {targets[side].size} joints, "
                f"follower has {arm.num_dofs}"
            )
    return targets


def _stop_sequence(
    arms: Dict[str, "FollowerArm"],
    limiters: Dict[str, "SafetyLimiter"],
    home: Dict[str, np.ndarray],
    *,
    release: bool,
    park: bool,
    park_pose: str,
    release_seconds: float,
    park_seconds: float,
    send_hz: float,
    speed_scale: float,
) -> None:
    """The end of a run the operator asked for: let go of the box, then fold up out of the way.

    The order matters. Grippers first, with the arms held still, because moving first would drag
    whatever is being held across the table. Then the arms walk to ``park_pose``:

        folded   (default) via WORKING_Q to FOLDED_Q. Two legs, not one, because folding straight
                 down from wherever the rollout stopped sweeps the arm across the table and
                 through the workpiece; from the working pose the fold comes down beside the base.
                 FOLDED_Q is roughly where an unpowered arm rests, which is the point: tearing the
                 followers down zeroes every motor torque, so an arm parked anywhere else -- in
                 mid-air over the table, say -- drops out of it.
        working  stop at WORKING_Q, clear of the table but still up.
        start    the pose the run began in. Only safe when the run itself started folded.

    One set of slowed limiters carries every leg, so the command never jumps at a handover, and
    each leg gets its own ``park_seconds`` budget. Everything runs at ``speed_scale`` of the
    rollout's slew limit: nothing is driving the arms here and the operator is usually standing
    next to them, so it should look unhurried.

    Both arms move together, as they did before this had legs. Joint 1 is 0 in both canonical
    poses, so each arm folds in its own sagittal plane and they do not reach across each other.
    """
    slow = {side: limiter.scaled(speed_scale) for side, limiter in limiters.items()}
    for side, limiter in slow.items():
        if limiter.last is None:  # never engaged: start from where the arm actually is
            limiter.reset(arms[side].joint_pos())

    if release:
        targets = {}
        for side, limiter in slow.items():
            hold = limiter.last.copy()
            hold[limiter.n_dofs - 1] = GRIPPER_OPEN
            targets[side] = hold
        print(f"[stop] opening both grippers over up to {release_seconds:.1f}s (arms hold still)")
        if not _slew_to(arms, slow, targets, seconds=release_seconds, send_hz=send_hz):
            print("[warn] the grippers did not reach fully open in time -- continuing to the park")

    if not park:
        return
    legs = {
        "folded": [("working pose", WORKING_Q), ("folded rest pose", FOLDED_Q)],
        "working": [("working pose", WORKING_Q)],
        "start": [("pose the run started in", None)],
    }[park_pose]
    for label, pose in legs:
        targets = {}
        for side, limiter in slow.items():
            n = limiter.n_dofs
            target = np.asarray(home[side] if pose is None else pose, dtype=np.float64)[:n].copy()
            # Leave the gripper wherever the release left it. Opening it here as a side effect of
            # the canonical poses would undo an explicit --no-release-grippers.
            target[n - 1] = GRIPPER_OPEN if release else limiter.last[n - 1]
            targets[side] = target
        print(f"[stop] to the {label} over up to {park_seconds:.1f}s ({speed_scale:g}x slew limit)")
        if not _slew_to(arms, slow, targets, seconds=park_seconds, send_hz=send_hz):
            print(f"[warn] ran out of time before reaching the {label} -- stopped part-way")
            return


def run(args: Args) -> None:
    procs: List["subprocess.Popen[bytes]"] = []
    shutdown = rec.ShutdownRequest(procs)
    rig: Optional[rec.CameraRig] = None
    arms: Dict[str, FollowerArm] = {}
    policy: Optional[PolicyProcess] = None
    wan: Optional[WanSubgoalProcess] = None
    writer: Optional[rec.EpisodeWriter] = None
    native_video: Optional[NativeVideoRecorder] = None
    keys: Optional[QuitWatcher] = None
    stop_reason = "startup"
    # Index of the subgoal set currently conditioning the policy: 0 is the one generated before
    # the loop, each online replan bumps it. Declared here so the meta.json in `finally` can read
    # it however early the run fell over.
    goal_gen = 0
    # Online replanning needs a generator to re-run, which --subgoal-dir does not have.
    replan_chunks = int(args.wan_replan_chunks) if args.policy_mode == "goal" and not args.subgoal_dir else 0

    try:
        ports = {"left": args.port_left, "right": args.port_right}
        if args.launch:
            _launch_followers(args, procs, shutdown)
        else:
            _check_attached_followers(args, ports)
        arms = {"left": FollowerArm("left", args.port_left), "right": FollowerArm("right", args.port_right)}
        print(f"[robot] followers connected ({', '.join(f'{s}:{a.num_dofs} dof' for s, a in arms.items())})")

        kin = rec.EEFKinematics(args.arm, args.version)
        limits = _joint_range(args.arm, args.version)
        limiters = {s: SafetyLimiter(limits, args.max_joint_vel, args.max_gripper_vel) for s in SIDES}

        home: Dict[str, np.ndarray] = {side: arms[side].joint_pos() for side in SIDES}
        for side in SIDES:
            limiters[side].reset(home[side])
        if args.execute and args.goto_start:
            if args.start_pose_file is None:
                raise RuntimeError("--goto-start requires --start-pose-file")
            start_path = Path(args.start_pose_file).expanduser().resolve()
            targets = _load_start_pose(start_path, arms)
            if targets is not None:
                print(
                    f"[start] slewing to {start_path} over up to {args.start_seconds:.1f}s "
                    f"({args.start_speed_scale:g}x slew limit)"
                )
                slow = {side: limiter.scaled(args.start_speed_scale) for side, limiter in limiters.items()}
                if not _slew_to(arms, slow, targets, seconds=args.start_seconds, send_hz=args.send_hz):
                    print("[warn] did not reach the recorded start pose in time -- continuing part-way")
                for side in SIDES:
                    limiters[side].last = slow[side].last.copy()

        rig = rec._open_camera_rig(args)
        deadline = time.monotonic() + 10.0
        while not rig.all_fresh() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not rig.all_fresh() and not args.allow_missing_cameras:
            raise RuntimeError("cameras did not deliver frames (check USB bandwidth and cables)")
        if args.record and not args.record_policy_input and args.require_top_video_profile and "top" in rig.roles:
            actual = tuple(int(v) for v in rig.cams_by_role["top"]["color"])
            requested = (args.top_width, args.top_height, args.top_fps)
            if actual != requested:
                raise RuntimeError(
                    f"top camera opened at {actual[0]}x{actual[1]}@{actual[2]}, not the requested "
                    f"{requested[0]}x{requested[1]}@{requested[2]}; refusing a reduced-quality eval video. "
                    "Fix the camera/USB profile or pass --no-require-top-video-profile to accept the fallback."
                )
            counts_before = rig.frame_counts()
            rate_started = time.monotonic()
            time.sleep(1.0)
            elapsed = time.monotonic() - rate_started
            top_rate = (rig.frame_counts()["top"] - counts_before["top"]) / elapsed
            if top_rate < args.top_fps * 0.9:
                raise RuntimeError(
                    f"top camera is delivering only {top_rate:.1f}/{args.top_fps} fps; refusing a choppy eval "
                    "video. Move it to an uncongested USB3 port, or pass "
                    "--no-require-top-video-profile to continue anyway."
                )
            print(
                f"[record] verified top camera: {actual[0]}x{actual[1]}@{actual[2]}, "
                f"delivering {top_rate:.1f} fps"
            )

        # The upper level first, so its ~60 s model load overlaps the policy's ~30 s one instead
        # of following it. The child exits as soon as it hands the subgoals over, giving its
        # ~20 GB of VRAM back before the control loop starts.
        subgoals: Optional[np.ndarray] = None
        if args.policy_mode == "goal" and args.subgoal_dir:
            subgoals = load_subgoals(args.subgoal_dir)
            print(f"[wan] {len(subgoals)} subgoals loaded from {args.subgoal_dir} (no generation)")
            if args.wan_replan_chunks:
                print("[wan] --subgoal-dir: online replanning is off, this fixed set drives the whole run")
        elif args.policy_mode == "goal":
            wan = WanSubgoalProcess(
                policy_root=args.policy_root,
                models=args.wan_models,
                steps=args.wan_steps,
                seed=args.wan_seed,
                # Replanning reuses the loaded pipeline: a reload would cost ~60 s against the
                # ~7 s the generation itself takes, so the child holds its ~20 GB for the run.
                keep_alive=bool(replan_chunks),
            )
            wan.start(_raw_top_frame(rig, allow_missing=args.allow_missing_cameras or args.sim))
            print(f"[wan] generating 15 subgoals from the current top view ({args.wan_steps} steps) ...")

        print("[policy] starting the policy process (loads the checkpoint and RADIO) ...")
        policy = PolicyProcess(
            policy_root=args.policy_root,
            checkpoint=args.checkpoint,
            mode=args.policy_mode,
            device=args.device,
            flow_steps=args.flow_steps,
            radio_dtype=args.radio_dtype,
            trunk_dtype=args.trunk_dtype,
            goal_advance=args.goal_advance,
            goal_stage_seconds=args.goal_stage_seconds,
            torch_home=args.torch_home,
            seed=args.seed,
            continuation=_continuation_config(args),
        )
        info = policy.start()
        history = FrameHistory(int(info["history_len"]))
        print(f"[policy] {policy.description}")
        print(f"[policy] instruction: {info['instruction']!r}")

        goal_frames: List[np.ndarray] = []
        if args.policy_mode == "goal":
            if subgoals is None:
                subgoals = wan.result()
                print(f"[wan] {len(subgoals)} subgoals generated in {wan.seconds:.1f}s")
                if not replan_chunks:
                    wan = None  # the child has already exited and given its VRAM back
            goal_frames = _condition_on_subgoals(policy, subgoals)
            print(f"[wan] policy conditioned on a {info['goal_sequence_len']}-subgoal window")
            if replan_chunks:
                print(
                    f"[wan] online replanning: a fresh set every {replan_chunks} action chunks "
                    f"(~{replan_chunks * args.replan_every * CONTROL_DT:.0f}s); the generator stays "
                    "loaded, so it holds ~20 GB for the whole run"
                )
        print(
            f"[mode] {'EXECUTE -- the arms will move' if args.execute else 'DRY RUN -- no commands sent'}"
            f" for up to {args.max_seconds:.0f}s (--max-seconds), then the run ends"
        )

        keys = QuitWatcher(global_keys=args.global_keys)
        where = {"terminal": "this terminal", "global": "any window"}
        sources = [where[s] for s in keys.sources] + (["the preview window"] if args.display else [])
        if sources:
            steps = []
            if args.release_grippers:
                steps.append("open both grippers")
            if args.park_on_exit:
                steps.append(
                    {"folded": "fold up", "working": "go to the working pose", "start": "return to the start pose"}[
                        args.park_pose
                    ]
                )
            steps.append("disconnect")
            print(f"[keys] press q (or Esc) in {' / '.join(sources)} to stop: {', '.join(steps)}")
        else:
            print(
                "[warn] no key source: stdin is not a terminal, pynput is unavailable and --no-display -- "
                "only Ctrl-C and --max-seconds can end this run, and Ctrl-C leaves the arms holding"
            )

        last_frames: Dict[str, Tuple[np.ndarray, float]] = {}
        # Named unconditionally -- only EpisodeWriter creates it -- so a replanned subgoal set has
        # somewhere to go without re-deriving the path in the control loop.
        run_dir = Path(args.save_root).expanduser() / (
            args.run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if args.record:
            writer = rec.EpisodeWriter(run_dir, fps=round(1.0 / CONTROL_DT))
            print(f"[record] {run_dir}")
            if not args.record_policy_input:
                native_video = NativeVideoRecorder(
                    rig,
                    run_dir,
                    crf=args.video_crf,
                    preset=args.video_preset,
                )
                native_video.start()
            if goal_frames:
                # The subgoals belong to the run: without them the recorded goal_cursor column
                # says which subgoal was active but not what it looked like.
                print(f"[record] subgoals -> {_save_generation(run_dir, 0, subgoals)}")
        send_dt = 1.0 / float(args.send_hz)
        start = time.monotonic()
        next_tick = start
        next_send = start
        tick = 0
        engaged = False
        active: Optional[Plan] = None
        stop_reason = "max-seconds"
        # Action chunks executed since the last generation was *asked for* (not since the last one
        # landed), so the request cadence stays even when a generation overruns its window.
        chunks_since_wan = 0
        fresh_subgoals: Optional[np.ndarray] = None
        last_state = np.zeros(28, np.float32)
        last_stale_warn = 0.0
        last_send: Optional[float] = None
        timing = {k: Ema() for k in ("grab", "state", "send")}
        # Observation-to-arrival age of recent plans, for the continuation delay estimate, and how
        # far the command target jumped at each chunk switch -- the thing continuation should shrink.
        plan_ages: Deque[float] = deque(maxlen=5)
        switch_jumps: List[float] = []

        while True:
            shutdown.raise_if_requested()
            # Checked every iteration, not once a tick: the loop wakes at least every 20 ms, so
            # the key is acted on within that rather than waiting for the next 100 ms observation.
            if keys.triggered:
                stop_reason = "operator quit (q)"
                break
            now = time.monotonic()
            if now - start > args.max_seconds:
                break

            # ---- observation tick (10 Hz) -------------------------------------------------
            if now >= next_tick:
                if native_video is not None:
                    native_video.raise_if_failed()
                t_obs = next_tick
                next_tick += CONTROL_DT
                mark = time.perf_counter()
                frames, raw_frames = _grab_frames(rig, last_frames, args.allow_missing_cameras or args.sim)
                timing["grab"].add(time.perf_counter() - mark)
                history.push(frames)
                tick += 1

                # Start the next plan as soon as the policy process is free and the interval has
                # passed, instead of only on every Nth tick: on the real robot a plan can take
                # longer than the interval, and waiting for the next multiple then added a whole
                # extra period of staleness on top of the overrun.
                due = t_obs - policy.last_request_t0 >= args.replan_every * CONTROL_DT
                planning = history.ready and not policy.busy and tick > args.warmup_ticks and (active is None or due)
                # Without recording the state is read only when a plan needs it -- the policy
                # conditions on the newest state alone, and every RPC costs control-loop time.
                # Recording wants a row per tick, so then it is read every tick.
                if planning or (writer is not None and engaged):
                    mark = time.perf_counter()
                    last_state, _ = _read_state(arms, kin)
                    timing["state"].add(time.perf_counter() - mark)
                if planning:
                    ids, pixels = history.snapshot()
                    prev_offset = 0 if active is None else round((t_obs - active.t0) / CONTROL_DT)
                    policy.request(
                        t_obs,
                        pixels,
                        last_state,
                        ids,
                        prev_chunk=None if active is None else active.chunk,
                        prev_offset=prev_offset,
                        delay_steps=_delay_steps(args, plan_ages, int(info["action_chunk_len"])),
                    )

                if writer is not None:
                    target_now = active.target(now)[0] if active is not None else None
                    cursor = policy.goal_cursor if args.policy_mode == "goal" else -1
                    sample = _tick_sample(last_state, limiters, target_now, active, now, engaged, cursor, goal_gen)
                    if args.record_policy_input:
                        shown = {
                            view: (cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.0) for view, img in frames.items()
                        }
                        writer.add(sample, shown)
                    else:
                        # Full-rate videos are written by NativeVideoRecorder. Keep the camera
                        # timestamps sampled on the policy timeline in low_dim.npz, as before.
                        sample.update({f"cam_t_{view}": timestamp for view, (_, timestamp) in raw_frames.items()})
                        writer.add(sample, {})

                if args.display:
                    plan_age = "--" if active is None else f"{now - active.t0:.2f}s"
                    goal = f"  subgoal {policy.goal_cursor + 1}/{policy.n_goals}" if args.policy_mode == "goal" else ""
                    if goal and replan_chunks:
                        goal += f" gen {goal_gen} ({chunks_since_wan}/{replan_chunks})"
                    lines = [
                        f"tick {tick}  t {now - start:5.1f}s  plans {policy.n_plans}  plan age {plan_age}"
                        f"  {'EXECUTE' if args.execute else 'DRY RUN'}{goal}"
                    ]
                    for side in SIDES:
                        cmd = limiters[side].last
                        shown = "--" if cmd is None else np.array2string(cmd, precision=2, suppress_small=True)
                        lines.append(f"cmd {side[0].upper()} {shown}")
                    shown_goal = goal_frames[min(policy.goal_cursor, len(goal_frames) - 1)] if goal_frames else None
                    keys.feed(_preview(frames, lines, args.display_width, shown_goal))

                if tick % max(1, args.log_every) == 0:
                    lat = f"{active.latency_s * 1e3:.0f}" if active else "--"
                    cont = ""
                    if args.continuation != "none" and switch_jumps:
                        d = policy.last_continuation.get("delay", "--")
                        cont = f" jump={switch_jumps[-1]:.3f} rad d={d}"
                    # "DRY RUN" on every status line, not only in the banner at startup: the
                    # follower launch scrolls the banner away, and a dry run looks exactly like a
                    # broken one from across the room -- the policy plans, the arms do not move.
                    print(
                        f"[tick {tick:4d}] {'EXEC   ' if args.execute else 'DRY RUN'} "
                        f"t={now - start:5.1f}s plans={policy.n_plans} plan={lat} ms "
                        f"grab={timing['grab'].value:.0f} state={timing['state'].value:.0f} "
                        f"send={timing['send'].value:.1f} ms engaged={engaged}{cont}"
                    )

            # ---- newest plan --------------------------------------------------------------
            done = policy.poll()
            if done is not None:
                t0, chunk, latency, _cursor = done
                arrived = time.monotonic()
                plan_ages.append(arrived - t0)
                fresh = Plan(chunk=chunk, t0=t0, latency_s=latency)
                if active is not None and engaged:
                    # Arm joints only: the gripper is in normalised units, not radians.
                    old, new = active.target(arrived)[0], fresh.target(arrived)[0]
                    switch_jumps.append(float(np.max(np.abs(np.delete(new - old, [6, 13])))))
                active = fresh
                chunks_since_wan += 1
                if not engaged:
                    target, _ = active.target(time.monotonic())
                    err = {
                        s: float(np.max(np.abs(split_action(target)[i][:6] - arms[s].joint_pos()[:6])))
                        for i, s in enumerate(SIDES)
                    }
                    worst = max(err.values())
                    print(f"[engage] first target is {worst:.3f} rad from the measured pose ({err})")
                    if worst > args.max_start_error:
                        raise RuntimeError(
                            f"first target is {worst:.3f} rad away (limit {args.max_start_error:.2f}); the "
                            "policy does not recognise this scene -- move the arms to a demo-like start "
                            "pose, or raise --max-start-error if you are sure"
                        )
                    engaged = True

            # ---- online replanning (goal mode) ---------------------------------------------
            # The upper level on the same receding-horizon idea as the lower one: after every
            # --wan-replan-chunks chunks the plan is regenerated from what the scene looks like
            # now, so a stage the arms botched or skipped is planned from where they actually
            # are. None of it blocks the control loop -- the request is fired and forgotten, and
            # the policy keeps planning against the current set until the new one is encoded.
            if replan_chunks and wan is not None:
                if not wan.busy and fresh_subgoals is None and chunks_since_wan >= replan_chunks:
                    wan.start(_raw_top_frame(rig, allow_missing=args.allow_missing_cameras or args.sim))
                    print(f"[wan] replanning from the current top view ({chunks_since_wan} chunks executed) ...")
                    chunks_since_wan = 0
                if wan.busy:
                    ready = wan.poll()
                    if ready is not None:
                        fresh_subgoals = ready
                # set_goals shares the policy's pipe, so it has to sit out a plan in flight; at
                # ~105 ms per plan every --replan-every ticks that is a tick or two at most.
                if fresh_subgoals is not None and not policy.busy:
                    goal_frames = _condition_on_subgoals(policy, fresh_subgoals)
                    goal_gen += 1
                    saved = _save_generation(run_dir, goal_gen, fresh_subgoals) if writer is not None else None
                    print(
                        f"[wan] generation {goal_gen} in {wan.seconds:.1f}s -- policy re-conditioned, "
                        f"cursor back to 0{f' -> {saved}' if saved else ''}"
                    )
                    fresh_subgoals = None

            # ---- send (30 Hz) --------------------------------------------------------------
            if active is not None and engaged and time.monotonic() >= next_send:
                now = time.monotonic()
                # Charge the limiter for the time that actually passed, capped at the nominal
                # period. next_send = max(now, ...) means a stalled loop can fire two sends
                # back to back; billing each a full send_dt would let the command move at twice
                # --max-joint-vel for that step.
                send_elapsed = send_dt if last_send is None else min(now - last_send, send_dt)
                last_send = now
                next_send = max(now, next_send + send_dt)
                age = now - active.t0
                horizon_s = CONTROL_DT * active.chunk.shape[0]
                if age > args.max_plan_age:
                    stop_reason = f"no fresh plan for {age:.1f}s -- planning has stopped"
                    break
                if age > horizon_s and now - last_stale_warn > 1.0:
                    last_stale_warn = now
                    print(
                        f"[warn] active plan is {age:.1f}s old (> {horizon_s:.1f}s horizon) -- holding its "
                        "last step; raise --replan-every or lower --flow-steps if this persists"
                    )
                target, _ = active.target(now)
                mark = time.perf_counter()
                for side, part in zip(SIDES, split_action(target), strict=True):
                    cmd = limiters[side].step(part, send_elapsed)
                    if args.execute:
                        arms[side].command(cmd)
                timing["send"].add(time.perf_counter() - mark)

            # Sleep to the next scheduled event rather than spinning on a fixed 2 ms poll. (It is
            # not what limits plan latency -- that is GPU time shared with the 10 Hz RADIO encodes,
            # measured at ~450 ms per plan while the loop runs vs ~130 ms for a plan on its own --
            # but there is no reason to wake 500 times a second to find out nothing is due.)
            due_at = next_tick if active is None or not engaged else min(next_tick, next_send)
            time.sleep(float(np.clip(due_at - time.monotonic(), 0.0, 0.02)))

        print(f"[stop] {stop_reason}")
        if native_video is not None:
            rec._cleanup_step("finalize native videos", native_video.stop, timeout=25.0)
        if args.execute:
            if engaged and (args.release_grippers or args.park_on_exit):
                _stop_sequence(
                    arms,
                    limiters,
                    home,
                    release=args.release_grippers,
                    park=args.park_on_exit,
                    park_pose=args.park_pose,
                    release_seconds=args.release_seconds,
                    park_seconds=args.park_seconds,
                    send_hz=args.send_hz,
                    speed_scale=args.park_speed_scale,
                )
            if args.launch:
                print(
                    "[stop] the follower processes are about to exit, which sets every motor torque "
                    "to zero -- the arms go limp and drop. To keep them powered between runs, start "
                    "the followers separately (scripts/run_policy_followers.sh) and pass --no-launch."
                )
            else:
                print("[stop] the followers keep running, so the arms hold this pose")

    except KeyboardInterrupt:
        # Either a real Ctrl-C or ShutdownRequest.raise_if_requested() acting on the first signal.
        stop_reason = "interrupted"
        print("[stop] interrupted -- the arms hold their last command")
    except BaseException as e:
        stop_reason = f"{type(e).__name__}: {e}"
        raise
    finally:
        # Nothing on the way out may hang; the step that matters (killing the followers, which
        # drive the arms) is last, exactly as in the recorder. The terminal is restored first, so
        # the shell has its echo back even if a later step has to be abandoned.
        if keys is not None:
            rec._cleanup_step("restore terminal", keys.close, timeout=5.0)
        if native_video is not None:
            rec._cleanup_step("finalize native videos", native_video.stop, timeout=25.0)
        if writer is not None and writer.n_frames:
            meta = {
                "run": "box_folding_policy_rollout",
                "stop_reason": stop_reason,
                "executed": bool(args.execute),
                "policy_mode": args.policy_mode,
                "policy": policy.description if policy is not None else None,
                "instruction": policy.info.get("instruction") if policy is not None else None,
                "plans": policy.n_plans if policy is not None else 0,
                "subgoals": int(policy.n_goals) if policy is not None else 0,
                "final_goal_cursor": int(policy.goal_cursor) if policy is not None else -1,
                # How many subgoal sets drove the run and which directory each one is in:
                # subgoals/ then subgoals_01/ ... The npz's goal_gen column says which was active.
                "subgoal_sets": goal_gen + 1 if args.policy_mode == "goal" else 0,
                "wan_replan_chunks": replan_chunks,
                "continuation": args.continuation,
                # Max over arm joints of the command-target jump at each chunk switch, in rad.
                "switch_jump_rad": (
                    {
                        "n": len(switch_jumps),
                        "mean": float(np.mean(switch_jumps)),
                        "median": float(np.median(switch_jumps)),
                        "max": float(np.max(switch_jumps)),
                    }
                    if switch_jumps
                    else None
                ),
                "frames_are": "policy input 240x320 RGB" if args.record_policy_input else "raw camera BGR",
                "video": (
                    {
                        "timeline": "10 Hz policy observations",
                        "views": {},
                    }
                    if args.record_policy_input
                    else {
                        "timeline": "native camera rate, timestamp gaps filled to preserve wall-clock duration",
                        "views": native_video.summary() if native_video is not None else {},
                    }
                ),
                "args": {
                    k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                    for k, v in asdict(args).items()
                },
            }
            rec._cleanup_step("save recording", lambda: print(f"[record] saved {writer.save(meta)}"))
        elif writer is not None:
            rec._cleanup_step("discard empty recording", writer.discard)
        if args.display:
            rec._cleanup_step("close display", cv2.destroyAllWindows)
        for side, arm in arms.items():
            rec._cleanup_step(f"close {side} client", arm.close)
        if rig is not None:
            rec._cleanup_step("stop cameras", rig.stop)
        if wan is not None:
            rec._cleanup_step("stop wan process", wan.close, timeout=15.0)
        if policy is not None:
            rec._cleanup_step("stop policy process", policy.close, timeout=10.0)
        if procs:
            rec._cleanup_step("terminate followers", lambda: rec._terminate(procs), timeout=15.0)


def _resolve_checkpoint_selection(args: Args) -> Path:
    """Resolve a named bundle or the existing free-form root/checkpoint arguments.

    A DAgger preset owns the root as well as the checkpoint because the fine-tuned run has
    different action/state normalisation statistics. Selecting only its ``.pt`` file while
    leaving the old root in place would load successfully but produce incorrectly scaled inputs
    and commands.
    """
    if args.checkpoint_preset != "default":
        if args.policy_mode != "e2e":
            raise SystemExit(f"[error] --checkpoint-preset {args.checkpoint_preset} is an e2e policy")
        if args.policy_root is not None or args.checkpoint is not None:
            raise SystemExit("[error] do not combine --checkpoint-preset with --policy-root or --checkpoint")
        root_spec, relative_checkpoint = _CHECKPOINT_PRESETS[args.checkpoint_preset]
        root = Path(root_spec).expanduser().resolve()
        args.policy_root = str(root)
        args.checkpoint = str(root / relative_checkpoint)
    else:
        args.policy_root = args.policy_root or default_policy_root(args.policy_mode)

    root = Path(args.policy_root).expanduser().resolve()
    return Path(args.checkpoint).expanduser() if args.checkpoint else default_checkpoint(root, args.policy_mode)


def main(args: Args) -> None:
    if args.replan_every < 1:
        raise SystemExit("[error] --replan-every must be >= 1")
    if not 0 <= args.video_crf <= 51:
        raise SystemExit("[error] --video-crf must be between 0 and 51")
    if args.record and not args.record_policy_input and shutil.which("ffmpeg") is None:
        raise SystemExit("[error] ffmpeg is required for full-resolution eval video recording")
    # Resolve the bundle once, here, so every later use (the checkpoint check, the policy process,
    # the Wan child) sees the same path and the banner prints what will actually be loaded.
    ckpt = _resolve_checkpoint_selection(args)
    if args.display_width < 320:
        raise SystemExit("[error] --display-width must be at least 320")
    if args.policy_mode == "e2e" and args.subgoal_dir:
        raise SystemExit("[error] --subgoal-dir only means something with --policy-mode goal")
    try:
        _continuation_config(args)
    except ValueError as e:
        raise SystemExit(f"[error] {e}") from e
    if args.continuation != "none" and args.replan_every >= 32:
        print(
            f"[warn] --continuation {args.continuation} with --replan-every {args.replan_every}: consecutive "
            "chunks do not overlap, so there is nothing to continue"
        )
    if args.wan_replan_chunks < 0:
        raise SystemExit("[error] --wan-replan-chunks must be >= 0 (0 = one generation per episode)")
    # Fail on a missing checkpoint before the followers are launched and the arms are powered,
    # not 30 s later inside the policy process.
    if not ckpt.is_file():
        raise SystemExit(f"[error] checkpoint not found: {ckpt}")
    print(f"[policy] {args.policy_mode}: {ckpt}")
    if not args.execute:
        print(
            "[warn] DRY RUN -- the policy will plan but no command is sent, so the arms just sit "
            "there holding position under gravity compensation. Pass --execute to drive them."
        )
    run(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
