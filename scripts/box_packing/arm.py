"""Cartesian control of one YAM follower arm over the minimum_gello RPC server.

Wraps a ``portal`` client (every call bounded, like the recorder), the arm+gripper MuJoCo model
for FK, and mink for IK on the ``grasp_site`` frame (the point between the fingertips, 13.47 cm
beyond the terminal mount body). All motion is streamed as absolute joint targets at
``send_hz`` with a slew-rate limit, so nothing moves faster than ``max_joint_vel``.

Frames: everything is in the arm's own base frame (x forward, z up). Gripper command is the
follower's normalized value, 0 = closed, 1 = open.

    arm = Arm("left", 1235)
    pos, R = arm.grasp_pose()             # current fingertip-centre pose
    arm.move_grasp(pos + [0, 0, 0.05], R) # IK + slewed joint move, blocking
    arm.set_gripper(0.0)                  # close
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mink
import mujoco
import numpy as np
import portal

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

_RPC_TIMEOUT_S = 2.0
_GRASP_SITE = "grasp_site"
_MOUNT_BODY = "gripper"
_WRIST_RADIUS = 0.10
"""Metres of fingertip travel one radian of joint motion is charged as, when a path is measured
for resampling and speed limiting. A wrist turn in place moves the tips almost not at all, so on
Cartesian distance alone the resampler would collapse it to nothing and then run it at the joint
slew limit; charging it 10 cm per radian keeps it in the path and ties its speed to the tip cap
(2 rad/s at 20 cm/s, before max_joint_vel is applied on top)."""


@dataclass
class Workspace:
    """Axis-aligned box the grasp point must stay inside, base frame, metres."""

    x: Tuple[float, float] = (0.12, 0.62)
    y: Tuple[float, float] = (-0.45, 0.45)
    z: Tuple[float, float] = (0.0, 0.55)

    def contains(self, p: np.ndarray) -> bool:
        return bool(self.x[0] <= p[0] <= self.x[1] and self.y[0] <= p[1] <= self.y[1] and self.z[0] <= p[2] <= self.z[1])

    def clip(self, p: np.ndarray) -> np.ndarray:
        lo = np.array([self.x[0], self.y[0], self.z[0]])
        hi = np.array([self.x[1], self.y[1], self.z[1]])
        return np.clip(p, lo, hi)


def load_joint_offset(side: str) -> np.ndarray:
    """Per-arm joint offsets, ``hardware = model + offset`` (6 values, radians).

    Normally all zero, as they are now. The right arm's joint 6 once read 26.5 deg low (its
    gripper had rotated on the joint-6 flange), and this file used to carry that as a
    box_packing-only patch at the RPC boundary. Zero corrections moved to
    scripts/arm_offsets.conf, where they are applied at the motor chain inside every
    minimum_gello follower (see arm_offsets.py), so what the RPC server reports and accepts
    already IS the model frame -- for this package, the recorder, the policy rollout and
    teleop alike. (That table is empty again since 2026-09-11; see its header for the history.)
    This hook stays for a package-local tweak that must not leak into the shared frame;
    anything that is a real zero error belongs in the table.
    """
    path = Path(__file__).resolve().parent / "calib" / f"joint_offset_{side}.json"
    try:
        import json

        return np.asarray(json.loads(path.read_text())["offset"], dtype=np.float64)
    except Exception:
        return np.zeros(6)


class RpcArm:
    """Bounded RPC access to one follower server, in the *model* joint frame."""

    def __init__(self, port: int, host: str = "127.0.0.1", offset: Optional[np.ndarray] = None) -> None:
        self._addr = f"{host}:{port}"
        self._client = portal.Client(self._addr)
        self.num_dofs = int(self._client.num_dofs().result(timeout=10.0))
        self.offset = np.zeros(6) if offset is None else np.asarray(offset, dtype=np.float64)

    def joint_pos(self) -> np.ndarray:
        q = np.asarray(self._client.get_joint_pos().result(timeout=_RPC_TIMEOUT_S), dtype=np.float64)
        q[:6] -= self.offset
        return q

    def command(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).copy()
        q[:6] += self.offset
        self._client.command_joint_pos(q).result(timeout=_RPC_TIMEOUT_S)

    def close(self) -> None:
        self._client.close(timeout=2.0)


class Kinematics:
    """FK/IK on the composed arm+gripper MJCF. IK solves for the 6 arm joints only."""

    def __init__(self, arm: str = "yam", version: int = 1, gripper: str = "linear_4310") -> None:
        xml = combine_arm_and_gripper_xml(
            ArmType.from_string_name(arm), GripperType.from_string_name(gripper), version=version
        )
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.n_arm = 6
        self.site_id = self.model.site(_GRASP_SITE).id
        self.mount_id = self.model.body(_MOUNT_BODY).id
        self.joint_range = self.model.jnt_range[: self.n_arm].copy()
        self._cfg = mink.Configuration(self.model)
        self._task = mink.FrameTask(_GRASP_SITE, "site", position_cost=1.0, orientation_cost=0.5, lm_damping=1e-3)
        self._posture = mink.PostureTask(self.model, cost=1e-3)
        self._limits = [mink.ConfigurationLimit(self.model)]

    def _set(self, q: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[: self.n_arm] = q[: self.n_arm]
        mujoco.mj_kinematics(self.model, self.data)

    def grasp(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(pos(3), R(3x3)) of the grasp site in the base frame."""
        self._set(q)
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def mount(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(pos(3), R(3x3)) of the terminal mount body (the recorder's ``gripper`` body)."""
        self._set(q)
        return self.data.xpos[self.mount_id].copy(), self.data.xmat[self.mount_id].reshape(3, 3).copy()

    def ik(
        self,
        pos: np.ndarray,
        R: np.ndarray,
        q_init: np.ndarray,
        iters: int = 200,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
    ) -> Tuple[np.ndarray, float, float]:
        """Arm joints (6) reaching the grasp-site pose. Returns (q, pos_err_m, rot_err_rad)."""
        q0 = np.zeros(self.model.nq)
        q0[: self.n_arm] = q_init[: self.n_arm]
        self._cfg.update(q0)
        self._posture.set_target(q0)
        self._task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(np.asarray(R)), np.asarray(pos)))
        dt = 0.05
        best = np.inf
        stalled = 0
        for _ in range(iters):
            v = mink.solve_ik(self._cfg, [self._task, self._posture], dt, "daqp", damping=1e-4, limits=self._limits)
            v[self.n_arm :] = 0.0
            self._cfg.integrate_inplace(v, dt)
            err = self._task.compute_error(self._cfg)
            if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
                break
            # Stop once it stops improving. An unreachable target otherwise burns all 200
            # iterations, and a plan asks for hundreds of these -- that was seconds of the
            # click tool's lag, not the cameras.
            total = float(np.linalg.norm(err[:3]) + 0.1 * np.linalg.norm(err[3:]))
            if total > best - 1e-5:
                stalled += 1
                if stalled >= 5:
                    break
            else:
                stalled = 0
            best = min(best, total)
        err = self._task.compute_error(self._cfg)
        return self._cfg.q[: self.n_arm].copy(), float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))

    def ik_topdown(self, pos: np.ndarray, yaw: float, q_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vertical grasp at ``pos`` with the finger axis at ``yaw`` (mod pi -- the gripper is
        symmetric, so yaw+pi is the same grasp and is tried as well; joint6 only spans +-120 deg,
        which is why one of the two is often unreachable). Returns (q6, R) of the best branch."""
        best = None
        # mink's IK is local: seed it with the current pose and with a canonical over-the-top
        # configuration, which is the branch every vertical grasp in front of the arm lives on.
        seeds = [np.asarray(q_init)[:6], _TOPDOWN_SEED, np.zeros(6)]
        for cand in (yaw, yaw + np.pi, yaw - np.pi):
            R = rot_top_down(cand)
            for seed in seeds:
                q6, perr, rerr = self.ik(pos, R, seed)
                # prefer the branch closest to where the arm is, then a centred wrist roll
                score = perr + 0.1 * rerr + 0.002 * abs(q6[5]) + 0.01 * np.linalg.norm(q6 - np.asarray(q_init)[:6])
                if perr < 5e-3 and rerr < 0.05 and (best is None or score < best[0]):
                    best = (score, q6, R)
        if best is None:
            raise ValueError(f"no top-down IK solution at {np.round(pos, 3)} yaw {yaw:.2f}")
        return best[1], best[2]


    def ik_grasp(
        self,
        pos: np.ndarray,
        finger_dir: np.ndarray,
        q_init: np.ndarray,
        n_roll: int = 8,
        finger_axis: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Grasp at ``pos`` with the fingers pointing along ``finger_dir`` (base frame), the
        roll about that axis free: the gripper is symmetric and a cup does not care, so the
        ``n_roll`` candidates are scanned and the reachable one closest to ``q_init`` wins.
        ``finger_axis`` (base frame, sign-free) asks for the fingers to open along that
        direction; reachable rolls are then ranked by how close they come to it."""
        d = np.asarray(finger_dir, dtype=np.float64)
        d = d / np.linalg.norm(d)
        # a reference vector perpendicular to d
        ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ref = ref - d * (ref @ d)
        ref /= np.linalg.norm(ref)
        want = None
        if finger_axis is not None:
            want = np.asarray(finger_axis, dtype=np.float64)
            want = want - d * (want @ d)
            want = want / max(np.linalg.norm(want), 1e-9)
        seeds = [np.asarray(q_init)[:6], _TOPDOWN_SEED, _FORWARD_SEED]
        best = None
        rolls = list(np.linspace(0.0, 2 * np.pi, n_roll, endpoint=False))
        if want is not None:
            # exact alignment first: site y (the finger axis) = want, both signs -- the
            # sampled rolls only come within 180/n_roll degrees, which is what put the
            # fingers 22 deg off a cup's radial direction
            y_dir = np.cross(d, np.cross(want, d))
            y_dir /= max(np.linalg.norm(y_dir), 1e-9)
            for sign in (1.0, -1.0):
                x_exact = np.cross(sign * y_dir, d)
                rolls.insert(0, float(np.arctan2(x_exact @ np.cross(d, ref), x_exact @ ref)))
        for roll in rolls:
            x_hint = np.cos(roll) * ref + np.sin(roll) * np.cross(d, ref)
            R = rot_from_z_axis(d, x_hint)
            for seed in seeds:
                q6, perr, rerr = self.ik(pos, R, seed)
                if perr < 5e-3 and rerr < 0.05:
                    score = 0.002 * abs(q6[5]) + 0.01 * np.linalg.norm(q6 - np.asarray(q_init)[:6])
                    if want is not None:
                        # site y is the finger-opening axis (tips sit at +-y of the mount)
                        score += 1.0 - abs(float(R[:, 1] @ want))
                    if best is None or score < best[0]:
                        best = (score, q6, R)
                    break  # this roll is reachable; no need for more seeds
        if best is None:
            raise ValueError(f"no IK solution at {np.round(pos, 3)} with finger dir {np.round(d, 2)}")
        return best[1], best[2]

    def ik_reach(
        self,
        pos: np.ndarray,
        q_init: np.ndarray,
        tilts: Tuple[float, ...] = (0.0, 20.0, 35.0, 50.0, 65.0, 80.0),
        finger_axis: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Vertical grasp if reachable, else tilt the fingers forward (away from the base, in
        the direction of the target) by increasing angles. Returns (q6, R, tilt_deg)."""
        pos = np.asarray(pos, dtype=np.float64)
        away = np.array([pos[0], pos[1], 0.0])
        away = away / max(np.linalg.norm(away), 1e-6)
        for tilt in tilts:
            t = np.radians(tilt)
            d = -np.array([0.0, 0.0, 1.0]) * np.cos(t) + away * np.sin(t)
            try:
                q6, R = self.ik_grasp(pos, d, q_init, finger_axis=finger_axis)
                return q6, R, float(tilt)
            except ValueError:
                continue
        raise ValueError(f"unreachable at any tilt: {np.round(pos, 3)}")


_TOPDOWN_SEED = np.array([0.4, 2.0, 1.9, -1.5, 0.0, 0.4])
"""A fingers-down configuration in front of the arm (joint2/3 folded over the top)."""
_FORWARD_SEED = np.array([0.0, 1.2, 1.2, -0.8, 0.0, 0.0])
"""A fingers-forward-and-down configuration, the seed for tilted reaches further out."""


def rot_top_down(yaw: float = 0.0) -> np.ndarray:
    """Grasp-site rotation for a vertical, fingers-down grasp. The site's frame is the mount
    frame flipped (quat 0 1 0 0 = 180 deg about x), so site +z points *out of* the fingers,
    i.e. down into the table; ``yaw`` rotates the finger-opening axis about world z."""
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    # site frame: x -> world x, y -> -world y, z -> -world z (a 180 deg turn about x)
    Rx_pi = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return Rz @ Rx_pi


def rot_from_z_axis(z_world: np.ndarray, x_hint: np.ndarray = np.array([1.0, 0.0, 0.0])) -> np.ndarray:
    """Site rotation whose +z (finger direction) is ``z_world``; x as close to ``x_hint`` as possible."""
    z = np.asarray(z_world, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, dtype=np.float64)
    x = x - z * (x @ z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - z * z[1]
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


@dataclass
class MoveResult:
    reached: bool
    max_err: float
    seconds: float
    aborted: str = ""
    trace: List[np.ndarray] = field(default_factory=list)


class Arm:
    """One follower arm: measured state, FK/IK, slewed blocking moves, gripper."""

    def __init__(
        self,
        side: str,
        port: int,
        arm: str = "yam",
        version: int = 1,
        gripper: str = "linear_4310",
        max_joint_vel: float = 0.8,
        max_gripper_vel: float = 1.5,
        send_hz: float = 50.0,
        workspace: Optional[Workspace] = None,
        execute: bool = True,
    ) -> None:
        self.side = side
        self.joint_offset = load_joint_offset(side)
        self.rpc = RpcArm(port, offset=self.joint_offset)
        if np.any(self.joint_offset):
            logging.info(f"[{side}] joint offset (hardware - model): {np.round(self.joint_offset, 3)}")
        self.kin = Kinematics(arm, version, gripper)
        self.max_joint_vel = float(max_joint_vel)
        self.max_gripper_vel = float(max_gripper_vel)
        self.send_dt = 1.0 / float(send_hz)
        self.workspace = workspace or Workspace()
        self.execute = execute
        self.q_cmd: np.ndarray = self.rpc.joint_pos()  # last command we streamed (starts at measured)
        self.track_abort_rad = 0.35
        """Abort a move if any arm joint lags the command by more than this (collision / stall)."""
        logging.info(f"[{side}] up, q={np.round(self.q_cmd, 3)}")

    # ---- state -------------------------------------------------------------------------
    def q(self) -> np.ndarray:
        return self.rpc.joint_pos()

    def grasp_pose(self, q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        return self.kin.grasp(self.q() if q is None else q)

    def mount_pose(self, q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        return self.kin.mount(self.q() if q is None else q)

    def gripper(self) -> float:
        return float(self.q()[6])

    # ---- moves -------------------------------------------------------------------------
    def _send(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64).copy()
        q[:6] = np.clip(q[:6], self.kin.joint_range[:, 0], self.kin.joint_range[:, 1])
        q[6] = float(np.clip(q[6], 0.0, 1.0))
        self.q_cmd = q
        if self.execute:
            self.rpc.command(q)

    def move_joints(self, q_target: np.ndarray, min_seconds: float = 0.0, settle: float = 0.3) -> MoveResult:
        """Slew from the last command to ``q_target`` (7-D, gripper included) and wait to settle."""
        q_target = np.asarray(q_target, dtype=np.float64)
        if q_target.size == 6:
            q_target = np.concatenate([q_target, [self.q_cmd[6]]])
        start = self.q_cmd.copy()
        delta = q_target - start
        seconds = max(
            float(np.max(np.abs(delta[:6])) / self.max_joint_vel),
            float(abs(delta[6]) / self.max_gripper_vel),
            min_seconds,
            self.send_dt,
        )
        t0 = time.monotonic()
        # Gravity sag is not contact: at a stretched pose the soft wrist joints trail the
        # command by 0.1 rad or more just holding still. The abort threshold is therefore
        # measured *relative to* the lag the move starts with.
        base_lag = np.abs(self.q()[:6] - self.q_cmd[:6]) if self.execute else np.zeros(6)
        result = MoveResult(reached=False, max_err=0.0, seconds=seconds)
        while True:
            now = time.monotonic()
            s = min(1.0, (now - t0) / seconds)
            s_smooth = 0.5 - 0.5 * np.cos(np.pi * s)  # ease in/out; peak vel = pi/2 x mean
            self._send(start + delta * s_smooth)
            if self.execute:
                lag = np.abs(self.q()[:6] - self.q_cmd[:6]) - base_lag
                if np.max(lag) > self.track_abort_rad:
                    result.aborted = f"joint {int(np.argmax(lag)) + 1} lags command by {np.max(lag):.2f} rad beyond its baseline"
                    logging.error(f"[{self.side}] move aborted: {result.aborted}")
                    self._send(self.q())  # hold where we are
                    return result
            if s >= 1.0:
                break
            time.sleep(self.send_dt)
        time.sleep(settle)
        if self.execute:
            err = np.abs(self.q()[:6] - q_target[:6])
            result.max_err = float(np.max(err))
            result.reached = result.max_err < 0.05
        else:
            result.reached = True
        return result

    def spline(self, waypoints: List[np.ndarray], samples: int = 60) -> np.ndarray:
        """Catmull-Rom spline through joint-space waypoints, sampled uniformly in its own
        parameter. Point-to-point moves stop at every waypoint, which is what makes a plan
        look like a machine; one curve through them keeps the tool moving."""
        Q = np.stack([np.asarray(w, dtype=np.float64)[: self.q_cmd.size] for w in waypoints])
        if len(Q) == 2:
            t = np.linspace(0.0, 1.0, samples)[:, None]
            return Q[0] + (Q[1] - Q[0]) * t
        # duplicate the ends so the curve starts and finishes at the first/last waypoint
        P = np.vstack([Q[0], Q, Q[-1]])
        out = []
        per = max(2, samples // (len(Q) - 1))
        for i in range(len(Q) - 1):
            p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
            for j in range(per):
                t = j / per
                t2, t3 = t * t, t * t * t
                out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
        out.append(Q[-1])
        return np.stack(out)

    def move_through(self, waypoints: List[np.ndarray], speed_scale: float = 1.0, settle: float = 0.0) -> MoveResult:
        """Stream one smooth curve through ``waypoints`` (joint space) without stopping at
        the intermediate ones. Same slew limit, ease and abort as move_joints."""
        path = self.spline([self.q_cmd] + list(waypoints))
        step = np.abs(np.diff(path[:, :6], axis=0)).max(axis=1)
        total = float(step.sum())
        seconds = max(total / (self.max_joint_vel * speed_scale), self.send_dt)
        arc = np.concatenate([[0.0], np.cumsum(step)])
        arc = arc / max(arc[-1], 1e-9)
        t0 = time.monotonic()
        base_lag = np.abs(self.q()[:6] - self.q_cmd[:6]) if self.execute else np.zeros(6)
        result = MoveResult(reached=False, max_err=0.0, seconds=seconds)
        while True:
            s = min(1.0, (time.monotonic() - t0) / seconds)
            u = 0.5 - 0.5 * np.cos(np.pi * s)  # ease in/out along the whole curve
            k = int(np.searchsorted(arc, u, side="right")) - 1
            k = int(np.clip(k, 0, len(path) - 2))
            w = (u - arc[k]) / max(arc[k + 1] - arc[k], 1e-9)
            self._send(path[k] * (1 - w) + path[k + 1] * w)
            if self.execute:
                lag = np.abs(self.q()[:6] - self.q_cmd[:6]) - base_lag
                if np.max(lag) > self.track_abort_rad:
                    result.aborted = f"joint {int(np.argmax(lag)) + 1} lags by {np.max(lag):.2f} rad beyond its baseline"
                    self._send(self.q())
                    return result
            if s >= 1.0:
                break
            time.sleep(self.send_dt)
        if settle:
            time.sleep(settle)
        if self.execute:
            result.max_err = float(np.max(np.abs(self.q()[:6] - path[-1][:6])))
            result.reached = result.max_err < 0.05
        else:
            result.reached = True
        return result

    # ---- continuous, speed-capped paths ------------------------------------------------
    #
    # Point-to-point moves are what make a plan look like a machine: every waypoint is a full
    # stop, and the turn through it is a right angle. The calls below build one dense joint
    # path for a whole leg of a task instead -- fly out, drop onto the object; lift, carry,
    # come down on the target -- round its corners off, and stream it under a *Cartesian*
    # speed cap. What comes out moves the way a hand on a teleoperation leader moves: one
    # bounded, curved, unbroken motion that eases in at the start and out at the end and
    # holds a steady tip speed in between.
    #
    #     path = arm.joint_curve([q_over_object, q_pre_grasp])          # gross transit
    #     path = np.vstack([path, arm.line_curve(grasp, R, q_start=path[-1])[1:]])
    #     path = arm.smooth_path(arm.resample_path(path))               # round the corner
    #     arm.move_smooth(path, v_eef=0.20, v_slow=0.06, slow_len=0.14)
    #
    # A path is (N, 7) joint samples -- 6 arm joints and the gripper, which is streamed along
    # with them, so the fingers open while the arm is still flying. An optional 8th column is
    # a per-sample tracking-lag abort threshold, which is how a descent that has to stop on
    # contact rides in the same path as a transit that must not false-trigger.

    def joint_curve(self, waypoints: List[np.ndarray], q_start: Optional[np.ndarray] = None, samples: int = 160) -> np.ndarray:
        """Catmull-Rom curve from ``q_start`` (default: the last command) through ``waypoints``."""
        q0 = self.q_cmd if q_start is None else np.asarray(q_start, dtype=np.float64)
        q0 = self._q7(q0)
        pts = [q0] + [self._q7(w, grip=float(q0[6])) for w in waypoints]
        return self.spline(pts, samples=samples)

    def line_curve(
        self,
        pos_to: np.ndarray,
        R: np.ndarray,
        q_start: Optional[np.ndarray] = None,
        step: float = 0.006,
        strict: bool = True,
        seeds: Tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        """Joint samples along a straight Cartesian line of the grasp site, at fixed ``R``.

        Unlike move_linear this only *builds* the path -- nothing is sent -- so a straight
        descent can be concatenated onto the transit that leads into it and the two run as one
        motion. ``strict=False`` stops at the last reachable point instead of raising, which is
        what a lift out of a box wants: however far up the arm can carry it is far enough.

        ``seeds`` are extra IK starting points tried when the chained one comes up short. Pass
        the solution for the far end of the line: mink is a local solver that gives up as soon
        as it stops improving, and walking it down from the near end drifts off the branch a
        stretched-out reach lives on -- 22 mm short at the bottom of a 7 cm descent at the far
        edge of the table, where seeding from the destination itself is exact all the way.
        """
        q0 = self._q7(self.q_cmd if q_start is None else q_start)
        p0 = self.kin.grasp(q0)[0]
        pos_to = np.asarray(pos_to, dtype=np.float64)
        if not self.workspace.contains(pos_to):
            raise ValueError(f"[{self.side}] target {np.round(pos_to, 3)} outside workspace {self.workspace}")
        n = max(2, int(np.ceil(float(np.linalg.norm(pos_to - p0)) / step)) + 1)
        out = [q0]
        for i in range(1, n):
            p = p0 + (pos_to - p0) * (i / (n - 1))
            # mink's IK is local and its stall test gives up early: near the edge of the
            # workspace the chained seed can land a few millimetres short where the pose the
            # line was solved around still reaches, so that one is tried as well.
            for seed in (out[-1], *seeds, q0):
                q6, perr, rerr = self.kin.ik(p, R, seed)
                if perr <= 5e-3 and rerr <= 0.05:
                    break
            if perr > 5e-3 or rerr > 0.05:
                if strict or len(out) < 2:
                    raise ValueError(f"[{self.side}] IK failed on the line at {np.round(p, 3)}: {perr * 1e3:.1f} mm")
                break
            out.append(np.concatenate([q6, [q0[6]]]))
        return np.stack(out)

    def arch_point(
        self,
        q_from: np.ndarray,
        pos_to: np.ndarray,
        R: Optional[np.ndarray] = None,
        height: float = 0.07,
        finger_axis: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Joints for the top of an arch over the straight line from ``q_from`` to ``pos_to``.

        Carrying something across at a constant height is the one part of a pick that never
        looks hand-driven: a person lifts the object clear, swings it over and lowers it in,
        which traces an arc, not a table edge. This is the apex of that arc -- above the
        midpoint, ``height`` over whichever end is higher and never more than a third of the
        distance covered, so a short move barely bulges and a long one really does go over the
        top. It holds ``R`` if it can -- the orientation the object was picked up with, so the
        wrist only turns to the release orientation on the way down -- and otherwise leans as
        far as it must (ik_reach), because the apex is by definition the highest point of the
        run and a vertical wrist gives out somewhere above 20 cm on most of this table.

        Returns None if nothing that high is reachable, in which case the carry stays flat.
        """
        p0 = self.kin.grasp(self._q7(q_from))[0]
        pos_to = np.asarray(pos_to, dtype=np.float64)
        q_seed = self._q7(q_from)
        h = min(float(height), 0.3 * float(np.linalg.norm((pos_to - p0)[:2])))
        while h > 0.005:
            apex = np.array([0.5 * (p0[0] + pos_to[0]), 0.5 * (p0[1] + pos_to[1]), max(p0[2], pos_to[2]) + h])
            if self.workspace.contains(apex):
                if R is not None:
                    q6, perr, rerr = self.kin.ik(apex, R, q_seed)
                    if perr <= 5e-3 and rerr <= 0.05:
                        return q6
                try:
                    return self.kin.ik_reach(apex, q_seed, finger_axis=finger_axis)[0]
                except ValueError:
                    pass
            h *= 0.5
        return None

    def _q7(self, q: np.ndarray, grip: Optional[float] = None) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.size >= 7:
            return q[:7].copy()
        return np.concatenate([q[:6], [self.q_cmd[6] if grip is None else grip]])

    def path_arc(self, path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(grasp-site positions, cumulative travel in metres) along a joint path.

        Travel is the larger of the tip's own motion and ``_WRIST_RADIUS`` per radian of joint
        motion, so a re-orientation in place still counts as distance.
        """
        path = np.asarray(path, dtype=np.float64)
        pos = np.stack([self.kin.grasp(q)[0] for q in path])
        d_cart = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        d_joint = np.abs(np.diff(path[:, :6], axis=0)).max(axis=1)
        return pos, np.concatenate([[0.0], np.cumsum(np.maximum(d_cart, _WRIST_RADIUS * d_joint))])

    def resample_path(self, path: np.ndarray, spacing: float = 0.004) -> np.ndarray:
        """Re-space a path evenly along its own travel, so a window in samples is a window in
        metres -- which is what lets one smoothing radius mean the same thing everywhere."""
        path = np.asarray(path, dtype=np.float64)
        if len(path) < 2:
            return path
        _, s = self.path_arc(path)
        keep = np.concatenate([[True], np.diff(s) > 1e-9])
        path, s = path[keep], s[keep]
        if len(path) < 2 or s[-1] < 1e-9:
            return path
        u = np.linspace(0.0, s[-1], max(2, int(np.ceil(s[-1] / spacing)) + 1))
        return np.stack([np.interp(u, s, path[:, j]) for j in range(path.shape[1])], axis=1)

    def smooth_path(
        self, path: np.ndarray, radius: float = 0.03, pin: float = 0.02, spacing: float = 0.004, passes: int = 2
    ) -> np.ndarray:
        """Round the corners off an evenly spaced path with a tapered moving average.

        A straight run is unchanged by an average of its own neighbours, so this only bites
        where the path actually turns -- the junction between a transit and the descent that
        follows it. ``pin`` metres at each end are left exactly as they were and the weight
        ramps in over ``pin`` more, which is what keeps the last stretch of a descent truly
        vertical and every endpoint exactly on target.
        """
        path = np.asarray(path, dtype=np.float64)
        n = len(path)
        half = min(round(radius / max(spacing, 1e-6)), (n - 1) // 2)
        if n < 5 or half < 1:
            return path
        keep = max(1, round(pin / max(spacing, 1e-6)))
        idx = np.clip(np.arange(n)[:, None] + np.arange(-half, half + 1)[None, :], 0, n - 1)
        ramp = np.clip((np.arange(n) - keep) / keep, 0.0, 1.0)
        w = (0.5 - 0.5 * np.cos(np.pi * np.minimum(ramp, ramp[::-1])))[:, None]
        out = path
        for _ in range(passes):
            out = out * (1.0 - w) + out[idx].mean(axis=1) * w
        return out

    def path_time(
        self,
        path: np.ndarray,
        v_eef: float,
        v_slow: Optional[float] = None,
        slow_len: float = 0.10,
        accel: float = 0.35,
        start_speed: float = 0.03,
    ) -> np.ndarray:
        """When each sample of ``path`` is due, from a tip-speed cap.

        The cap is ``v_eef`` for most of the way and, if ``v_slow`` is given, eases down to
        that over the last ``slow_len`` metres -- the way a hand slows as it closes on what it
        is reaching for. The two ends are then bounded by ``accel`` m/s^2, the classic
        ``v <= sqrt(2 a d)`` ramp, so nothing starts or stops with a jerk.

        The braking law matters more than it looks. Scaling the cap by a smooth 0..1 window
        over a fixed length of path *asymptotes*: the last centimetre is run at a fixed
        fraction of an already-slow approach speed, which took two full seconds to cover the
        last four centimetres of a descent and read, to anyone watching, as the arm stopping
        dead on the object and thinking about it. A real deceleration limit spends 0.76 s on
        that same stretch and comes to rest in the last 5 mm.

        ``start_speed`` is the floor the deceleration ramp is not allowed to go below (unless
        the cap itself is lower). Without it the ramp is exactly zero at each end, and since
        the path is sampled every few millimetres the *first* interval alone then takes 0.15 s
        -- the arm sitting still on the object it has just gripped, which is what a stop before
        the lift looks like from outside. A few cm/s is not a jerk on a position-streamed
        servo, and it is what makes a motion look like it starts when it starts.

        The joint and gripper slew limits still apply on top.
        """
        path = np.asarray(path, dtype=np.float64)
        _, s = self.path_arc(path)
        total = float(s[-1])
        v = np.full(len(s), float(v_eef))
        if v_slow is not None and slow_len > 1e-6:
            near = 0.5 - 0.5 * np.cos(np.pi * np.clip((total - s) / slow_len, 0.0, 1.0))
            v = float(v_slow) + (v - float(v_slow)) * near
        capped = v.copy()  # the local speed cap, before the ends are rounded off
        if accel > 0:
            room = np.minimum(s, total - s)  # distance to the nearer end of the path
            v = np.minimum(v, np.sqrt(2.0 * float(accel) * np.maximum(room, 0.0)))
        v = np.maximum(v, np.minimum(capped, float(start_speed)))
        dt = np.diff(s) / np.maximum(0.5 * (v[:-1] + v[1:]), 1e-6)
        dt = np.maximum(dt, np.abs(np.diff(path[:, :6], axis=0)).max(axis=1) / max(self.max_joint_vel, 1e-6))
        dt = np.maximum(dt, np.abs(np.diff(path[:, 6])) / max(self.max_gripper_vel, 1e-6))
        return np.concatenate([[0.0], np.cumsum(np.maximum(dt, 1e-4))])

    def move_smooth(
        self,
        path: np.ndarray,
        v_eef: float = 0.10,
        v_slow: Optional[float] = None,
        slow_len: float = 0.10,
        accel: float = 0.35,
        start_speed: float = 0.03,
        settle: float = 0.0,
    ) -> MoveResult:
        """Stream a whole path at a bounded tip speed, without stopping anywhere inside it.

        ``path`` is (N, 7) or (N, 8) -- see the note above; column 7, if present, is the
        tracking-lag abort threshold in force at that sample.
        """
        path = np.atleast_2d(np.asarray(path, dtype=np.float64))
        abort = path[:, 7] if path.shape[1] > 7 else None
        q_path = path[:, :7]
        if len(q_path) < 2:
            self._send(q_path[-1])
            time.sleep(settle)
            return MoveResult(reached=True, max_err=0.0, seconds=0.0)
        t = self.path_time(q_path, v_eef, v_slow, slow_len, accel, start_speed)
        total = float(t[-1])
        t0 = time.monotonic()
        base_lag = np.abs(self.q()[:6] - self.q_cmd[:6]) if self.execute else np.zeros(6)
        result = MoveResult(reached=False, max_err=0.0, seconds=total)
        while True:
            now = min(total, time.monotonic() - t0)
            k = int(np.clip(np.searchsorted(t, now, side="right") - 1, 0, len(t) - 2))
            w = (now - t[k]) / max(t[k + 1] - t[k], 1e-9)
            self._send(q_path[k] * (1.0 - w) + q_path[k + 1] * w)
            if self.execute:
                lag = np.abs(self.q()[:6] - self.q_cmd[:6]) - base_lag
                limit = self.track_abort_rad if abort is None else float(abort[k])
                if np.max(lag) > limit:
                    result.aborted = f"joint {int(np.argmax(lag)) + 1} lags by {np.max(lag):.2f} rad beyond its baseline"
                    logging.error(f"[{self.side}] smooth move aborted: {result.aborted}")
                    self._send(self.q())
                    return result
            if now >= total:
                break
            time.sleep(self.send_dt)
        if settle:
            time.sleep(settle)
        if self.execute:
            result.max_err = float(np.max(np.abs(self.q()[:6] - q_path[-1][:6])))
            result.reached = result.max_err < 0.05
        else:
            result.reached = True
        return result

    def move_grasp(
        self, pos: np.ndarray, R: np.ndarray, min_seconds: float = 0.0, settle: float = 0.3, correct: int = 1
    ) -> MoveResult:
        """IK for the grasp site then a joint move. Refuses targets outside the workspace box.

        ``correct`` extra rounds re-aim at the target offset by the measured position residual:
        the wrist joints run soft gains (kp 10) and sag a centimetre under gravity, and a
        commanded pose that is off by the sag lands on target."""
        pos = np.asarray(pos, dtype=np.float64)
        if not self.workspace.contains(pos):
            raise ValueError(f"[{self.side}] target {np.round(pos, 3)} outside workspace {self.workspace}")
        q6, perr, rerr = self.kin.ik(pos, R, self.q_cmd)
        if perr > 5e-3 or rerr > 0.05:
            raise ValueError(f"[{self.side}] IK failed for {np.round(pos, 3)}: pos err {perr * 1e3:.1f} mm, rot err {rerr:.3f}")
        result = self.move_joints(np.concatenate([q6, [self.q_cmd[6]]]), min_seconds=min_seconds, settle=settle)
        for _ in range(correct if self.execute else 0):
            if result.aborted:
                break
            result = self._correct(pos, R, settle)
        return result

    def _correct(self, pos: np.ndarray, R: np.ndarray, settle: float, v_eef: Optional[float] = None) -> MoveResult:
        """One round of position residual compensation (see move_grasp).

        ``v_eef`` caps how fast the nudge itself travels, m/s. Without it a centimetre of sag
        is taken out in a slew-limited 30 ms -- a flick at a third of a metre a second on the
        end of an otherwise unhurried motion, which is exactly the tic that gives a scripted
        arm away.
        """
        p_now, _ = self.grasp_pose()
        residual = pos - p_now
        gap = float(np.linalg.norm(residual))
        if gap < 2e-3:
            return MoveResult(reached=True, max_err=gap, seconds=0.0)
        aim = self.kin.grasp(self.q_cmd)[0] + residual  # shift the *commanded* pose by the residual
        q6, perr, rerr = self.kin.ik(aim, R, self.q_cmd)
        if perr > 5e-3 or rerr > 0.05:
            return MoveResult(reached=False, max_err=gap, seconds=0.0, aborted="ik on correction")
        floor = 0.0 if v_eef is None else gap / max(float(v_eef), 1e-6)
        result = self.move_joints(np.concatenate([q6, [self.q_cmd[6]]]), min_seconds=floor, settle=settle)
        p_now, _ = self.grasp_pose()
        result.max_err = float(np.linalg.norm(pos - p_now))
        result.reached = result.max_err < 5e-3
        return result

    def settle_onto(
        self, pos: np.ndarray, R: np.ndarray, rounds: int = 3, settle: float = 0.08, v_eef: Optional[float] = None
    ) -> MoveResult:
        """Drive the *measured* fingertips onto ``pos``, a residual at a time (see move_grasp).

        The public form of the correction rounds move_grasp/move_linear run internally, for a
        path that was streamed by move_smooth and so has no correction of its own. Stops as
        soon as a round finds nothing left to take out.
        """
        result = MoveResult(reached=True, max_err=0.0, seconds=0.0)
        for _ in range(rounds if self.execute else 0):
            result = self._correct(np.asarray(pos, dtype=np.float64), R, settle, v_eef=v_eef)
            if result.aborted or result.reached:
                break
        return result

    def move_reach(
        self, pos: np.ndarray, correct: int = 1, settle: float = 0.3, finger_axis: Optional[np.ndarray] = None
    ) -> Tuple[MoveResult, np.ndarray, float]:
        """Grasp-site to ``pos`` with the fingers as vertical as reachability allows (see
        Kinematics.ik_reach). Returns (result, R used, tilt_deg)."""
        pos = np.asarray(pos, dtype=np.float64)
        if not self.workspace.contains(pos):
            raise ValueError(f"[{self.side}] target {np.round(pos, 3)} outside workspace {self.workspace}")
        q6, R, tilt = self.kin.ik_reach(pos, self.q_cmd, finger_axis=finger_axis)
        result = self.move_joints(np.concatenate([q6, [self.q_cmd[6]]]), settle=settle)
        for _ in range(correct if self.execute else 0):
            if result.aborted:
                break
            result = self._correct(pos, R, settle)
        return result, R, tilt

    def move_topdown(self, pos: np.ndarray, yaw: float = 0.0, correct: int = 1, settle: float = 0.3) -> Tuple[MoveResult, np.ndarray]:
        """Vertical grasp pose at ``pos``; returns (result, R actually used) so a following
        move_linear can keep the same wrist branch."""
        pos = np.asarray(pos, dtype=np.float64)
        if not self.workspace.contains(pos):
            raise ValueError(f"[{self.side}] target {np.round(pos, 3)} outside workspace {self.workspace}")
        q6, R = self.kin.ik_topdown(pos, yaw, self.q_cmd)
        result = self.move_joints(np.concatenate([q6, [self.q_cmd[6]]]), settle=settle)
        for _ in range(correct if self.execute else 0):
            if result.aborted:
                break
            result = self._correct(pos, R, settle)
        return result, R

    def move_linear(
        self,
        pos: np.ndarray,
        R: np.ndarray,
        speed: float = 0.08,
        step: float = 0.01,
        correct: int = 1,
        track_abort_rad: Optional[float] = None,
    ) -> MoveResult:
        """Straight-line Cartesian move of the grasp site at ``speed`` m/s, as a chain of IK
        waypoints ``step`` apart streamed without stopping in between. ``correct`` rounds of
        residual compensation follow (see move_grasp). ``track_abort_rad`` tightens the
        tracking-lag abort for this move -- a slow descent that meets something stalls the
        wrist joints long before the default limit, and pushing on is what shoves objects."""
        saved = self.track_abort_rad
        if track_abort_rad is not None:
            self.track_abort_rad = float(track_abort_rad)
        try:
            result = self._move_linear(pos, R, speed, step)
        finally:
            self.track_abort_rad = saved
        for _ in range(correct if self.execute else 0):
            if result.aborted:
                break
            result = self._correct(np.asarray(pos, dtype=np.float64), R, 0.1)
        return result

    def _move_linear(self, pos: np.ndarray, R: np.ndarray, speed: float, step: float) -> MoveResult:
        pos = np.asarray(pos, dtype=np.float64)
        if not self.workspace.contains(pos):
            raise ValueError(f"[{self.side}] target {np.round(pos, 3)} outside workspace {self.workspace}")
        p0, R0 = self.kin.grasp(self.q_cmd)
        dist = float(np.linalg.norm(pos - p0))
        n = max(2, int(np.ceil(dist / step)) + 1)
        seconds = max(dist / speed, self.send_dt)
        # pre-solve all waypoints so IK never stalls the stream
        q_prev = self.q_cmd.copy()
        qs = []
        for i in range(n):
            s = i / (n - 1)
            p = p0 + (pos - p0) * s
            q6, perr, rerr = self.kin.ik(p, R, q_prev)
            if perr > 5e-3 or rerr > 0.05:
                raise ValueError(f"[{self.side}] IK failed on waypoint {i} at {np.round(p, 3)}: {perr * 1e3:.1f} mm")
            q_prev = np.concatenate([q6, [q_prev[6]]])
            qs.append(q_prev)
        qs_arr = np.stack(qs)
        t0 = time.monotonic()
        base_lag = np.abs(self.q()[:6] - self.q_cmd[:6]) if self.execute else np.zeros(6)
        result = MoveResult(reached=False, max_err=0.0, seconds=seconds)
        while True:
            s = min(1.0, (time.monotonic() - t0) / seconds)
            u = s * (n - 1)
            lo = int(np.floor(u))
            hi = min(lo + 1, n - 1)
            q = qs_arr[lo] * (1 - (u - lo)) + qs_arr[hi] * (u - lo)
            self._send(q)
            if self.execute:
                lag = np.abs(self.q()[:6] - self.q_cmd[:6]) - base_lag
                if np.max(lag) > self.track_abort_rad:
                    result.aborted = f"joint {int(np.argmax(lag)) + 1} lags by {np.max(lag):.2f} rad beyond its baseline"
                    logging.error(f"[{self.side}] linear move aborted: {result.aborted}")
                    self._send(self.q())
                    return result
            if s >= 1.0:
                break
            time.sleep(self.send_dt)
        time.sleep(0.05)
        if self.execute:
            p_now, _ = self.grasp_pose()
            result.max_err = float(np.linalg.norm(p_now - pos))
            result.reached = result.max_err < 0.01
        else:
            result.reached = True
        return result

    def set_gripper(self, value: float, wait: float = 0.8) -> None:
        """0 = closed, 1 = open. Streams the ramp at the gripper slew limit, then waits."""
        target = self.q_cmd.copy()
        target[6] = float(np.clip(value, 0.0, 1.0))
        self.move_joints(target, settle=wait)

    def wait_settled(self, tol_rad: float = 0.04, timeout: float = 0.8) -> float:
        """Block until every arm joint is within ``tol_rad`` of the last command (or the
        timeout passes). Streamed moves return when the *command* ramp ends; the arm itself
        lags by up to a few cm at 0.6 rad/s, and a joint move started from that lagging pose
        can swing through space the plan never visited -- the rim of a box, say."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            lag = float(np.max(np.abs(self.q()[:6] - self.q_cmd[:6])))
            if lag < tol_rad:
                return lag
            time.sleep(0.02)
        return lag

    def grip_until_contact(
        self,
        target: float = 0.0,
        lag_limit: float = 0.035,
        squeeze: float = 0.012,
        step: float = 0.02,
        dt: float = 0.03,
        hold_wait: float = 0.05,
    ) -> Dict[str, float]:
        """Close on whatever is between the fingers instead of clamping shut.

        The gripper is a position-controlled joint with a fixed gain, so the gap between what
        it is told and where it actually is *is* the squeezing force: commanding 0 on a rigid
        object means the full stroke's worth of error, which is what crushes things. Here the
        command walks in a step at a time and stops as soon as the fingers fall ``lag_limit``
        behind it -- that is contact -- then holds ``squeeze`` tighter than where they stopped,
        which is a small, repeatable force whatever the object's width.

        ``hold_wait`` is how long it waits after commanding the squeeze before reading the
        width back. That read only feeds the line this prints, but the arm is stationary for
        all of it, right between gripping the object and lifting it -- which is where a pause
        is most visible. It is short on purpose: the lift that follows starts slowly anyway,
        so the squeeze has as long as it ever did to build up, it just builds up while the
        arm is already moving.

        Returns the measured width (normalised and in metres), the command held, and whether
        anything was actually found.
        """
        cmd = float(self.q_cmd[6])
        while True:
            cmd = max(target, cmd - step)
            q = self.q_cmd.copy()
            q[6] = cmd
            self._send(q)
            time.sleep(dt)
            meas = float(self.q()[6]) if self.execute else cmd
            if meas - cmd > lag_limit:
                hold = max(target, meas - squeeze)
                q[6] = hold
                self._send(q)
                time.sleep(hold_wait)
                held = float(self.q()[6]) if self.execute else hold
                return {"contacted": 1.0, "width": held, "width_m": held * 0.096, "hold": hold}
            if cmd <= target + 1e-6:
                time.sleep(hold_wait)
                meas = float(self.q()[6]) if self.execute else cmd
                return {"contacted": 0.0, "width": meas, "width_m": meas * 0.096, "hold": cmd}

    def hold(self) -> None:
        """Re-send the current measured pose as the command (stop wherever we are)."""
        self._send(self.q())

    def close(self) -> None:
        self.rpc.close()
