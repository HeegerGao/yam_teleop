"""Touch-free check of the calibrated table height: hover the closed gripper over an empty
patch of table and measure, with the top camera's depth, the real gap between the fingertips
and the table. The difference to the FK-predicted gap is the z bias of the calibration.

    python scripts/box_packing/check_table_z.py --pixel 480 500 --hover 0.04
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import tyro

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from arm import Arm, Workspace  # noqa: E402
from calibrate_top import arm_mask, median_depth  # noqa: E402
from cameras import CameraRig  # noqa: E402
from perception import Scene  # noqa: E402

PORTS = {"left": 1235, "right": 1234}


@dataclass
class Args:
    side: str = "left"
    pixel: Tuple[int, int] = (480, 500)
    """Top-image pixel of an empty patch of table."""
    hover: float = 0.04
    """Commanded fingertip height above the calibrated table."""
    transit_clear: float = 0.16
    clear_radius: float = 0.08
    """The patch must have nothing above the table within this radius (m)."""
    execute: bool = True
    auto: bool = True
    """Ignore --pixel and search the emptiest spot in the reachable area instead."""


def main(args: Args) -> None:
    scene = Scene.from_calib(args.side)
    calib = json.loads((_HERE / "calib" / f"top_{args.side}.json").read_text())
    n_cam, d_cam = np.array(calib["table_normal_cam"]), float(calib["table_d_cam"])
    debug = _HERE / "calib" / "debug"
    rig = CameraRig.open(roles=("top",))
    arm = None
    try:
        bg = median_depth(rig, "top")
        pts_cam, _ = bg.point_cloud((bg.depth > 0.3) & (bg.depth < 1.5))
        pts = scene.to_base(pts_cam)
        if args.auto:
            # the emptiest reachable spot: farthest from anything standing above the table
            above = pts[scene.height(pts) > 0.012][:, :2]
            from scipy.spatial import cKDTree

            tree = cKDTree(above)
            best = None
            for x in np.arange(0.24, 0.46, 0.02):
                for y in np.arange(-0.12, 0.16, 0.02):
                    dist, _ = tree.query([x, y])
                    if best is None or dist > best[0]:
                        best = (dist, x, y)
            print(f"[check] emptiest spot ({best[1]:.2f}, {best[2]:.2f}), nearest object {best[0] * 100:.1f} cm away")
            spot = np.array([best[1], best[2], scene.table_z(best[1], best[2])])
        else:
            spot = scene.pixel_to_table(bg, *args.pixel)
            print(f"[check] pixel {args.pixel} -> table point {np.round(spot, 3)} (base frame)")
        # is the patch empty?  points above the table near the spot
        near = pts[np.linalg.norm(pts[:, :2] - spot[:2], axis=1) < args.clear_radius]
        hgt = scene.height(near)
        print(f"[check] {len(near)} table points within {args.clear_radius:.2f} m; height 99th pct {np.percentile(hgt, 99) * 100:.1f} cm, max {hgt.max() * 100:.1f} cm")
        if np.percentile(hgt, 99) > 0.012:
            raise SystemExit("[check] the patch is not empty -- choose another pixel")

        table_z = scene.table_z(spot[0], spot[1])
        target = np.array([spot[0], spot[1], table_z + args.hover])
        transit = np.array([spot[0], spot[1], table_z + args.transit_clear])
        print(f"[check] table z {table_z:.4f}; hover target {np.round(target, 3)}")
        arm = Arm(args.side, PORTS[args.side], max_joint_vel=0.6, workspace=Workspace(z=(table_z + 0.02, 0.6)), execute=args.execute)
        q_home = arm.q().copy()
        if not args.execute:
            for p in (transit, target):
                q6, R, tilt = arm.kin.ik_reach(p, q_home)
                print(f"   ik ok {np.round(p, 3)} tilt {tilt:.0f} q {np.round(q6, 2)}")
            return
        arm.set_gripper(0.0)
        res, R, tilt = arm.move_reach(transit)
        print(f"[move] transit tilt {tilt:.0f}: at {np.round(arm.grasp_pose()[0], 3)} {res.aborted}")
        res = arm.move_linear(target, R, speed=0.05)
        p_fk = arm.grasp_pose()[0]
        print(f"[move] hover: fk fingertip {np.round(p_fk, 3)}, predicted gap {(p_fk[2] - table_z) * 100:.1f} cm {res.aborted}")
        time.sleep(0.5)
        frame = median_depth(rig, "top")
        mask = arm_mask(frame, bg.depth)
        uv = frame.project(scene.to_cam(p_fk))[0]
        yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
        win = mask & ((xx - uv[0]) ** 2 + (yy - uv[1]) ** 2 < 70**2)
        p_cam, _ = frame.point_cloud(win)
        h = -(p_cam @ n_cam + d_cam)  # height above the table, camera-frame plane
        print(f"[check] {len(p_cam)} arm points near the predicted tip; height pcts 1/5/50: {np.round(np.percentile(h, [1, 5, 50]) * 100, 1)} cm")
        low = float(np.percentile(h, 2))
        bias = (p_fk[2] - table_z) - low
        print(f"[check] measured tip gap {low * 100:.1f} cm vs predicted {(p_fk[2] - table_z) * 100:.1f} cm -> calibration table z bias {bias * 100:+.1f} cm (positive: real table is lower)")
        overlay = frame.color.copy()
        overlay[win] = (0.5 * overlay[win] + np.array([0, 0, 127])).astype(np.uint8)
        cv2.circle(overlay, (int(uv[0]), int(uv[1])), 8, (0, 255, 0), 2)
        cv2.imwrite(str(debug / "table_z_check.png"), overlay)
        (_HERE / "calib" / "table_z_check.json").write_text(json.dumps({"bias_m": bias, "spot": spot.tolist(), "hover": args.hover}))
        arm.move_linear(transit, R, speed=0.08)
    finally:
        if arm is not None and args.execute:
            arm.move_joints(q_home)
            arm.set_gripper(1.0)
        if arm is not None:
            arm.close()
        rig.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
