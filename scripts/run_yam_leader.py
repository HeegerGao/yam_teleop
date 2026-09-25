import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_offsets import offsets_for_channel

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType

log_level = os.environ.get("LOGLEVEL", "INFO")
logging.basicConfig(level=log_level)


args = argparse.ArgumentParser()
args.add_argument("--channel", type=str, default="can0")
args = args.parse_args()


robot = get_yam_robot(
    channel=args.channel,
    gripper_type=GripperType.YAM_TEACHING_HANDLE,
    joint_offsets=offsets_for_channel(args.channel),  # scripts/arm_offsets.conf, by CAN channel
)
motor_chain = robot.motor_chain

while True:
    obs = robot.get_observations()
    encoder_obs = motor_chain.get_same_bus_device_states()
    qpos = obs["joint_pos"]
    print(qpos)
    print(encoder_obs)
    time.sleep(0.01)
