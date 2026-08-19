#!/usr/bin/env bash
# RIGHT FOLLOWER arm (linear_4310 gripper): bring it up on its own CAN bus in gravity
# compensation, to verify it is powered, cabled and talking. Do not run while teleop runs.
#
#   scripts/run_follower_right.sh
#   scripts/run_follower_right.sh --operation_mode stay_current_qpos
#   scripts/run_follower_right.sh --operation_mode test_gripper
#   scripts/run_follower_right.sh --channel canN     # override the channel from scripts/can_map.conf

ARM_LABEL="follower-right"
ARM_ROLE="follower_right"   # the CAN netdev is looked up in scripts/can_map.conf
ARM_GRIPPER="linear_4310"

source "$(dirname "$(readlink -f "$0")")/_yam_arm_common.sh"
run_arm "$@"
