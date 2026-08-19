#!/usr/bin/env bash
# LEFT LEADER arm (yam_teaching_handle -- passive handle, no powered gripper): bring it up on
# its own CAN bus in gravity compensation, to verify it is powered, cabled and talking.
# Do not run while teleop runs.
#
#   scripts/run_leader_left.sh
#   scripts/run_leader_left.sh --operation_mode stay_current_qpos
#   scripts/run_leader_left.sh --channel canN     # override the channel from scripts/can_map.conf
#
# --operation_mode test_gripper does not apply here (the teaching handle is passive).

ARM_LABEL="leader-left"
ARM_ROLE="leader_left"   # the CAN netdev is looked up in scripts/can_map.conf
ARM_GRIPPER="yam_teaching_handle"

source "$(dirname "$(readlink -f "$0")")/_yam_arm_common.sh"
run_arm "$@"
