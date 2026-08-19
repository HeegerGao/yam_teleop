#!/usr/bin/env bash
# Shared implementation behind the four per-arm launchers (run_follower_left.sh, ...).
#
# Not meant to be run directly: each launcher sets ARM_LABEL / ARM_ROLE / ARM_GRIPPER,
# sources this file, and calls `run_arm "$@"`.
#
# ARM_ROLE is one of leader_left / leader_right / follower_left / follower_right; the CAN
# netdev it maps to is looked up in scripts/can_map.conf, the one table that assigns arms to
# buses (see that file's header, and scripts/_can_map.sh for the parser). No udev persistent
# names and no serial-based auto-resolution any more: both kept binding to the un-cabled
# channel of the dual-channel adapters. run_arm just launches
# i2rt/robots/motor_chain_robot.py on that channel in gravity-compensation mode.
#
# Environment overrides:
#   YAM_PYTHON        interpreter to use (default: <repo>/.venv/bin/python)
#   YAM_CAN_CHANNEL   use this channel instead of the one the table assigns
#   YAM_CAN_MAP       read a different mapping table (default: scripts/can_map.conf)
#   YAM_ARM           arm variant (default: yam)
#   YAM_VERSION       arm hardware revision (default: 1)
#   YAM_GRIPPER       gripper override, e.g. no_gripper to bring an arm up without its gripper
#                     motor (useful when the gripper motor itself is the thing that is broken)
# Any extra command-line arguments are forwarded to motor_chain_robot.py, e.g.
#   --operation_mode stay_current_qpos | test_gripper     --record

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${YAM_PYTHON:-$REPO_ROOT/.venv/bin/python}"

source "$SCRIPT_DIR/_can_map.sh"

run_arm() {
    if [[ ! -x "$PYTHON" ]]; then
        echo "[$ARM_LABEL] python not found at $PYTHON -- set YAM_PYTHON=/path/to/python" >&2
        exit 1
    fi

    # An explicit --channel on the command line wins: skip resolution and pass it through once.
    local arg user_channel=0
    for arg in "$@"; do
        [[ "$arg" == "--channel" || "$arg" == --channel=* ]] && user_channel=1
    done

    local channel_args=()
    if (( user_channel )); then
        echo "[$ARM_LABEL] using the --channel given on the command line"
    else
        local channel
        if [[ -n "${YAM_CAN_CHANNEL:-}" ]]; then
            channel="$YAM_CAN_CHANNEL"
        elif ! channel="$(can_channel_for "$ARM_ROLE")"; then
            echo "[$ARM_LABEL] no CAN channel for role $ARM_ROLE -- check $CAN_MAP_FILE" >&2
            exit 1
        fi
        if [[ ! -e "/sys/class/net/$channel" ]]; then
            echo "[$ARM_LABEL] CAN interface $channel does not exist -- USB-CAN adapter unplugged, or the" >&2
            echo "[$ARM_LABEL] numbering moved: re-check it and edit $CAN_MAP_FILE (recipe in its header)." >&2
            echo "[$ARM_LABEL] After a reset: sudo scripts/reset_all_can.sh" >&2
            exit 1
        fi
        echo "[$ARM_LABEL] CAN channel: $channel (from $CAN_MAP_FILE)"
        channel_args=(--channel "$channel")
    fi

    local gripper="${YAM_GRIPPER:-$ARM_GRIPPER}"
    echo "[$ARM_LABEL] arm ${YAM_ARM:-yam} v${YAM_VERSION:-1}, gripper $gripper"
    echo "[$ARM_LABEL] starting motor_chain_robot -- the arm becomes gravity-compensated (it will move"
    echo "[$ARM_LABEL] freely by hand and hold itself up). Keep the workspace clear; Ctrl-C to stop."
    exec "$PYTHON" "$REPO_ROOT/i2rt/robots/motor_chain_robot.py" \
        --arm "${YAM_ARM:-yam}" \
        --version "${YAM_VERSION:-1}" \
        --gripper "$gripper" \
        ${channel_args[@]+"${channel_args[@]}"} \
        "$@"
}
