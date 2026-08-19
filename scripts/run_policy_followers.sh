#!/usr/bin/env bash
# BOTH follower arms as RPC servers, kept alive independently of any policy run.
#
# scripts/box_folding_policy_rollout.py normally spawns these itself, and tears them down when
# it exits -- which sets every motor torque to zero (MotorChainRobot.close), so the arms go limp
# and drop at the end of every run. Start them here instead and the arms stay powered between
# runs, holding the last command under gravity compensation:
#
#   scripts/run_policy_followers.sh                        # terminal 1, leave it running
#   python scripts/box_folding_policy_rollout.py --no-launch --execute   # terminal 2, repeatable
#
# Ctrl-C here is what finally disables the motors, so park the arms low before you do it.
# Do not run this while bimanual_teleop_record.py is running: both want the same CAN buses and
# the same ports.
#
# Ports match the rollout's --port-left / --port-right defaults (1235 / 1234). CAN netdevs come
# from scripts/can_map.conf, the one mapping table.
#
# Environment overrides:
#   YAM_PYTHON   interpreter (default: <repo>/.venv/bin/python)
#   YAM_ARM      arm variant (default: yam)
#   YAM_VERSION  arm hardware revision (default: 1)
#   YAM_GRIPPER  gripper (default: linear_4310)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${YAM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
GELLO="$REPO_ROOT/examples/minimum_gello/minimum_gello.py"
ARM="${YAM_ARM:-yam}"
VERSION="${YAM_VERSION:-1}"
GRIPPER="${YAM_GRIPPER:-linear_4310}"
PORT_LEFT=1235
PORT_RIGHT=1234

source "$SCRIPT_DIR/_can_map.sh"

if [[ ! -x "$PYTHON" ]]; then
    echo "[followers] python not found at $PYTHON -- set YAM_PYTHON=/path/to/python" >&2
    exit 1
fi

port_busy() {
    if (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; then
        return 0
    fi
    return 1
}

for port in "$PORT_LEFT" "$PORT_RIGHT"; do
    if port_busy "$port"; then
        echo "[followers] port $port is already served -- another follower (or a rollout that" >&2
        echo "[followers] launched its own) is running. Starting here would die on bind, and a" >&2
        echo "[followers] rollout would then drive that other process instead of these arms." >&2
        ss -ltnp 2>/dev/null | grep ":$port " >&2 || true
        exit 1
    fi
done

pids=()
cleanup() {
    echo
    echo "[followers] stopping -- motors are about to be disabled, the arms will go limp"
    for pid in "${pids[@]}"; do
        kill -INT "$pid" 2>/dev/null || true
    done
    # SIGINT alone is not enough: a follower stuck in teardown keeps the port bound, and the next
    # run then attaches to a dead server instead of the arms. Escalate rather than leave that.
    local deadline=$((SECONDS + 8)) alive=1
    while (( SECONDS < deadline )); do
        alive=0
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=1
            fi
        done
        (( alive )) || break
        sleep 0.5
    done
    if (( alive )); then
        echo "[followers] some followers did not exit in time -- killing them" >&2
        for pid in "${pids[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

start_one() {
    local role="$1" port="$2" channel
    if ! channel="$(can_channel_for "$role")"; then
        echo "[followers] no CAN channel for role $role -- check $CAN_MAP_FILE" >&2
        exit 1
    fi
    if [[ ! -e "/sys/class/net/$channel" ]]; then
        echo "[followers] CAN interface $channel does not exist -- adapter unplugged, or the" >&2
        echo "[followers] numbering moved: re-check it and edit $CAN_MAP_FILE." >&2
        echo "[followers] After a reset: sudo scripts/reset_all_can.sh" >&2
        exit 1
    fi
    echo "[followers] $role on $channel, RPC port $port"
    "$PYTHON" "$GELLO" --arm "$ARM" --version "$VERSION" --gripper "$GRIPPER" \
        --can_channel "$channel" --server_port "$port" &
    pids+=($!)
}

start_one follower_right "$PORT_RIGHT"
start_one follower_left "$PORT_LEFT"

echo "[followers] both arms are coming up (gripper calibration runs first, a few seconds)."
echo "[followers] Then run:  python scripts/box_folding_policy_rollout.py --no-launch --execute"
echo "[followers] Ctrl-C here disables the motors -- park the arms low first."
wait
