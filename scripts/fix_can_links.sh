#!/usr/bin/env bash
# Put the four arm CAN buses in their working state: each arm's fixed netdev UP at 1 Mbit, the
# adapter's other (un-cabled) netdev DOWN. Run with sudo, after any replug or reset:
#
#     sudo scripts/fix_can_links.sh              # all four arms
#     sudo scripts/fix_can_links.sh follower_right
#
# The arm -> channel mapping is read from scripts/can_map.conf -- the one table, shared with
# the Python entry points. This script does not carry a copy: edit that table if the numbering
# moved (its header has the recipe for re-deriving it).
#
# Nothing is renamed and no udev persistent name is used any more: the rule matched on adapter
# serial only, every adapter is dual-channel, and the name kept landing on the un-cabled netdev.
#
# What still has to happen per adapter: only ONE of its two netdevs may be UP. While the
# un-cabled sibling is UP it absorbs part of the motors' replies, so the arm looks half-dead on
# both channels. The sibling is found by USB serial (which netdev sits on the same adapter) --
# that is a physical-pairing lookup, not an identity one, so it does not depend on any
# particular serial value.
#
# Read-only preview (no changes, no root needed):  scripts/fix_can_links.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${YAM_PYTHON:-$REPO_ROOT/.venv/bin/python}"
BITRATE=1000000
MOTORS=6

source "$SCRIPT_DIR/_can_map.sh"

# "<role> <netdev>" per arm, straight out of scripts/can_map.conf -- no copy of the table here.
ROLES_TEXT="$(can_map_roles)" || { echo "bad CAN mapping table $CAN_MAP_FILE (see above)" >&2; exit 1; }
ROLES=()
mapfile -t ROLES <<<"$ROLES_TEXT"

DRY_RUN=0
WANTED=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) WANTED+=("$arg") ;;
    esac
done

if (( ! DRY_RUN )) && [[ "$(id -u)" != "0" ]]; then
    echo "needs root to change interfaces: sudo $0 $*" >&2
    exit 1
fi

if pgrep -f "minimum_gello|motor_chain_robot|bimanual_teleop" >/dev/null; then
    echo "teleop / motor_chain_robot is running -- stop it first (this script bounces the CAN links)" >&2
    exit 1
fi

run() {
    if (( DRY_RUN )); then
        echo "    would run: $*"
    else
        "$@"
    fi
}

serial_of() {
    cat "$(readlink -f "/sys/class/net/$1/device")/../serial" 2>/dev/null || true
}

# the other netdev(s) of the same dual-channel adapter as $1
siblings_of() {
    local target="$1" serial name dev
    serial="$(serial_of "$target")"
    [[ -n "$serial" ]] || return 0
    for dev in /sys/class/net/can*; do
        name="$(basename "$dev")"
        [[ "$name" == "$target" ]] && continue
        [[ "$(serial_of "$name")" == "$serial" ]] && echo "$name"
    done
}

fix_role() {
    local role="$1" want="$2"
    if [[ ! -e "/sys/class/net/$want" ]]; then
        echo "[$role] $want does not exist -- USB-CAN adapter unplugged, or the channel numbering"
        echo "[$role] moved (re-check it and edit $CAN_MAP_FILE; the recipe is in its header)"
        return 1
    fi

    local siblings=() s
    mapfile -t siblings < <(siblings_of "$want")
    if (( ${#siblings[@]} )); then
        echo "[$role] $want, sibling(s) on the same adapter: ${siblings[*]} -> down"
        for s in "${siblings[@]}"; do
            run ip link set "$s" down
        done
    else
        echo "[$role] $want (no sibling netdev found)"
    fi

    run ip link set "$want" down
    run ip link set "$want" up type can bitrate "$BITRATE"

    local n
    n="$("$PYTHON" "$REPO_ROOT/scripts/resolve_can.py" --channel "$want" --motors "$MOTORS" || echo 0)"
    [[ "$n" =~ ^[0-9]+$ ]] || n=0
    if (( DRY_RUN )); then
        echo "[$role] $want: $n/$MOTORS motors (probed as-is; nothing changed)"
        return 0
    fi
    if (( n == 0 )); then
        echo "[$role] $want: no motors answered -- arm unpowered / e-stop / CAN cable?"
        echo "[$role] if this arm is definitely powered, the channel numbering may have moved:"
        echo "[$role] probe the others with scripts/resolve_can.py --channel canN, then edit $CAN_MAP_FILE"
        return 1
    fi

    # Re-probe a second later. A bus that answers right after the bounce and then stops is a
    # flaky connector / power problem: once the motors go quiet the TX queue jams (ENOBUFS)
    # until the next link bounce, so a single probe makes it look fine.
    sleep 1
    local again
    again="$("$PYTHON" "$REPO_ROOT/scripts/resolve_can.py" --channel "$want" --motors "$MOTORS" || echo 0)"
    [[ "$again" =~ ^[0-9]+$ ]] || again=0
    echo "[$role] $want: $n/$MOTORS motors on bring-up, $again/$MOTORS one second later"
    if (( again < n )); then
        echo "[$role] UNSTABLE: the bus goes quiet after bring-up -- check the arm's power/e-stop,"
        echo "[$role] the CAN connector at the adapter and at the base motor, and the terminator."
        return 1
    fi
    echo "[$role] OK: $want, $n/$MOTORS motors answering"
}

status=0
for entry in "${ROLES[@]}"; do
    read -r role want <<<"$entry"
    if (( ${#WANTED[@]} )) && [[ " ${WANTED[*]} " != *" $role "* ]]; then
        continue
    fi
    fix_role "$role" "$want" || status=1
done

echo
echo "current state:"
ip -brief link show | grep -E '^can' || true
echo
echo "verify with: $PYTHON scripts/check_arms.py --motors 1 2 3 4 5 6 7"
exit "$status"
