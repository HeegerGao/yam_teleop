#!/usr/bin/env bash
# Wipe the persistent CAN interface names and go back to kernel defaults (can0, can1, ...),
# so the naming scheme can be rebuilt from scratch. Run with sudo.
#
#     sudo scripts/clear_can_udev.sh                 # remove the naming rules, re-enumerate
#     sudo scripts/clear_can_udev.sh --no-reload     # remove the rules only (names change on replug)
#     sudo scripts/clear_can_udev.sh --disable-flow-base-can-up
#     scripts/clear_can_udev.sh --dry-run            # show what would happen, no root needed
#
# What it touches:
#   /etc/udev/rules.d/90-can.rules      the only rule that NAMEs CAN interfaces -- removed
#   /etc/udev/rules.d/90-can.rules.{old,bak}  inert (udev only reads *.rules) -- removed too
#   /etc/udev/rules.d/flow_base.rules   does NOT name anything, but its last block brings every
#                                       gs_usb interface-00 netdev up at 1 Mbit -- which is how
#                                       the spare channel of a dual-channel adapter keeps coming
#                                       back UP and stealing replies. Left alone unless you pass
#                                       --disable-flow-base-can-up (its other blocks, driver
#                                       autoload and new_id binding, are always kept).
#
# Everything removed or edited is copied to a timestamped backup directory first, printed at
# the end -- restore with: sudo cp <backup>/*.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules

set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/_can_map.sh"

RULES_DIR=/etc/udev/rules.d
NAME_RULES=("$RULES_DIR/90-can.rules" "$RULES_DIR/90-can.rules.old" "$RULES_DIR/90-can.rules.bak")
FLOW_RULES="$RULES_DIR/flow_base.rules"

DRY_RUN=0
RELOAD=1
DISABLE_FLOW=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --no-reload) RELOAD=0 ;;
        --disable-flow-base-can-up) DISABLE_FLOW=1 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

if (( ! DRY_RUN )) && [[ "$(id -u)" != "0" ]]; then
    echo "needs root: sudo $0 $*" >&2
    exit 1
fi

if pgrep -f "minimum_gello|motor_chain_robot|bimanual_teleop" >/dev/null; then
    echo "teleop / motor_chain_robot is running -- stop it first" >&2
    exit 1
fi

run() { if (( DRY_RUN )); then echo "    would run: $*"; else "$@"; fi; }

echo "=== current CAN netdevs (record this before wiping) ==="
for dev in /sys/class/net/can*; do
    [[ -e "$dev" ]] || continue
    name="$(basename "$dev")"
    serial="$(cat "$(readlink -f "$dev/device")/../serial" 2>/dev/null || echo '?')"
    state="$(ip -brief link show "$name" | grep -o ' UP \| DOWN ' | tr -d ' ')"
    printf "  %-16s %-26s %s\n" "$name" "$serial" "$state"
done

BACKUP="/var/backups/i2rt-can-udev-$(date +%Y%m%d-%H%M%S)"
echo
echo "=== backup -> $BACKUP ==="
run mkdir -p "$BACKUP"
for f in "${NAME_RULES[@]}" "$FLOW_RULES"; do
    [[ -f "$f" ]] && run cp -a "$f" "$BACKUP/"
done

echo
echo "=== removing the naming rules ==="
for f in "${NAME_RULES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "  $f"
        run rm -f "$f"
    fi
done

if (( DISABLE_FLOW )) && [[ -f "$FLOW_RULES" ]]; then
    echo
    echo "=== commenting out the auto-up block in $FLOW_RULES ==="
    # The block starts at the 'Configure CAN interface' comment and runs to the end of its
    # line continuations; prefixing every one of its lines with '#' disables it in place.
    run sed -i '/^# Configure CAN interface when detected/,/ip link set \$name up"/ s/^/#/' "$FLOW_RULES"
fi

echo
echo "=== reloading udev rules ==="
run udevadm control --reload-rules
run udevadm trigger --subsystem-match=net --action=add

if (( RELOAD )); then
    echo
    echo "=== re-enumerating the gs_usb adapters (this also bounces the Flow Base adapter) ==="
    run modprobe -r gs_usb
    sleep 1
    run modprobe gs_usb
    sleep 2
fi

echo
echo "=== result ==="
for dev in /sys/class/net/can*; do
    [[ -e "$dev" ]] || continue
    name="$(basename "$dev")"
    serial="$(cat "$(readlink -f "$dev/device")/../serial" 2>/dev/null || echo '?')"
    ifnum="$(cat "$(readlink -f "$dev/device")/bInterfaceNumber" 2>/dev/null || echo '?')"
    state="$(ip -brief link show "$name" | grep -o ' UP \| DOWN ' | tr -d ' ')"
    printf "  %-16s serial=%-26s usb_if=%-3s %s\n" "$name" "$serial" "$ifnum" "$state"
done
echo
echo "backup kept at: $BACKUP"
echo "names are now kernel defaults (can0, can1, ...), which is what every script here expects."
echo "The arm -> channel mapping lives in ONE table, $CAN_MAP_FILE; it currently says:"
can_map_roles | sed 's/^/  /' || true
echo "If the numbering above does NOT match, edit that table -- nothing else duplicates it."
echo "Then run: sudo scripts/fix_can_links.sh && python scripts/check_arms.py"
