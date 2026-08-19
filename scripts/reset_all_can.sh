#!/bin/bash

if [ "$(id -u)" != "0" ]; then
    SUDO="sudo"
else
    SUDO=""
fi

# Function to reset a CAN interface
reset_can_interface() {
    local iface=$1
    echo "Resetting CAN interface: $iface"
    $SUDO ip link set "$iface" down
    $SUDO ip link set "$iface" up type can bitrate 1000000
}

# Get all CAN interfaces
can_interfaces=$(ip link show | grep -oP '(?<=: )(can\w+)')

# Check if any CAN interfaces were found
if [[ -z "$can_interfaces" ]]; then
    echo "No CAN interfaces found."
    exit 1
fi

# Reset each CAN interface
echo "Detected CAN interfaces: $can_interfaces"
for iface in $can_interfaces; do
    reset_can_interface "$iface"
done

echo "All CAN interfaces have been reset with bitrate 1000000."

# Every netdev is UP again at this point, and each dual-channel adapter exposes TWO of them on
# the SAME physical bus: frames then get attributed to either channel index, so a single socket
# sees only part of the motors' replies (and its TX queue jams). Hand over to fix_can_links.sh,
# which leaves exactly one netdev per adapter up -- the channel each arm is assigned in
# scripts/can_map.conf (the one mapping table; edit it if the numbering moved).
FIX="$(dirname "$(readlink -f "$0")")/fix_can_links.sh"
if [[ -x "$FIX" ]]; then
    echo
    $SUDO "$FIX" "$@"
else
    echo "note: $FIX not found -- the spare netdev of each adapter is left UP and will steal replies" >&2
    echo "note: bring each adapter's un-cabled sibling netdev down by hand before starting teleop" >&2
fi
