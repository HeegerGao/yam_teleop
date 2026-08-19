#!/usr/bin/env bash
# Shell access to the one CAN mapping table, scripts/can_map.conf.
#
# Not meant to be run directly -- source it:
#
#     source "$(dirname "$(readlink -f "$0")")/_can_map.sh"
#     can_map_roles                  # prints "<role> <netdev>" per arm, in role order
#     can_channel_for follower_left  # prints "can4"
#
# The table is parsed, never duplicated: scripts/can_map.conf is the only place the arm ->
# channel assignment is written down, and scripts/can_channels.py reads the same file for
# the Python entry points. Edit that table after a reboot/replug moves the numbering; the
# recipe for re-deriving it is in the table's own header.
#
# Override the table's location with YAM_CAN_MAP=/path/to/other.conf (for a second rig).

CAN_MAP_FILE="${YAM_CAN_MAP:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/can_map.conf}"
CAN_MAP_ARMS="leader_left leader_right follower_left follower_right"

# "<role> <netdev>" per arm, in the fixed role order above (not file order, so callers get a
# stable ordering). Exits non-zero with a message on stderr if the table is malformed.
can_map_roles() {
    if [[ ! -r "$CAN_MAP_FILE" ]]; then
        echo "cannot read the CAN mapping table $CAN_MAP_FILE" >&2
        return 1
    fi
    awk -v arms="$CAN_MAP_ARMS" -v file="$CAN_MAP_FILE" '
        BEGIN { n = split(arms, order, " "); for (i = 1; i <= n; i++) known[order[i]] = 1 }
        { sub(/#.*/, "") }
        NF == 0 { next }
        NF != 2 { printf "%s:%d: expected \"<can number> <arm role>\", got \"%s\"\n", file, NR, $0 > "/dev/stderr"; bad = 1; next }
        {
            channel = ($1 ~ /^can/) ? $1 : "can" $1
            if (channel !~ /^can[0-9]+$/) { printf "%s:%d: \"%s\" is not a CAN netdev number (e.g. 4 or can4)\n", file, NR, $1 > "/dev/stderr"; bad = 1; next }
            if (!($2 in known)) { printf "%s:%d: unknown arm role \"%s\"\n", file, NR, $2 > "/dev/stderr"; bad = 1; next }
            if ($2 in chan) { printf "%s:%d: arm \"%s\" is assigned twice\n", file, NR, $2 > "/dev/stderr"; bad = 1; next }
            if (channel in arm) { printf "%s:%d: %s is assigned to two arms\n", file, NR, channel > "/dev/stderr"; bad = 1; next }
            chan[$2] = channel; arm[channel] = $2
        }
        END {
            for (i = 1; i <= n; i++) {
                role = order[i]
                if (!(role in chan)) { printf "%s: no channel for %s -- all four arms must be in the table\n", file, role > "/dev/stderr"; bad = 1; continue }
                printf "%s %s\n", role, chan[role]
            }
            exit bad ? 1 : 0
        }
    ' "$CAN_MAP_FILE"
}

# The netdev cabled to $1, e.g. `can_channel_for leader_left` -> can2.
can_channel_for() {
    local want="$1" entries role channel
    entries="$(can_map_roles)" || return 1
    while read -r role channel; do
        [[ "$role" == "$want" ]] && { echo "$channel"; return 0; }
    done <<<"$entries"
    echo "no channel for arm \"$want\" in $CAN_MAP_FILE (expected one of $CAN_MAP_ARMS)" >&2
    return 1
}
