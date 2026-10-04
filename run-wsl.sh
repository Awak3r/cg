#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# WSLg may start lazily. An app launched before its Windows connection is
# established can keep running without ever receiving a visible window.
if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    ready=0
    for _ in {1..30}; do
        # Querying X11 wakes the compositor; the log confirms its Windows peer.
        xrandr --current >/dev/null 2>&1 || true
        if grep -Eq 'rdp_rail_notify_app_list\(\): rdp_peer 0x[[:xdigit:]]+' /mnt/wslg/weston.log 2>/dev/null; then
            ready=1
            break
        fi
        sleep 1
    done
    if (( ! ready )); then
        echo 'WSLg display is not ready. Check WSLg or try restarting WSL.' >&2
        exit 1
    fi
    sleep 2
fi

exec ./build-debug/vulkan-starter-app
