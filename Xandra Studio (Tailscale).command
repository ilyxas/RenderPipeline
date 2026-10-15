#!/bin/sh
# Same launcher, also reachable from your phone over Tailscale (URL printed below).
XMS_GUI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$XMS_GUI_ROOT" || exit 1
exec ./xms gui --tailscale
