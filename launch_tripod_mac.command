#!/bin/zsh
set -eu
exec "${0:A:h}/launch_mac.command" --interaction tripod "$@"
