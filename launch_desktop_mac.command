#!/bin/zsh
set -eu
cd -- "${0:A:h}"

if command -v uv >/dev/null 2>&1; then
  pinchpilot_uv="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  pinchpilot_uv="$HOME/.local/bin/uv"
else
  print 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'
  print 'If Homebrew is installed: brew install uv'
  read -r '?Press Enter to close...'
  exit 1
fi

"$pinchpilot_uv" sync --locked --no-dev
exec "$pinchpilot_uv" run --no-sync pinchpilot desktop "$@"
