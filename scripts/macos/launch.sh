#!/bin/zsh
set -u
cd -- "${0:A:h}"
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV

if [[ "$(uname -m)" != arm64 ]]; then
  print 'This archive requires an Apple Silicon Mac (M series).'
  exit 1
fi

"./runtime/bin/python3.11" -I -B -X utf8 "./portable.py" "$@"
pinchpilot_exit=$?
if (( pinchpilot_exit != 0 )); then
  print 'See README.txt and reports/ for troubleshooting.'
  if [[ -t 0 ]]; then
    read -r '?Press Enter to close...'
  fi
fi
exit "$pinchpilot_exit"
