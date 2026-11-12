#!/usr/bin/env bash
set -euo pipefail
role="${1:-help}"
case "$role" in
  prefill|decode|router)
    shift
    exec bash "$(dirname "${BASH_SOURCE[0]}")/$role.sh" "$@"
    ;;
  help|-h|--help)
    echo 'Usage: image [prefill|decode|router]'
    ;;
  *)
    echo "Unknown role: $role" >&2
    exit 2
    ;;
esac
