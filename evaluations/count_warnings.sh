#!/usr/bin/env bash
# Count occurrences of known warning messages across a directory of log files.
#
# Usage:
#   count_warning_occurrences.sh <directory> [--hide-zero]
#
# By default all warnings are listed, including those with zero matches. Use
# --hide-zero to suppress warnings that do not appear in the target directory.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <directory> [--hide-zero]" >&2
    exit 1
fi
target_dir=$1
shift || true
if [[ ! -d "$target_dir" ]]; then
    echo "Error: $target_dir is not a directory" >&2
    exit 1
fi
include_zero=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hide-zero)
            include_zero=0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
warnings_file="$script_dir/warning_messages.txt"
if [[ ! -f "$warnings_file" ]]; then
    echo "Error: warnings file $warnings_file not found" >&2
    exit 1
fi
WARNINGS=()
while IFS= read -r line; do
    [[ -z "${line// }" ]] && continue
    if [[ $line == '#'* ]]; then
        continue
    fi
    WARNINGS+=("$line")
done < "$warnings_file"
printf "%-6s | %s\n" "Count" "Warning text"
printf "%-6s-+-%s\n" "------" "--------------------------------------------------------------"
for entry in "${WARNINGS[@]}"; do
    [[ -z "$entry" ]] && continue
    if [[ "$entry" == *"|"* ]]; then
        display=${entry%%|*}
        snippet=${entry#*|}
    else
        display=$entry
        snippet=$entry
    fi
    if matches=$(grep -R -F -h -s "$snippet" "$target_dir"); then
        count=$(printf '%s\n' "$matches" | awk 'END {print NR}')
    else
        count=0
    fi
    if [[ $include_zero -eq 1 || $count -gt 0 ]]; then
        printf "%-6s | %s\n" "$count" "$display"
    fi
done