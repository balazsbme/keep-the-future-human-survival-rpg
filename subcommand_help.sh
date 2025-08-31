#!/usr/bin/env bash
# Enumerate subcommands of a command using bash completion
# and dump each subcommand's --help text to files.
# Usage: subcommand_help.sh <command> [output_dir] [max_depth]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command> [output_dir] [max_depth]" >&2
  exit 1
fi

cmd=$1
outdir=${2:-"${cmd}_help"}
max_depth=${3:-2}
mkdir -p "$outdir"

# Capture root command help once for comparison and save it.
root_help=$("$cmd" --help 2>&1 || true)
printf '%s\n' "$root_help" >"$outdir/${cmd}_help.txt"

# Load bash completion for the command if available.
if ! complete -p "$cmd" &>/dev/null; then
  [[ -f /usr/share/bash-completion/bash_completion ]] && \
    source /usr/share/bash-completion/bash_completion
  if ! complete -p "$cmd" &>/dev/null; then
    compfile="/usr/share/bash-completion/completions/$cmd"
    [[ -f "$compfile" ]] && source "$compfile"
  fi
fi

if ! complete -p "$cmd" &>/dev/null; then
  echo "No bash completion available for $cmd" >&2
  exit 1
fi

# Extract completion function name
comp_func=$(complete -p "$cmd" | awk '{for(i=1;i<=NF;i++) if ($i=="-F") {print $(i+1); exit}}')

# Track visited command paths to avoid repeats
declare -A visited

recurse() {
  local depth=$1; shift
  local words=("$@")
  local key="${words[*]}"
  [[ -n "${visited[$key]:-}" ]] && return
  visited[$key]=1

  (( depth >= max_depth )) && return

  # Prepare completion environment for next token
  COMP_WORDS=("${words[@]}" "")
  COMP_CWORD=${#words[@]}
  COMP_LINE="${key} "
  COMP_POINT=${#COMP_LINE}
  COMPREPLY=()
  "$comp_func" "$cmd" >/dev/null 2>&1 || true

  for sub in "${COMPREPLY[@]}"; do
    sub=${sub%/}
    [[ $sub == -* ]] && continue
    [[ $sub == help ]] && continue
    [[ $sub == */* ]] && continue
    [[ -e $sub ]] && continue
    local new_words=("${words[@]}" "$sub")
    local outfile="$outdir/$(printf '%s_' "${new_words[@]}")help.txt"
    local out
    out=$("${new_words[@]}" --help 2>&1 || true)
    [[ "$out" == "$root_help" ]] && continue
    printf '%s\n' "$out" >"$outfile"
    if grep -q '^\s*Commands:' <<<"$out"; then
      recurse $((depth+1)) "${new_words[@]}"
    fi
  done
}

recurse 0 "$cmd"
