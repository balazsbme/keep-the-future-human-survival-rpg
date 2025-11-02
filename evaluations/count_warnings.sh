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
readarray -t WARNINGS <<'WARNINGS'
'factions_file' parameter is deprecated; please supply a scenario file instead.|'factions_file' parameter is deprecated; please supply a scenario file instead.
Skipping character with missing name or faction: %s|Skipping character with missing name or faction
No faction specification found for %s (character %s)|No faction specification found for
Falling back to temporary instance directory at %s due to permission error|Falling back to temporary instance directory at
Unknown player key '%s'; defaulting to random|Unknown player key
Invalid AUTOMATED_AGENT_MAX_EXCHANGES value %r; falling back to default limit|Invalid AUTOMATED_AGENT_MAX_EXCHANGES value
AUTOMATED_AGENT_MAX_EXCHANGES must be at least 1; using default limit|AUTOMATED_AGENT_MAX_EXCHANGES must be at least 1; using default
Selected option for %s not in available list; defaulting to first|not in available list; defaulting to first
Failed to prepare assessment cache for %s: %s|Failed to prepare assessment cache
Using default faction label '%s' for %s|Using default faction label
Failed to create cached content for %s: %s|Failed to create cached content
Failed to parse response JSON for %s: %s|Failed to parse response JSON
Overwriting related triplet %s for %s option '%s' due to out-of-range index (max=%d)|due to out-of-range index
Restricted prompt for %s still produced triplet-related actions|Restricted prompt for
Overwriting related triplet %s for %s action '%s' due to credibility restriction|due to credibility restriction
Model returned no usable responses for %s|Model returned no usable responses
Expected at least one chat option from %s; defaulting to action proposals|defaulting to action proposals
Using hardcoded fallback action for %s due to empty option list; text='%s', attribute='%s'|Using hardcoded fallback action
Scenario file %s not found; player triplets unavailable|player triplets unavailable
Faction context file %s not found; using empty context|using empty context
Player model suggested action-oriented responses; using scripted prompts instead|using scripted prompts instead
No chat options generated for player; using fallback prompts|using fallback prompts
Invalid integer value %r encountered in configuration; using %d|Invalid integer value
Invalid float value %r encountered in configuration; using %s|Invalid float value
Game configuration file %s not found; falling back to defaults|Game configuration file
Failed to parse game configuration %s: %s; using defaults|Failed to parse game configuration
Failed to parse credibility matrix JSON: %s|Failed to parse credibility matrix JSON
Credibility matrix JSON root must be an object; using fallback|JSON root must be an object; using fallback
Missing 'factions' list in credibility matrix; using fallback order|Missing 'factions' list in credibility matrix
Missing 'credibility' mapping in matrix; using fallback values|Missing 'credibility' mapping in matrix
Using default action label '%s' for %s option '%s'|Using default action label
Invalid GEMINI_CACHE_TTL_SECONDS value %r; defaulting to %d seconds|Invalid GEMINI_CACHE_TTL_SECONDS value
GEMINI_CACHE_TTL_SECONDS must be at least 1; using default %d seconds|GEMINI_CACHE_TTL_SECONDS must be at least 1; using default
Failed to list Gemini caches: %s|Failed to list Gemini caches
Failed to initialise Gemini cache manager: %s|Failed to initialise Gemini cache manager
Character loading interrupted by exhausted mock; reusing existing roster|Character loading interrupted by exhausted mock
Player character reset interrupted by exhausted mock; reusing existing persona|Player character reset interrupted by exhausted mock
WARNINGS
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