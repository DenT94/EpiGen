#!/bin/zsh
# Auto-commit safety net for the EpiGen hackathon build (Aug 15-16 2026).
# Runs periodically via a launchd agent (see scripts/com.epigen.autocommit.plist).
# Commits only if there are actual changes; never pushes.

set -euo pipefail

REPO_DIR="/Users/dtrimcev/Dropbox/Work_Main/EpiGen"
LOG_FILE="$REPO_DIR/scripts/auto_commit.log"

cd "$REPO_DIR"

# Nothing to do if there's no repo or nothing changed.
if [[ -z "$(git status --porcelain)" ]]; then
  exit 0
fi

TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
git add -A
git commit -m "Auto-commit: $TIMESTAMP" >> "$LOG_FILE" 2>&1
echo "[$TIMESTAMP] committed" >> "$LOG_FILE"
