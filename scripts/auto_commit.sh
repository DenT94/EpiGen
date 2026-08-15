#!/bin/zsh
# Auto-commit + auto-push safety net for the EpiGen hackathon build
# (Aug 15-16 2026). Runs periodically via a launchd agent (see
# scripts/com.epigen.autocommit.plist). Commits only if there are actual
# changes, then pushes to origin/main. Push failures (e.g. no network) are
# logged but don't fail the run -- the commit itself always succeeds
# locally and will push on a later tick.

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

if git push origin main >> "$LOG_FILE" 2>&1; then
  echo "[$TIMESTAMP] pushed" >> "$LOG_FILE"
else
  echo "[$TIMESTAMP] push failed (will retry next tick)" >> "$LOG_FILE"
fi
