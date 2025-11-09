#!/usr/bin/env -S bash -euo pipefail
set -euo pipefail
cd /app

if [ ! -f package.json ]; then
  echo "package.json not found in /app. Opening a shell."
  exec bash
fi

if [ ! -d node_modules ] || [ -z "$(ls -A node_modules 2>/dev/null || true)" ]; then
  if [ -f package-lock.json ]; then
    echo "Installing deps with npm ci..."
    npm ci
  else
    echo "Installing deps with npm install..."
    npm install
  fi
fi

SCRIPT="${NPM_SCRIPT:-dev}"
if node -e "process.exit(require('./package.json').scripts && require('./package.json').scripts['$SCRIPT'] ? 0 : 1)"; then
  echo "Starting: npm run $SCRIPT"
  exec npm run "$SCRIPT"
else
  echo "Script \"$SCRIPT\" not found in package.json. Opening a shell."
  echo "Available scripts:"
  node -e "const s=require('./package.json').scripts||{}; console.log(Object.keys(s).join('\n'))"
  exec bash
fi

echo "Sleeping for 1 minute..."
sleep 60
