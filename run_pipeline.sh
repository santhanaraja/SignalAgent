#!/bin/bash
# ============================================================
# Signal Pipeline Runner
# Run manually or schedule via cron
# ============================================================
#
# USAGE:
#   ./run_pipeline.sh              # Single run
#   ./run_pipeline.sh --loop       # Run every 15 min (market hours)
#   ./run_pipeline.sh --serve      # Run + start local web server
#
# CRON SETUP (every 15 min during market hours, Mon-Fri):
#   */15 9-16 * * 1-5 cd /path/to/live-pipeline && ./run_pipeline.sh >> pipeline.log 2>&1
#
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
DIM='\033[0;90m'
NC='\033[0m'

run_once() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Signal Pipeline — $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

    echo -e "${GREEN}[1/2] Running Signal Engine...${NC}"
    python3 signal_engine.py

    echo -e "\n${GREEN}[2/2] Running History Manager...${NC}"
    python3 history_manager.py

    echo -e "\n${GREEN}✓ Pipeline complete.${NC}"
    echo -e "${DIM}Dashboard: public/index.html${NC}"
    echo -e "${DIM}History:   public/history.html${NC}\n"
}

serve() {
    echo -e "${GREEN}Starting local server on http://localhost:8080${NC}"
    cd public
    python3 -m http.server 8080
}

case "${1:-}" in
    --loop)
        echo -e "${BLUE}Running in loop mode (every 15 minutes)${NC}"
        echo -e "${DIM}Press Ctrl+C to stop${NC}\n"
        while true; do
            run_once
            echo -e "${DIM}Sleeping 15 minutes...${NC}"
            sleep 900
        done
        ;;
    --serve)
        run_once
        serve
        ;;
    *)
        run_once
        ;;
esac
