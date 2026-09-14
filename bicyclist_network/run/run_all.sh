#!/usr/bin/env bash
# run_all.sh — run the whole analysis end to end for the area in ../bikelane.yaml.
# No manual checks in between; gap candidates are joined as found.
# Safe to re-run: every stage skips work that is already done.
#
# Before this: the tiles for the same area must exist under imagery_root
# (download_MassGIS2025AerialImagery: fetch.py + tile.py), and the weights in
# run/src/ (python download.py — done here if missing).
#
#   tmux new -s bikelane
#   bash run/run_all.sh                   # everything, from predict
#   bash run/run_all.sh signs             # start from this stage
#   WORKERS=16 bash run/run_all.sh        # centerline extract parallelism
#
# Log: <data_root>/logs/run_all_<REGION>.log  (also printed to the terminal)

set -u -o pipefail
cd "$(dirname "$0")/.."            # the repository root, where bikelane.yaml lives

# the venv must be active (or on PATH) — otherwise every stage silently fails
if ! command -v bikelane >/dev/null; then
  for v in "$HOME/.venvs/bikelane/bin/activate" ".venv/bin/activate"; do
    [[ -f "$v" ]] && { source "$v"; unset PYTHONPATH LD_LIBRARY_PATH; break; }
  done
fi
command -v bikelane >/dev/null || { echo "bikelane not found — activate the venv first (pip install -e .)"; exit 127; }

WORKERS="${WORKERS:-}"
START="${1:-predict}"

STAGES=(
  "predict"
  "centerlines extract${WORKERS:+ --workers $WORKERS}"
  "centerlines graph"
  "centerlines clean"
  "signs detect"
  "signs filter"
  "join match"
  "join clean"
  "osm"
  "gaps scan"
  "gaps join"
  "gaps intersections"
  "gaps network"
)

# resolve data_root / region / inputs for the log file and the pre-flight check
eval "$(python - <<'EOF'
from bikelane_extract.config import Config
c = Config.from_args(None, None, None)
print(f'DATA_ROOT="{c.data_root}"; REGION="{c.region}"; TILES="{c.path("tiles_dir")}"; '
      f'MAPPINGS="{c.path("tile_mappings_csv")}"; SEG="{c.weights("seg")}"; YOLO="{c.weights("yolo")}"')
EOF
)"
mkdir -p "$DATA_ROOT/logs"
LOG="$DATA_ROOT/logs/run_all_${REGION}.log"
echo "=== run_all $REGION  $(date)  start=$START ===" | tee -a "$LOG"

# pre-flight: tiles (from the download tool) and weights (from download.py)
if [[ ! -f "$MAPPINGS" || ! -d "$TILES" ]]; then
  echo "no tiles for $REGION: expected $TILES and $MAPPINGS" | tee -a "$LOG"
  echo "run download_MassGIS2025AerialImagery for the same area first, or set imagery_root in bikelane.yaml" | tee -a "$LOG"
  exit 2
fi
if [[ ! -f "$SEG" || ! -f "$YOLO" ]]; then
  echo "----- python download.py  (weights → run/src/)" | tee -a "$LOG"
  python download.py 2>&1 | tee -a "$LOG" || exit 1
fi

started=0
t0=$(date +%s)
for stage in "${STAGES[@]}"; do
  name="${stage%% *}"
  [[ "$started" == 0 && "$stage" != "$START"* ]] && { echo "skip  $stage" | tee -a "$LOG"; continue; }
  started=1
  echo "----- bikelane $stage  ($(date +%H:%M))" | tee -a "$LOG"
  ts=$(date +%s)
  bikelane $stage 2>&1 | tee -a "$LOG"
  rc=$?
  echo "----- done $stage  rc=$rc  $(( ($(date +%s) - ts) / 60 )) min" | tee -a "$LOG"
  if [[ "$rc" != 0 ]]; then
    echo "FAILED at: bikelane $stage  — fix and re-run:  bash run/run_all.sh $name" | tee -a "$LOG"
    exit "$rc"
  fi
done
echo "=== all stages done  $(( ($(date +%s) - t0) / 3600 )) h  ===" | tee -a "$LOG"
echo "main products: $DATA_ROOT/bikelanes/$REGION/main/  (bike_facilities, network_edges, network_nodes)"
