#!/usr/bin/env bash
# Repeat the runtime adaptation scenarios (S1-S7) with seeds 1235-1238.
# The thresholds and delays are calibrated from the same seed-1234 baseline as the
# original runs (switch_baselines/*.csv), so every seed sees the same scenario.
#
# usage:  bash run_switch_seeds.sh resnet_cifar10     (or cnn_mnist)
#         SHUTDOWN=0 bash run_switch_seeds.sh cnn_mnist   # do not power off at the end
set -u
cd "$(dirname "$0")"
SETUP=$1
SEEDS=${SEEDS:-"1235 1236 1237 1238"}
source venv/bin/activate
for s in $SEEDS; do
  echo "$(date '+%F %T') === $SETUP seed $s"
  THESIS_SEED=$s python run_switch.py run --setup "$SETUP" \
      --baseline "switch_baselines/${SETUP}_rounds.csv" --out "switch_results_s$s" 2>&1 | tee -a "switch_${SETUP}_s$s.log"
  # checkpoints are only needed during a run
  find "switch_results_s$s" -name "*.pt" -delete
done
echo "$(date '+%F %T') all seeds finished"
tar czf ~/switch_${SETUP}_seeds.tar.gz switch_results_s*/"$SETUP" switch_${SETUP}_s*.log
if [ "${SHUTDOWN:-1}" = "1" ]; then sudo shutdown -h now; fi
