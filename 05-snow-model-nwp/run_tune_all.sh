#!/bin/sh
# 対象地点（引数1: hirosaki または aomori、既定は hirosaki）について、
# 3つの特徴量セット × 手法 × 3つのseed を、9件ずつ並列でチューニングする（各件は1スレッド）。
#   回帰3手法 + 大雪の分類2手法（20cm）。弘前は参考の分類（15cm、clf_ref）2手法も加える。
#   → 青森 45件、弘前 63件。
# ログ: logs/<地点>/tune/<task>_<set>_<method>_seed<seed>.log、まとめ logs/<地点>/tune_all.log
cd "$(dirname "$0")"
TARGET=${1:-hirosaki}
mkdir -p logs/$TARGET/tune
JOBS=logs/$TARGET/jobs.txt
: > $JOBS
TASKS="reg:rf reg:hgb clf:hgbc reg:ridge clf:logreg"
[ "$TARGET" = hirosaki ] && TASKS="$TASKS clf_ref:hgbc clf_ref:logreg"
for m in $TASKS; do for s in 42 7 2024; do for f in SB SA S0; do
  echo "${m%%:*} $f ${m##*:} $s" >> $JOBS
done; done; done
export SNOW_TARGET=$TARGET
xargs -P 9 -L 1 sh -c 'OMP_NUM_THREADS=1 ../.venv/bin/python -W ignore tune.py --task $0 --set $1 --method $2 --seed $3 --trials 100 > logs/$SNOW_TARGET/tune/$0_$1_$2_seed$3.log 2>&1' < $JOBS
cat logs/$TARGET/tune/*.log > logs/$TARGET/tune_all.log
