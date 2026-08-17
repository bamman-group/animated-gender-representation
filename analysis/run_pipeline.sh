#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

ANIMATED_DIR="data/animated"
LIVE_ACTION_DIR="data/live_action"
ANIMATED_URL="http://yosemite.ischool.berkeley.edu/filmanalytics/animation/data"
LIVE_ACTION_URL="http://yosemite.ischool.berkeley.edu/filmanalytics/live-action/data"

download_corpus() {
  local dir=$1
  local url=$2

  if [ -d "$dir/recog" ]; then
    echo "== $dir already has recog/ - skipping download =="
    return
  fi

  echo "== Downloading $url into $dir =="
  mkdir -p "$dir"
  (
    cd "$dir"
    for f in recog tracks fps shots; do
      wget "$url/$f.tar.gz"
    done
    tar xzf recog.tar.gz && tar xzf tracks.tar.gz && tar xzf fps.tar.gz && tar xzf shots.tar.gz
  )
}

download_corpus "$ANIMATED_DIR" "$ANIMATED_URL"
download_corpus "$LIVE_ACTION_DIR" "$LIVE_ACTION_URL"

echo "== screentime_vs_live_action_by_decade.pdf / screentime_by_category_by_decade.pdf =="

python scripts/measure_screentime.py --data-dir "$ANIMATED_DIR" --output data/animated_screentime.tsv
python scripts/measure_screentime.py --data-dir "$LIVE_ACTION_DIR" --corpus live_action --output data/liveaction_screentime.tsv
python scripts/percent_female_screentime.py
python scripts/live_action_percent_female_screentime.py

python3 scripts/bootstrap_decade.py data/animated_screentime_gender.tsv "% screentime, female (all)" --year-col imdb_date > figures/ci_decade_screentime_female_all.res.txt
python3 scripts/bootstrap_decade.py data/liveaction_screentime_gender.tsv "% screentime, female" --year-col year > figures/ci_decade_liveaction_screentime_female.res.txt
Rscript scripts/plot_screentime_vs_live_action_by_decade.R

python3 scripts/bootstrap_decade.py data/animated_screentime_gender.tsv "% screentime, female (human)" --year-col imdb_date > figures/ci_decade_screentime_female_human.res.txt
python3 scripts/bootstrap_decade.py data/animated_screentime_gender.tsv "% screentime, female (animal/other)" --year-col imdb_date > figures/ci_decade_screentime_female_animal_other.res.txt
Rscript scripts/plot_screentime_by_category_by_decade.R

echo "== pct_role_by_gender_by_decade.pdf =="

python scripts/live_action_screentime_by_role.py --data-dir "$ANIMATED_DIR"
python scripts/live_action_screentime_by_role.py --data-dir "$LIVE_ACTION_DIR" --corpus live_action
python scripts/pct_role_by_gender.py --input data/animated_screentime_by_role.tsv --output data/pct_role_by_gender_animated.tsv
python scripts/pct_role_by_gender.py --input data/live_action_screentime_by_role.tsv --output data/pct_role_by_gender_live_action.tsv

python3 scripts/bootstrap_decade.py data/pct_role_by_gender_animated.tsv pct_antagonist_female --year-col year > figures/ci_decade_pct_antagonist_female_animated.res.txt
python3 scripts/bootstrap_decade.py data/pct_role_by_gender_animated.tsv pct_antagonist_male --year-col year > figures/ci_decade_pct_antagonist_male_animated.res.txt
python3 scripts/bootstrap_decade.py data/pct_role_by_gender_live_action.tsv pct_antagonist_female --year-col year > figures/ci_decade_pct_antagonist_female_live_action.res.txt
python3 scripts/bootstrap_decade.py data/pct_role_by_gender_live_action.tsv pct_antagonist_male --year-col year > figures/ci_decade_pct_antagonist_male_live_action.res.txt
Rscript scripts/plot_pct_role_by_gender_by_decade.R

echo "== female_cooccurrence_by_decade_fem.pdf =="

python scripts/female_cooccurrence.py --data-dir "$ANIMATED_DIR"
python scripts/female_cooccurrence.py --data-dir "$LIVE_ACTION_DIR" --corpus live_action

python3 scripts/bootstrap_decade.py data/female_cooccurrence.tsv ratio_fem_human --year-col year > figures/ci_decade_female_cooccurrence_fem_human.res.txt
python3 scripts/bootstrap_decade.py data/live_action_cooccurrence.tsv ratio_fem --year-col year > figures/ci_decade_live_action_cooccurrence_fem.res.txt
Rscript scripts/plot_female_cooccurrence_by_decade_fem.R

Rscript scripts/plot_screentime_by_franchise.R

echo "== Done. Figures written to figures/*.pdf =="
