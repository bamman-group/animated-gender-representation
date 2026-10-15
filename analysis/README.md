# Analysis

The analysis in the paper measures gender representation in animated film (1937-2025), and how it
compares to live action. This directory creates four figures:

- `screentime_vs_live_action_by_decade.pdf` - % of screentime (female), animated vs. live action
- `screentime_by_category_by_decade.pdf` - % of screentime (female), human vs. animal/other characters (animated only)
- `pct_role_by_gender_by_decade.pdf` - % of screentime (antagonist vs. protagonist), female + male, animated + live action
- `female_cooccurrence_by_decade_fem.pdf` - % of shots with 2+ female characters (in shots with at least one female character), animated vs. live action
- `screentime_by_franchise.pdf` - % of screentime (female) within franchises containing 3+ movies

These scripts reproduce all four figures, starting from raw pipeline data plus the metadata files already
in `data/`:

- `animation_metadata.tsv`, `character_gender.tsv` (animated)
- `liveaction_metadata.tsv`, `wikidata.actor.historical.gender.080526.tsv`,
  `live_action_protagonist_antagonist.tsv` (live action)

`run_pipeline.sh` (in the current directory) executes all of the following commands, placing raw data in the `data/` directory.

## 1. Download raw data

`recog`/`tracks`/`fps`/`shots`, one tarball per file type per corpus:

```bash
mkdir -p /path/to/animated && cd /path/to/animated
wget http://yosemite.ischool.berkeley.edu/filmanalytics/animation/data/recog.tar.gz
wget http://yosemite.ischool.berkeley.edu/filmanalytics/animation/data/tracks.tar.gz
wget http://yosemite.ischool.berkeley.edu/filmanalytics/animation/data/fps.tar.gz
wget http://yosemite.ischool.berkeley.edu/filmanalytics/animation/data/shots.tar.gz
tar xzf recog.tar.gz && tar xzf tracks.tar.gz && tar xzf fps.tar.gz && tar xzf shots.tar.gz

mkdir -p /path/to/live_action && cd /path/to/live_action
wget http://yosemite.ischool.berkeley.edu/filmanalytics/live-action/data/recog.tar.gz
wget http://yosemite.ischool.berkeley.edu/filmanalytics/live-action/data/tracks.tar.gz
wget http://yosemite.ischool.berkeley.edu/filmanalytics/live-action/data/fps.tar.gz
wget http://yosemite.ischool.berkeley.edu/filmanalytics/live-action/data/shots.tar.gz
tar xzf recog.tar.gz && tar xzf tracks.tar.gz && tar xzf fps.tar.gz && tar xzf shots.tar.gz
```

## 2. Build the figures

Run from `analysis/`, with `/path/to/animated` and `/path/to/live_action`
from step 1.

**`screentime_vs_live_action_by_decade.pdf` / `screentime_by_category_by_decade.pdf`:**

```bash
python scripts/measure_screentime.py --data-dir /path/to/animated --output data/animated_screentime.tsv
python scripts/measure_screentime.py --data-dir /path/to/live_action --corpus live_action --output data/liveaction_screentime.tsv
python scripts/percent_female_screentime.py
python scripts/live_action_percent_female_screentime.py

python3 scripts/bootstrap_decade.py data/animated_screentime_gender.tsv "% screentime, female (all)" --year-col imdb_date > figures/ci_decade_screentime_female_all.res.txt
python3 scripts/bootstrap_decade.py data/liveaction_screentime_gender.tsv "% screentime, female" --year-col year > figures/ci_decade_liveaction_screentime_female.res.txt
Rscript scripts/plot_screentime_vs_live_action_by_decade.R

python3 scripts/bootstrap_decade.py data/animated_screentime_gender.tsv "% screentime, female (human)" --year-col imdb_date > figures/ci_decade_screentime_female_human.res.txt
python3 scripts/bootstrap_decade.py data/animated_screentime_gender.tsv "% screentime, female (animal/other)" --year-col imdb_date > figures/ci_decade_screentime_female_animal_other.res.txt
Rscript scripts/plot_screentime_by_category_by_decade.R
```

**`pct_role_by_gender_by_decade.pdf`:**

```bash
python scripts/live_action_screentime_by_role.py --data-dir /path/to/animated
python scripts/live_action_screentime_by_role.py --data-dir /path/to/live_action --corpus live_action
python scripts/pct_role_by_gender.py --input data/animated_screentime_by_role.tsv --output data/pct_role_by_gender_animated.tsv
python scripts/pct_role_by_gender.py --input data/live_action_screentime_by_role.tsv --output data/pct_role_by_gender_live_action.tsv

python3 scripts/bootstrap_decade.py data/pct_role_by_gender_animated.tsv pct_antagonist_female --year-col year > figures/ci_decade_pct_antagonist_female_animated.res.txt
python3 scripts/bootstrap_decade.py data/pct_role_by_gender_animated.tsv pct_antagonist_male --year-col year > figures/ci_decade_pct_antagonist_male_animated.res.txt
python3 scripts/bootstrap_decade.py data/pct_role_by_gender_live_action.tsv pct_antagonist_female --year-col year > figures/ci_decade_pct_antagonist_female_live_action.res.txt
python3 scripts/bootstrap_decade.py data/pct_role_by_gender_live_action.tsv pct_antagonist_male --year-col year > figures/ci_decade_pct_antagonist_male_live_action.res.txt
Rscript scripts/plot_pct_role_by_gender_by_decade.R
```

**`female_cooccurrence_by_decade_fem.pdf`:**

```bash
python scripts/female_cooccurrence.py --data-dir /path/to/animated
python scripts/female_cooccurrence.py --data-dir /path/to/live_action --corpus live_action

python3 scripts/bootstrap_decade.py data/female_cooccurrence.tsv ratio_fem_human --year-col year > figures/ci_decade_female_cooccurrence_fem_human.res.txt
python3 scripts/bootstrap_decade.py data/live_action_cooccurrence.tsv ratio_fem --year-col year > figures/ci_decade_live_action_cooccurrence_fem.res.txt
Rscript scripts/plot_female_cooccurrence_by_decade_fem.R
```

**`screentime_by_franchise.pdf`:**

```
Rscript scripts/plot_screentime_by_franchise.R
```

