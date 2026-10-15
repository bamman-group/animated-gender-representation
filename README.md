# Gender Representation in Animated Film

Code and data to support:

David Bamman, Allison Cooper, Ruby Alvarez Rubio, Reina Kushihashi and Madison Mar (2026), "Measuring Gender Representation in Animated Films", [preprint](https://people.ischool.berkeley.edu/~dbamman/pubs/pdf/gender_animation.pdf).

This work includes training and evaluating models for animated character detection and recognition, and applying those models to a collection of 224 animated feature films spanning 1937-2025.

## Repository map

| Directory | |
|---|---|
| [`analysis/`](analysis/) | Code and data to generate the figures in the paper. |
| [`pipeline/`](pipeline/) | The end-to-end path from a film to per-character measurements, runnable on a public-domain sample clip. |
| [`benchmarks/`](benchmarks/) | Training and evaluation scripts for animated character detection and recognition. |

Each directory has its own README with the details.

## License

Code in this repository is MIT-licensed unless a subdirectory states otherwise;
vendored components keep the upstream licenses recorded in their own `NOTICE.md`. 

## AI Usage

We use Claude Code for coding assistance in this work. No AI was used in the writing of the paper, or
in any manual annotations created.