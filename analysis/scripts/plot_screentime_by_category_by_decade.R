#!/usr/bin/env Rscript
# % female screentime, human vs. animal/other characters, animated movies
# only, as bars with bootstrapped 95% error bars per decade.
#
# Both confidence intervals are precomputed by bootstrap_decade.py (not
# computed here):
#   python3 bootstrap_decade.py data/animated_screentime_gender.tsv \
#       "% screentime, female (human)" --year-col imdb_date > figures/ci_decade_screentime_female_human.res.txt
#   python3 bootstrap_decade.py data/animated_screentime_gender.tsv \
#       "% screentime, female (animal/other)" --year-col imdb_date > figures/ci_decade_screentime_female_animal_other.res.txt
#
# Usage:
#   Rscript plot_screentime_by_category_by_decade.R [human_ci_file] [animal_other_ci_file] [outfile]

library(ggplot2)

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
figures_dir <- file.path(script_dir, "..", "figures")

args <- commandArgs(trailingOnly = TRUE)
human_ci_file <- ifelse(length(args) >= 1, args[1], file.path(figures_dir, "ci_decade_screentime_female_human.res.txt"))
animal_other_ci_file <- ifelse(length(args) >= 2, args[2], file.path(figures_dir, "ci_decade_screentime_female_animal_other.res.txt"))
outfile <- ifelse(length(args) >= 3, args[3], file.path(figures_dir, "screentime_by_category_by_decade.pdf"))

# Lampshade Amber (Human) and Doorway Brick (Animal/Other) from the
# govorit_moskva palette.
series_colors <- c("Human" = "#FFC155", "Animal/Other" = "#7B2E23")

MIN_DECADE <- 1980
MAX_DECADE <- 2020

read_ci <- function(path, label) {
  df <- read.table(path, sep = "\t", quote = "", head = FALSE, na.strings = "None", comment.char = "")
  names(df) <- c("decade", "p2.5", "mean", "p97.5", "n")
  df$series <- label
  df
}

ci_combined <- rbind(
  read_ci(human_ci_file, "Human"),
  read_ci(animal_other_ci_file, "Animal/Other")
)
ci_combined <- ci_combined[ci_combined$decade >= MIN_DECADE & ci_combined$decade <= MAX_DECADE, ]
ci_combined$series <- factor(ci_combined$series, levels = c("Human", "Animal/Other"))
ci_combined$decade_label <- paste0(ci_combined$decade, "s")
ci_combined$decade_label <- factor(
  ci_combined$decade_label,
  levels = paste0(seq(MIN_DECADE, MAX_DECADE, by = 10), "s")
)

ymin_val <- 0
ymax_val <- 100

g <- ggplot(ci_combined, aes(x = decade_label, y = mean, fill = series)) +
  theme_classic() +
  geom_col(position = position_dodge(width = 0.7), width = 0.65, color = "black") +
  geom_errorbar(
    aes(ymin = p2.5, ymax = p97.5),
    position = position_dodge(width = 0.7),
    width = 0.2
  ) +
  scale_fill_manual(name = NULL, values = series_colors) +
  coord_cartesian(ylim = c(ymin_val, ymax_val)) +
  ylab("% Screentime, Female") + xlab("") +
  geom_hline(yintercept = seq(ymin_val, ymax_val, by = 10), linetype = "dotdash", size = .2, color = "grey", alpha = 0.5) +
  theme(
    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 18),
    axis.text.x = element_text(size = 14),
    axis.text.y = element_text(size = 14),
    legend.position = "top",
    legend.text = element_text(size = 16)
  )

pdf(outfile, width = 7)
g
dev.off()
cat(sprintf("Wrote %s\n", outfile))
