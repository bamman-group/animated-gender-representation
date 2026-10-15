#!/usr/bin/env Rscript
# % female screentime, animated vs. live-action, 1980-2022, as bars with
# bootstrapped 95% error bars per decade.
#
# Both confidence intervals are precomputed by bootstrap_decade.py (not
# computed here):
#   python3 bootstrap_decade.py data/animated_screentime_gender.tsv \
#       "% screentime, female (all)" --year-col imdb_date > figures/ci_decade_screentime_female_all.res.txt
#   python3 bootstrap_decade.py data/liveaction_screentime_gender.tsv \
#       "% screentime, female" --year-col year > figures/ci_decade_liveaction_screentime_female.res.txt
# liveaction_screentime_gender.tsv's "% screentime, female" is already on a
# 0-100 scale, so no rescaling is needed below.
#
# Usage:
#   Rscript plot_screentime_vs_live_action_by_decade.R [animated_ci_file] [live_action_ci_file] [outfile]

library(ggplot2)

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
figures_dir <- file.path(script_dir, "..", "figures")

args <- commandArgs(trailingOnly = TRUE)
animated_ci_file <- ifelse(length(args) >= 1, args[1], file.path(figures_dir, "ci_decade_screentime_female_all.res.txt"))
live_action_ci_file <- ifelse(length(args) >= 2, args[2], file.path(figures_dir, "ci_decade_liveaction_screentime_female.res.txt"))
outfile <- ifelse(length(args) >= 3, args[3], file.path(figures_dir, "screentime_vs_live_action_by_decade.pdf"))

# Aerial Teal and Lampshade Amber from the govorit_moskva palette.
series_colors <- c("Animated" = "#4A8890", "Live action" = "#EA8A2A")

MIN_DECADE <- 1980
MAX_DECADE <- 2020

read_ci <- function(path, label, rescale = FALSE) {
  df <- read.table(path, sep = "\t", quote = "", head = FALSE, na.strings = "None", comment.char = "")
  names(df) <- c("decade", "p2.5", "mean", "p97.5", "n")
  if (rescale) {
    df$p2.5 <- df$p2.5 * 100
    df$mean <- df$mean * 100
    df$p97.5 <- df$p97.5 * 100
  }
  df$series <- label
  df
}

ci_combined <- rbind(
  read_ci(animated_ci_file, "Animated"),
  read_ci(live_action_ci_file, "Live action")
)
ci_combined <- ci_combined[ci_combined$decade >= MIN_DECADE & ci_combined$decade <= MAX_DECADE, ]
ci_combined$series <- factor(ci_combined$series, levels = c("Animated", "Live action"))
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
