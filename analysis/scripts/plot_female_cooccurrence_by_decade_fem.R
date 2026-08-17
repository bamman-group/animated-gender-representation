#!/usr/bin/env Rscript
# "Frames with 1+ female" as a single-panel figure. Two series - Animated,
# Live action - human characters only (animated uses its _human column;
# live action has no human/animal distinction), as bars with bootstrapped
# 95% error bars per decade. Fraction in [0, 1], rescaled to 0-100 here.
#
# Both confidence intervals are precomputed by bootstrap_decade.py (not
# computed here):
#   python3 bootstrap_decade.py data/female_cooccurrence.tsv ratio_fem_human \
#       --year-col year > figures/ci_decade_female_cooccurrence_fem_human.res.txt
#   python3 bootstrap_decade.py data/live_action_cooccurrence.tsv ratio_fem \
#       --year-col year > figures/ci_decade_live_action_cooccurrence_fem.res.txt
#
# Usage:
#   Rscript plot_female_cooccurrence_by_decade_fem.R [fem_human_ci] [live_fem_ci] [outfile]

library(ggplot2)

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
figures_dir <- file.path(script_dir, "..", "figures")

args <- commandArgs(trailingOnly = TRUE)
fem_human_ci_file <- ifelse(length(args) >= 1, args[1], file.path(figures_dir, "ci_decade_female_cooccurrence_fem_human.res.txt"))
live_fem_ci_file <- ifelse(length(args) >= 2, args[2], file.path(figures_dir, "ci_decade_live_action_cooccurrence_fem.res.txt"))
outfile <- ifelse(length(args) >= 3, args[3], file.path(figures_dir, "female_cooccurrence_by_decade_fem.pdf"))

scope_colors <- c(
  "Animated" = "#988391",
  "Live action" = "#FFC4C1"
)

MIN_DECADE <- 1980
MAX_DECADE <- 2020

read_ci <- function(path, scope) {
  df <- read.table(path, sep = "\t", quote = "", head = FALSE, na.strings = "None", comment.char = "")
  names(df) <- c("decade", "p2.5", "mean", "p97.5", "n")
  df$p2.5 <- df$p2.5 * 100
  df$mean <- df$mean * 100
  df$p97.5 <- df$p97.5 * 100
  df$scope <- scope
  df
}

ci_combined <- rbind(
  read_ci(fem_human_ci_file, "Animated"),
  read_ci(live_fem_ci_file, "Live action")
)
ci_combined$scope <- factor(ci_combined$scope, levels = names(scope_colors))

ci_combined <- ci_combined[ci_combined$decade >= MIN_DECADE & ci_combined$decade <= MAX_DECADE, ]
ci_combined$decade_label <- paste0(ci_combined$decade, "s")
ci_combined$decade_label <- factor(
  ci_combined$decade_label,
  levels = paste0(seq(MIN_DECADE, MAX_DECADE, by = 10), "s")
)

ymin_val <- 0
ymax_val <- 100

g <- ggplot(ci_combined, aes(x = decade_label, y = mean, fill = scope)) +
  theme_classic() +
  geom_col(position = position_dodge(width = 0.7), width = 0.65, color = "black") +
  geom_errorbar(
    aes(ymin = p2.5, ymax = p97.5),
    position = position_dodge(width = 0.7),
    width = 0.2
  ) +
  scale_fill_manual(name = NULL, values = scope_colors) +
  coord_cartesian(ylim = c(ymin_val, ymax_val)) +
  ylab("% of Shots with a Second Female Character") + xlab("") +
  geom_hline(yintercept = seq(ymin_val, ymax_val, by = 20), linetype = "dotdash", size = .2, color = "grey", alpha = 0.5) +
  theme(
    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 14),
    axis.text.y = element_text(size = 14),
    legend.position = "top",
    legend.text = element_text(size = 14)
  )

pdf(outfile, width = 7)
g
dev.off()
cat(sprintf("Wrote %s\n", outfile))
