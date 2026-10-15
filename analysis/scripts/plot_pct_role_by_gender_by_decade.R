#!/usr/bin/env Rscript
# What percentage of a gender's (known-role) screentime is antagonist,
# animated vs. live action, as bars with bootstrapped 95% error bars per
# decade. Only "antagonist" is plotted - protagonist is
# redundant, since protagonist + antagonist always sums to 100% (see
# scripts/pct_role_by_gender.py). Two facets - Animated / Live action -
# each with one bar for Female and one for Male.
#
# All four confidence intervals are precomputed by bootstrap_decade.py (not
# computed here), from data/pct_role_by_gender_animated.tsv and
# data/pct_role_by_gender_live_action.tsv (built by
# scripts/pct_role_by_gender.py from
# data/animated_screentime_by_role.tsv / data/live_action_screentime_by_role.tsv):
#   python3 bootstrap_decade.py data/pct_role_by_gender_animated.tsv \
#       pct_antagonist_female --year-col year > figures/ci_decade_pct_antagonist_female_animated.res.txt
#   python3 bootstrap_decade.py data/pct_role_by_gender_animated.tsv \
#       pct_antagonist_male --year-col year > figures/ci_decade_pct_antagonist_male_animated.res.txt
#   python3 bootstrap_decade.py data/pct_role_by_gender_live_action.tsv \
#       pct_antagonist_female --year-col year > figures/ci_decade_pct_antagonist_female_live_action.res.txt
#   python3 bootstrap_decade.py data/pct_role_by_gender_live_action.tsv \
#       pct_antagonist_male --year-col year > figures/ci_decade_pct_antagonist_male_live_action.res.txt
#
# Usage:
#   Rscript plot_pct_role_by_gender_by_decade.R [female_anim_ci] [male_anim_ci] \
#       [female_live_ci] [male_live_ci] [outfile]

library(ggplot2)

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
figures_dir <- file.path(script_dir, "..", "figures")

args <- commandArgs(trailingOnly = TRUE)
female_anim_ci_file <- ifelse(length(args) >= 1, args[1], file.path(figures_dir, "ci_decade_pct_antagonist_female_animated.res.txt"))
male_anim_ci_file <- ifelse(length(args) >= 2, args[2], file.path(figures_dir, "ci_decade_pct_antagonist_male_animated.res.txt"))
female_live_ci_file <- ifelse(length(args) >= 3, args[3], file.path(figures_dir, "ci_decade_pct_antagonist_female_live_action.res.txt"))
male_live_ci_file <- ifelse(length(args) >= 4, args[4], file.path(figures_dir, "ci_decade_pct_antagonist_male_live_action.res.txt"))
outfile <- ifelse(length(args) >= 5, args[5], file.path(figures_dir, "pct_role_by_gender_by_decade.pdf"))

gender_colors <- c("Female" = "#4A8890", "Male" = "#FFC155")

MIN_DECADE <- 1980
MAX_DECADE <- 2020

read_ci <- function(path, corpus, gender) {
  df <- read.table(path, sep = "\t", quote = "", head = FALSE, na.strings = "None", comment.char = "")
  names(df) <- c("decade", "p2.5", "mean", "p97.5", "n")
  df$corpus <- corpus
  df$gender <- gender
  df
}

ci_combined <- rbind(
  read_ci(female_anim_ci_file, "Animated", "Female"),
  read_ci(male_anim_ci_file, "Animated", "Male"),
  read_ci(female_live_ci_file, "Live action", "Female"),
  read_ci(male_live_ci_file, "Live action", "Male")
)
ci_combined$corpus <- factor(ci_combined$corpus, levels = c("Animated", "Live action"))
ci_combined$gender <- factor(ci_combined$gender, levels = c("Female", "Male"))

ci_combined <- ci_combined[ci_combined$decade >= MIN_DECADE & ci_combined$decade <= MAX_DECADE, ]
ci_combined$decade_label <- paste0(ci_combined$decade, "s")
ci_combined$decade_label <- factor(
  ci_combined$decade_label,
  levels = paste0(seq(MIN_DECADE, MAX_DECADE, by = 10), "s")
)

ymin_val <- 0
ymax_val <- 75

g <- ggplot(ci_combined, aes(x = decade_label, y = mean, fill = gender)) +
  theme_classic() +
  geom_col(position = position_dodge(width = 0.7), width = 0.65, color = "black") +
  geom_errorbar(
    aes(ymin = p2.5, ymax = p97.5),
    position = position_dodge(width = 0.7),
    width = 0.2
  ) +
  scale_fill_manual(name = NULL, values = gender_colors) +
  facet_wrap(~corpus) +
  coord_cartesian(ylim = c(ymin_val, ymax_val)) +
  ylab("% of Screentime that is Antagonist") + xlab("") +
  geom_hline(yintercept = seq(ymin_val, ymax_val, by = 25), linetype = "dotdash", size = .2, color = "grey", alpha = 0.5) +
  theme(
    axis.title.x = element_text(size = 18),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 14),
    strip.text = element_text(size = 14),
    legend.position = "top",
    legend.text = element_text(size = 14),
    panel.spacing = unit(1.4, "lines"),
    axis.line = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5)
  )

# Draw a vertical rule in the gap between the two facet panels (Animated |
# Live action). facet_wrap has no built-in "line between panels" option, so
# this reads the panel column widths/positions out of the rendered gtable to
# find exactly where that gap sits, then overlays a grid::grid.lines() at
# that x position, spanning the vertical extent of the panel row. Panel
# rows/columns are sized in "null" units (flexible, equal shares of
# whatever space is left over) which only resolve to absolute sizes once
# placed on a device of known size - so the leftover space is computed by
# hand from the known PDF width/height minus everything with a fixed size,
# split evenly across the null columns/rows (facet_wrap gives all panels
# equal 1null weight by default).
draw_panel_divider <- function(g, dev_width_in, dev_height_in) {
  gt <- ggplotGrob(g)
  panel_rows <- gt$layout[grepl("^panel", gt$layout$name), ]
  panel_rows <- panel_rows[order(panel_rows$l), ]
  stopifnot(nrow(panel_rows) == 2)

  resolve_nulls <- function(units, known_total_cm) {
    vals_cm <- grid::convertUnit(units, "cm", valueOnly = TRUE)
    is_null <- grid::unitType(units) == "null"
    leftover_cm <- known_total_cm - sum(vals_cm[!is_null])
    vals_cm[is_null] <- leftover_cm / sum(is_null)
    vals_cm
  }

  widths_cm <- resolve_nulls(gt$widths, dev_width_in * 2.54)
  heights_cm <- resolve_nulls(gt$heights, dev_height_in * 2.54)

  left_panel_col <- panel_rows$r[1]
  gap_col <- left_panel_col + 1
  x_npc <- (sum(widths_cm[seq_len(left_panel_col)]) + widths_cm[gap_col] / 2) / sum(widths_cm)

  top_row <- min(panel_rows$t)
  bottom_row <- max(panel_rows$b)
  y_top_npc <- 1 - sum(heights_cm[seq_len(top_row - 1)]) / sum(heights_cm)
  y_bottom_npc <- 1 - sum(heights_cm[seq_len(bottom_row)]) / sum(heights_cm)

  grid::grid.lines(
    x = unit(rep(x_npc, 2), "npc"),
    y = unit(c(y_bottom_npc, y_top_npc), "npc"),
    gp = grid::gpar(col = "black", lwd = 1)
  )
}

PDF_WIDTH_IN <- 10
PDF_HEIGHT_IN <- 5

pdf(outfile, width = PDF_WIDTH_IN, height = PDF_HEIGHT_IN)
print(g)
draw_panel_divider(g, PDF_WIDTH_IN, PDF_HEIGHT_IN)
dev.off()
cat(sprintf("Wrote %s\n", outfile))
