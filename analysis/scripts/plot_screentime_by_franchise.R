#!/usr/bin/env Rscript
# Plot "% screentime, female (all)" over time for each multi-movie
# franchise, one facet per franchise (movies within a franchise connected by
# a line, in release order), labeling each point with the movie's title.
#
# Usage:
#   Rscript plot_screentime_by_franchise.R [infile] [outfile]

library(ggplot2)
library(ggrepel)

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))

args <- commandArgs(trailingOnly = TRUE)
infile <- ifelse(length(args) >= 1, args[1], file.path(script_dir, "..", "data", "animated_screentime_gender.tsv"))
outfile <- ifelse(length(args) >= 2, args[2], file.path(script_dir, "..", "figures", "screentime_by_franchise.pdf"))

# Lampshade Amber from the govorit_moskva palette.
line_color <- "#EA8A2A"

MIN_FRANCHISE_SIZE <- 3

##########################################################################
# Movie label abbreviations (EDIT THIS DICTIONARY)
##########################################################################

movie_abbreviations <- c(
    "Madagascar 3: Europe's Most Wanted" = "Madagascar 3",
    "Madagascar: Escape 2 Africa" = "Madagascar 2",
    "Pokémon: The First Movie - Mewtwo Strikes Back" = "Pokémon: Mewtwo",
    "The Lego Batman Movie" = "Lego Batman",
    "Captain Underpants: The First Epic Movie" = "Captain Underpants",
    "The Land Before Time" = "Land Before Time",
    "A Boy Named Charlie Brown" = "Charlie Brown",
    "PAW Patrol: The Movie" = "PAW Patrol",
    "Spirit: Stallion of the Cimarron" = "Spirit",
    "Snow White and the Seven Dwarfs" = "Snow White",
    "The Three Caballeros" = "Three Caballeros",
    "Hotel Transylvania 3: Summer Vacation" = "Hotel Transylvania 3",
    "The Lego Movie 2: The Second Part" = "Lego Movie 2",
    "The Lego Movie" = "Lego Movie",
    "WALL·E" = "WALL-E",
    "Cloudy with a Chance of Meatballs" = "Cloudy",
    "Cloudy with a Chance of Meatballs 2" = "Cloudy 2",
    "Monsters vs. Aliens" = "Monsters vs Aliens",
    "The Addams Family" = "Addams Family",
    "The Croods: A New Age" = "Croods 2",
    "The Angry Birds Movie" = "Angry Birds",
    "How to Train Your Dragon" = "HTTYD",
    "How to Train Your Dragon 2" = "HTTYD 2",
    "The Simpsons Movie" = "Simpsons",
    "Horton Hears a Who!" = "Horton",
    "The Nightmare Before Christmas" = "Nightmare Before Christmas",
    "South Park: Bigger, Longer & Uncut" = "South Park",
    "Alvin and the Chipmunks" = "Chipmunks",
    "Alvin and the Chipmunks: The Squeakquel" = "Squeakquel",
    "Alvin and the Chipmunks: Chipwrecked" = "Chipwrecked",
    "DC League of Super-Pets" = "Super-Pets",
    "Teenage Mutant Ninja Turtles: Mutant Mayhem" = "Mutant Mayhem",
    "Meet the Robinsons" = "Robinsons",
    "The Lego Ninjago Movie" = "Lego Ninjago",
    "The Bob's Burgers Movie" = "Bob's Burgers",
    "Ice Age: Continental Drift" = "Ice Age 4",
    "The Super Mario Bros. Movie" = "Super Mario",
    "The Secret Life of Pets" = "Secret Life of Pets",
    "The Secret Life of Pets 2" = "Secret Life of Pets 2",
    "The Sword in the Stone" = "Sword in Stone",
    "Sleeping Beauty" = "Sleeping Beauty",
    "The Road to El Dorado" = "El Dorado",
    "Spider-Man: Into the Spider-Verse" = "Spider-Verse",
    "Spider-Man: Across the Spider-Verse" = "Spider-Verse 2",
    "Atlantis: The Lost Empire" = "Atlantis",
    "Despicable Me 2" = "DM 2",
    "Despicable Me 3" = "DM 3",
    "Despicable Me 4" = "DM 4",
    "Moana 2" = "Moana 2",
    "The Prince of Egypt" = "Prince of Egypt",
    "Fantastic Mr. Fox" = "Fantastic Mr Fox",
    "Mufasa: The Lion King" = "Mufasa",
    "Oliver & Company" = "Oliver & Co.",
    "One Hundred and One Dalmatians" = "101 Dalmatians",
    "Guillermo del Toro's Pinocchio" = "Pinocchio (del Toro)",
    "The Princess and the Frog" = "Princess & Frog",
    "The Little Mermaid" = "Little Mermaid",
    "Puss in Boots: The Last Wish" = "Puss in Boots 2",
    "Ice Age: Collision Course" = "Ice Age 5",
    "The Hunchback of Notre Dame" = "Hunchback",
    "Wallace & Gromit: The Curse of the Were-Rabbit" = "Wallace & Gromit",
    "Raya and the Last Dragon" = "Raya",
    "The Wild Robot" = "Wild Robot",
    "Shaun the Sheep Movie" = "Shaun Sheep",
    "Mr. Peabody & Sherman" = "Peabody & Sherman",
    "Shrek Forever After" = "Shrek 4",
    "Shrek the Third" = "Shrek 3",
    "Ice Age: Dawn of the Dinosaurs" = "Ice Age 3",
    "Ice Age: The Meltdown" = "Ice Age 2",
    "The Lion King" = "Lion King",
    "Penguins of Madagascar" = "Penguins of Madagascar",
    "PAW Patrol: The Mighty Movie" = "PAW Patrol 2",
    "The Peanuts Movie" = "Peanuts",
    "Monsters, Inc." = "Monsters Inc.",
    "The Black Cauldron" = "Black Cauldron",
    "The Boss Baby" = "Boss Baby",
    "The Adventures of Tintin" = "Tintin",
    "The Fox and the Hound" = "Fox & Hound",
    "The Rugrats Movie" = "Rugrats",
    "How to Train Your Dragon: The Hidden World" = "HTTYD 3",
    "The Smurfs 2" = "Smurfs 2",
    "The SpongeBob SquarePants Movie" = "SpongeBob",
    "The Addams Family 2" = "Addams Family 2",
    "The Lord of the Rings" = "Lord of the Rings",
    "Ralph Breaks the Internet" = "Ralph 2",
    "Spirit Untamed" = "Spirit Untamed",
    "Arthur Christmas" = "Arthur Christmas",
    "Cars 2" = "Cars 2",
    "The Emoji Movie" = "Emoji Movie",
    "The Garfield Movie" = "Garfield",
    "Inside Out 2" = "Inside Out 2",
    "Jimmy Neutron: Boy Genius" = "Jimmy Neutron",
    "Kubo and the Two Strings" = "Kubo",
    "Minions: The Rise of Gru" = "Minions 2",
    "The Nut Job" = "Nut Job",
    "Peter Rabbit 2: The Runaway" = "Peter Rabbit 2",
    "Teenage Mutant Ninja Turtles" = "TMNT",
    "Who Framed Roger Rabbit" = "Roger Rabbit"
)

##########################################################################
# Read data
##########################################################################

df <- read.delim(
    infile,
    sep="\t",
    quote="",
    stringsAsFactors=FALSE,
    check.names=FALSE
)

df$women_pct <- as.numeric(df[["% screentime, female (all)"]])
df$year <- as.numeric(df$imdb_date)

df <- subset(
    df,
    !is.na(women_pct) &
    !is.na(year) &
    year >= 1980
)

df$franchise[df$franchise == ""] <- NA

##########################################################################
# Filter franchises
##########################################################################

franchise_counts <- table(df$franchise)

multi_franchises <-
    names(franchise_counts[franchise_counts >= MIN_FRANCHISE_SIZE])

df <- subset(
    df,
    franchise %in% multi_franchises
)

##########################################################################
# Labels
##########################################################################

df$label <- ifelse(
    is.na(df$movie_name) | df$movie_name == "",
    df$movie_id,
    df$movie_name
)

df$label <- ifelse(
    df$label %in% names(movie_abbreviations),
    movie_abbreviations[df$label],
    df$label
)

##########################################################################
# Order facets chronologically
##########################################################################

franchise_order <-
    names(sort(tapply(df$year,
                      df$franchise,
                      min)))

df$franchise <- factor(
    df$franchise,
    levels=franchise_order
)

df <- df[order(df$franchise, df$year), ]

##########################################################################
# Plot
##########################################################################

p <- ggplot(
    df,
    aes(
        x=year,
        y=women_pct
    )
) +

geom_line(
    aes(group=franchise),
    color=line_color,
    linewidth=0.9
) +

geom_point(
    color=line_color,
    size=3.6
) +

geom_text_repel(
    aes(label=label),
    size=3.3,
    color="grey25",
    box.padding=0.8,
    point.padding=0.8,
    force=2,
    min.segment.length=Inf,
    max.overlaps=Inf,
    segment.color=NA,
    seed=1
) +

facet_wrap(
    ~franchise,
    ncol=3
) +

coord_cartesian(
    ylim=c(0,55)
) +

scale_x_continuous(
    breaks=scales::pretty_breaks(3)
) +

scale_y_continuous(
    breaks=c(10,20,30,40,50),
    limits=c(0,55)
) +

labs(
    x=NULL,
    y="% Screentime, Female"
) +

theme_minimal(base_size=15) +

theme(
    panel.grid.minor=element_blank(),
    strip.text=element_text(face="bold", size=12),
    axis.text.x=element_text(
        angle=45,
        hjust=1,
        size=10.5
    ),
    axis.text.y=element_text(size=10.5),
    axis.title.y=element_text(size=15),
    plot.margin=margin(8,8,8,8),
    panel.spacing.x=unit(1.2,"cm"),
    panel.spacing.y=unit(0.8,"cm")
)

##########################################################################
# Save editable vector PDF
##########################################################################

ggsave(
    outfile,
    p,
    width=8.5,
    height=11,
    units="in",
    device=pdf,
    limitsize=FALSE
)

cat(sprintf("Wrote %s\n", outfile))
