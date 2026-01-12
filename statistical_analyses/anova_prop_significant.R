library(tidyr)
library(dplyr)
library(ez)
library(apaTables)


# Set working directory and read data
setwd("/Users/bastugb/Desktop/repetition_coherence_project/statistical_analyses")
data <- read.csv("prop_sig_df_for_r.csv")
data$unit_dur <- factor(data$unit_dur)
data$participant_id <- factor(data$participant_id)
data$percentage <- factor(data$percentage)


# STEP 1
data_summary <- data %>%
  group_by(participant_id, unit_dur) %>%
  summarise(mean_mean_sig = mean(mean_sig), .groups = "drop")  # summary here

# STEP 2, VARIATION AMONG PARTICIPANTS
data_summary <- data_summary %>%
  group_by(unit_dur) %>%
  summarise(
    mean_accuracy = mean(mean_mean_sig, na.rm = TRUE),
    sd_accuracy = sd(mean_mean_sig, na.rm = TRUE),
    n = n()  # optional: number of observations
  )


# 3X10 ANOVA 
anova_results <- ezANOVA(
  data = data,
  dv = .(mean_sig),  # replace with your dependent variable
  wid = .(participant_id),
  within = .(unit_dur, percentage),
  type = 3,
  detailed = TRUE
)

# PRESENT ANOVA RESULTS IN A CLEAN APA STYLE WAY WITH GG CORRECTION
apa.ezANOVA.table(anova_results, correction = "GG", table.title = "ANOVA Results")


data_summary <- data %>%
  group_by(participant_id, unit_dur) %>%
  summarise(mean_mean_sig = mean(mean_sig), .groups = "drop")  # summary here

# PRESENT ANOVA RESULTS IN A CLEAN APA STYLE WAY WITH GG CORRECTION
apa.ezANOVA.table(anova_results, correction = "GG", table.title = "ANOVA Results")
library(ggpubr)
ggqqplot(data_summary, 'mean_mean_sig', facet.by = "unit_dur")


ez_results <- ezANOVA(
  data = data_summary,
  dv = .(mean_mean_sig),
  wid = .(participant_id),
  within = .(unit_dur),
  detailed = TRUE,
  type = 3
)

print(ez_results)


summary_across_participants <- data %>%
  group_by(unit_dur, participant_id) %>%
  summarise(mean_sig = mean(mean_sig), .groups = "drop") %>%
  group_by(unit_dur) %>%
  summarise(
    mean = mean(mean_sig),
    sd   = sd(mean_sig),
    sem  = sd(mean_sig) / sqrt(n()),
    n    = n(),
    .groups = "drop"
  )

print(summary_across_participants)



