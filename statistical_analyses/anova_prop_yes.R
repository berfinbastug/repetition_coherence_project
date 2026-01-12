library(tidyr)
library(dplyr)
library(ez)
library(apaTables)

# Set working directory and read data
setwd("/Users/bastugb/Desktop/repetition_coherence_project/statistical_analyses")
data <- read.csv("detection_performance_for_r.csv")
data$unitdur <- factor(data$unitdur)
data$participant_id <- factor(data$participant_id)
data$percentage <- factor(data$percentage)

# 3 X 10 within-subjects ANOVA 
anova_results <- ezANOVA(
  data = data,
  dv = .(mean_yes), 
  wid = .(participant_id),
  within = .(unitdur, percentage),
  type = 3,
  detailed = TRUE
)

anova_results

# PRESENT ANOVA RESULTS IN A CLEAN APA STYLE WAY WITH GG CORRECTION
apa.ezANOVA.table(anova_results, correction = "GG", table.title = "ANOVA Results")



# NOW, FA RATES 
# THEY ARE COMPARABLE ACROSS THREE UNIT DURATIONS
# BECAUSE WHEN THE COHERENCE IS 0, THEY ARE ALL RANDOM, HENCE THE SAME
# HERE I AM SHOWING THIS CLAIM STATISTICALLY
# FIRST, I NEED TO DO SOME FILTERING 
# PERCENTAGE == 0
fa_data <- data %>% filter(percentage == "0")


# Compute false alarm rate
fa_data <- fa_data %>%
  mutate(false_alarm = 1 - n_correct / n_trial)


# Run one-way repeated measures ANOVA across unit duration
fa_anova <- ezANOVA(
  data = fa_data,
  dv = .(false_alarm),
  wid = .(participant_id),
  within = .(unitdur),
  type = 3,
  detailed = TRUE
)

fa_anova



# HERE I DO NOT NEED TO PRESENT IT WITH GG CORRECTIONS BECAUSE SPHERICITY IS 
# NOT VIOLATED

# I WILL PRESENT ANOVA RESULTS TOGETHER WITH MEAN VALUES.
# I NEED TO SUMMARIZE THE DATA
# Summarize mean and SD of false alarm by unit duration
# Step 1: compute participant-level FA per unit duration
fa_participant_summary <- fa_data %>%
  group_by(participant_id, unitdur) %>%
  summarise(
    participant_fa = mean(false_alarm, na.rm = TRUE),
    .groups = "drop"
  )

# Step 2: compute mean and SD across participants
fa_summary <- fa_participant_summary %>%
  group_by(unitdur) %>%
  summarise(
    mean_fa = mean(participant_fa, na.rm = TRUE),
    sd_fa   = sd(participant_fa, na.rm = TRUE),
    n = n()   # number of participants
  )

fa_summary
