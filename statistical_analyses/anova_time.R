library(tidyr)
library(dplyr)
library(ez)
library(ggpubr)
library(rstatix)

# Set working directory and read data
setwd("/Users/bastugb/Desktop/repetition_coherence_project/statistical_analyses")
data <- read.csv("n_cycle_for_r.csv")
data$unitdur <- factor(data$unitdur)
data$participant_id <- factor(data$participant_id)


summary_data <- data %>%
  filter(correct == 1) %>%  # Keep only correct trials
  group_by(percentage, unitdur, participant_id) %>%
  summarise(
    mean_median_ncycle = mean(median_ncycle, na.rm = TRUE),
    mean_median_rt = mean(median_rt, na.rm = TRUE),
    .groups = "drop"
  )

print(summary_data)


### N CYCLE ANALYSIS
# Filter for percentage 
filtered_data <- summary_data %>%
  filter(percentage == 1)

# ggqqplot(filtered_data, 'mean_median_ncycle', facet.by = "unitdur")
filtered_data_summary <- filtered_data %>%
  group_by(participant_id, unitdur) %>%
  summarise(mean_ncycle = mean(mean_median_ncycle, na.rm = TRUE), .groups = "drop")

# Identify participants with complete data across all unitdur groups
complete_participants <- filtered_data_summary %>%
  group_by(participant_id) %>%
  summarise(n_groups = n_distinct(unitdur)) %>%
  filter(n_groups == length(unique(filtered_data_summary$unitdur))) %>%
  pull(participant_id)

# Filter your data to only those participants
filtered_complete <- filtered_data_summary %>%
  filter(participant_id %in% complete_participants)

# Compute group-level mean + SD across participants
overall_summary <- filtered_complete %>%
  group_by(unitdur) %>%
  summarise(
    mean = mean(mean_ncycle, na.rm = TRUE),
    sd   = sd(mean_ncycle, na.rm = TRUE),
    n_participants = n(),
    .groups = "drop"
  )

overall_summary


data_wide <- filtered_complete %>%
  pivot_wider(names_from = unitdur, values_from = mean_ncycle)

# Now run Friedman test
friedman.test(as.matrix(data_wide[,-1]))


# friedman_chisq <- 10.667
# n <- 27         # for example, number of participants
# k <- 3          # number of conditions
# 
# kendalls_w <- friedman_chisq / (n * (k - 1))
# kendalls_w


# Assuming your data is in long format, with columns: participant_id, unitdur, and the dependent variable (e.g., mean_median_ncycle)
# Pairwise Wilcoxon signed-rank tests with Bonferroni correction
posthoc_results <- filtered_complete %>%
  pairwise_wilcox_test(
    mean_ncycle ~ unitdur,
    paired = TRUE,
    p.adjust.method = "bonferroni"
  )

print(posthoc_results)





summary_across_participants <- data %>%
  filter(correct == 1) %>%  # Keep only correct trials
  group_by(unitdur, percentage, participant_id) %>%
  summarise(median_ncycle = mean(median_ncycle), .groups = "drop") %>%
  group_by(unitdur, percentage) %>%
  summarise(
    mean = mean(median_ncycle),
    sd   = sd(median_ncycle),
    sem  = sd(median_ncycle) / sqrt(n()),
    n    = n(),
    .groups = "drop"
  )



### RT ANALYSIS
# Filter for percentage == 
filtered_data <- summary_data %>%
  filter(percentage == 1)

filtered_data_summary <- filtered_data %>%
  group_by(participant_id, unitdur) %>%
  summarise(mean_rt = mean(mean_median_rt, na.rm = TRUE), .groups = "drop")

# Identify participants with complete data across all unitdur groups
complete_participants <- filtered_data_summary %>%
  group_by(participant_id) %>%
  summarise(n_groups = n_distinct(unitdur)) %>%
  filter(n_groups == length(unique(filtered_data_summary$unitdur))) %>%
  pull(participant_id)


# Filter your data to only those participants
filtered_complete <- filtered_data_summary %>%
  filter(participant_id %in% complete_participants)


# Compute group-level mean + SD across participants
overall_summary <- filtered_complete %>%
  group_by(unitdur) %>%
  summarise(
    mean = mean(mean_rt, na.rm = TRUE),
    sd   = sd(mean_rt, na.rm = TRUE),
    n_participants = n(),
    .groups = "drop"
  )

overall_summary


data_wide <- filtered_complete %>%
  pivot_wider(names_from = unitdur, values_from = mean_rt)

# Now run Friedman test
friedman.test(as.matrix(data_wide[,-1]))



# Assuming your data is in long format, with columns: participant_id, unitdur, and the dependent variable (e.g., mean_median_ncycle)
# Pairwise Wilcoxon signed-rank tests with Bonferroni correction
posthoc_results <- filtered_complete %>%
  pairwise_wilcox_test(
    mean_rt ~ unitdur,
    paired = TRUE,
    p.adjust.method = "bonferroni"
  )

print(posthoc_results)


