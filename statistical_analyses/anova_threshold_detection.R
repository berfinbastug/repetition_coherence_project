library(tidyr)
library(dplyr)
library(ez)
library(apaTables)
library(ggpubr)
library(rstatix)

# Set working directory and read data
setwd("/Users/bastugb/Desktop/repetition_coherence_project/statistical_analyses")
data <- read.csv("threshold_50_detection_for_r.csv")
data$unitdur <- factor(data$unitdur)

ggqqplot(data, 'threshold_percentage', facet.by = "unitdur")


ez_results <- ezANOVA(
  data = data,
  dv = .(threshold_percentage),
  wid = .(participant_id),
  within = .(unitdur),
  detailed = TRUE,
  type = 3
)


print(ez_results)

# PRESENT ANOVA RESULTS IN A CLEAN APA STYLE WAY WITH GG CORRECTION
apa.ezANOVA.table(ez_results, correction = "GG", table.title = "ANOVA Results")


summary_across_participants <- data %>%
  group_by(unitdur, participant_id) %>%
  summarise(threshold_percentage = mean(threshold_percentage), .groups = "drop") %>%
  group_by(unitdur) %>%
  summarise(
    mean = mean(threshold_percentage),
    sd   = sd(threshold_percentage),
    sem  = sd(threshold_percentage) / sqrt(n()),
    n    = n(),
    .groups = "drop"
  )

print(summary_across_participants)




# Post hoc pairwise comparisons (paired t-tests with Bonferroni correction)
data <- data %>%
  mutate(
    unitdur = as.factor(unitdur),
    participant_id = as.factor(participant_id)
  )

posthoc_results <- data %>%
  pairwise_t_test(
    threshold_percentage ~ unitdur,
    paired = TRUE,
    subject = "participant_id",
    p.adjust.method = "bonferroni"
  )

# Print results
posthoc_results
