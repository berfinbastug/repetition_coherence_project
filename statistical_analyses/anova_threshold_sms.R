library(tidyr)
library(dplyr)
library(ez)
library(ggpubr)

# Set working directory and read data
setwd("/Users/bastugb/Desktop/repetition_coherence_project/statistical_analyses")
data <- read.csv("threshold_50_sms_for_r.csv")
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




