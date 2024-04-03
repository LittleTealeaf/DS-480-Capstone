library(tidyverse)
df <- read.csv("data.csv")

plot <- ggplot(
  data = df,
  mapping = aes(x = iter, y = evaluation, color = model)
) +
  geom_smooth()
