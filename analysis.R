library(tidyverse)
df <- read.csv("data.csv") %>%
  mutate(model = ifelse(model == "control", "Experimental", "Control")) %>%
  filter(iter < 2000)

plot <- ggplot(
  data = df %>% mutate(
    Model = model,
    Evaluation = evaluation,
    Iteration = iter,
    Label = if_else(iter == max(iter), as.character(model), NA_character_)
  ),
  mapping = aes(x = Iteration, y = Evaluation, color = Model),
) +
  geom_smooth(method = "loess", se = FALSE, show.legend = FALSE) +
  geom_label(
    mapping = aes(label = Label),
    nudge_x = -1,
    na.rm = TRUE,
    show.legend = FALSE
  ) +
  theme_minimal()
ggsave("images/graph.png", plot = plot, width = 10, height = 8)
# look at matplotlib
