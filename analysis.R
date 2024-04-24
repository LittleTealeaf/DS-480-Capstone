library(tidyverse)
df <- read.csv("data.csv") %>%
  mutate(model = ifelse(model == "control", "Experimental", "Control")) %>%
  filter(id == 0) %>%
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
  geom_point() +
  theme_minimal()
ggsave("images/graph.svg", plot = plot, width = 10, height = 8)
ggsave("images/graph.png", plot = plot, width = 10, height = 8)
# look at matplotlib

df_directional <- df %>%
  pivot_longer(
    cols = c(up, down, left, right),
    names_to = "direction",
    values_to = "count"
  )

plot <- ggplot(
  data = df_directional %>%
    mutate(
      Model = model,
      Evaluation = evaluation,
      Iteration = iter,
      Direction = direction,
      Count = count
    ) %>%
    filter(iter < 125 & iter > 75),
  mapping = aes(
    x = Iteration,
    y = Count,
    color = Direction
  )
) +
  geom_line() +
  facet_wrap(~ Model, scales = "free_y", dir = "v") +
  geom_vline(xintercept = 100) +
  theme_minimal()
ggsave("images/graph_directional.svg", plot = plot, width = 4, height = 4)
ggsave("images/graph_directional.png", plot = plot, width = 7, height = 5)


