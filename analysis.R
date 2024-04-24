library(tidyverse)
df <- read.csv("data.csv") %>%
  mutate(model = ifelse(model == "control", "Experimental", "Control")) %>%
  filter(id == 0)

df_evaluation <- df %>%
  mutate(iter = floor(iter / 10.0) * 10.0) %>%
  group_by(model, id, iter) %>%
  summarize(evaluation = mean(evaluation))

plot <- ggplot(
  data = df_evaluation %>% mutate(
    Model = model,
    Evaluation = evaluation,
    Iteration = iter,
  ),
  mapping = aes(x = Iteration, y = Evaluation, color = Model),
) +
  geom_smooth(method = "loess", se = FALSE) +
  theme_minimal()
ggsave("images/graph_evaluation.svg", plot = plot, width = 10, height = 8)
ggsave(
  "paper/svg-graphs/graph_evaluation.svg",
  plot = plot,
  width = 8,
  height = 6
)
ggsave("images/graph.png", plot = plot, width = 10, height = 8)

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
  facet_wrap(~Model, scales = "free_y", dir = "v") +
  geom_vline(xintercept = 100) +
  theme_minimal()

ggsave("paper/svg/graph_directional.svg", plot = plot, width = 4, height = 4)
ggsave(
  "paper/svg-graphs/graph_directional.svg",
  plot = plot,
  width = 10,
  height = 8
)
ggsave("images/graph_directional.png", plot = plot, width = 7, height = 5)
