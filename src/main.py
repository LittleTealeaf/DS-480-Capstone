from agents import Agent, ExpAgent


agent = Agent(
    [1000, 1000, 1000],
    width=5,
    height=5,
    target_update_frequency=100,
    updates_per_step=1,
)

data = []

for model in ["control", "experimental"]:
    for i in range(10_000):
        agent.populate_replay(100)
        agent.train(100)
        ev, freq = agent.evaluate(50)
        data.append([model, i, ev, freq[0], freq[1], freq[2], freq[3]])
        print("Iter ", i, " ", ev, " ", freq)

    agent = agent.as_exp_agent()

with open("data.csv", "w") as file:
    file.write("model,iter,evaluation,up,down,right,left\n")
    file.writelines([",".join([str(i) for i in row]) + "\n" for row in data])
