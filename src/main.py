from agents import Agent, ExpAgent


agent = ExpAgent(
    [1000, 1000, 1000, 1000],
    width=5,
    height=5,
    target_update_frequency=25,
    updates_per_step=4,
)

data = []

for i in range(10_000):
    agent.populate_replay(100)
    agent.train(100)
    evals, freq = agent.evaluate(10)
    data.append(evals)
    print("Iter ", i, " ", evals, " ", freq)

    if agent.nan_check():
        print("HIT NAN")
        break

from matplotlib import pyplot as plt

plt.plot(data)
plt.show()
