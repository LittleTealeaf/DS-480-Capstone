from agents import Agent, ExpAgent


agent = ExpAgent(
    [1000, 1000, 1000, 1000],
    width=4,
    height=4,
    target_update_frequency=25,
    updates_per_step=4,
)

for i in range(10_000):
    agent.populate_replay(100)
    agent.train(10)
    print("Iter ", i, " ", agent.evaluate(100))

    if agent.nan_check():
        print("HIT NAN")
        break
