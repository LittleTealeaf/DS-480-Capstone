from agents import Agent, ExpAgent

agent = ExpAgent([1000, 1000], width=2, height=2, target_update_frequency=100)

for i in range(10_000):
    agent.populate_replay(100)
    agent.train(500)
    print("Iter ", i, " ", agent.evaluate(10))

    if agent.nan_check():
        print("HIT NAN")
        break
