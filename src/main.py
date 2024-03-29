from agents import Agent, ExpAgent

agent = ExpAgent([2000, 2000, 2000], width=10, height=10, target_update_frequency=100)

for i in range(10_000):
    agent.populate_replay(100)
    agent.train(500)
    print("Iter ", i, " ", agent.evaluate(10))

    if agent.nan_check():
        print("HIT NAN")
        break
