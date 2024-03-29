from agents import Agent, ExpAgent

agent = ExpAgent([500, 500], width=3, height=3, target_update_frequency=100)

for i in range(10_000):
    agent.populate_replay(50)
    agent.train(100)
    print("Iter ", i, " ", agent.evaluate(10))

    if agent.nan_check():
        print("HIT NAN")
        break
