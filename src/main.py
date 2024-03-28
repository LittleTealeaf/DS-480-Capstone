from agents import Agent, ExpAgent

agent = Agent([500, 500, 500, 500])

for i in range(100):
    print("iter ", i)
    if i % 10 == 0:
        agent.update_target()
    agent.populate_replay(1000)
    agent.train(500)
    print(agent.evaluate(10))
