from agents import Agent, ExpAgent

agent = Agent([10,10,10])

for i in range(100):
    print("iter ", i)
    agent.populate_replay(1000)
    agent.train(500)
    print(agent.evaluate(10))
