from agents import Agent, ExpAgent

agent = Agent([100, 100, 100])

agent.populate_replay(10)
agent.train(10)
