from agents import Agent, ExpAgent


agent = ExpAgent(
    [500, 500, 500],
    width=5,
    height=5,
    target_update_frequency=25,
    updates_per_step=4,
)

data = []

for i in range(20):
    agent.populate_replay(100)
    agent.train(100)
    evaluation = agent.evaluate(10)
    data.append(evaluation)
    print("Iter ", i, " ", evaluation)

    if agent.nan_check():
        print("HIT NAN")
        break

with open("data.csv", "w") as file:
    file.write("evaluation,up,down,right,left\n")
    file.writelines(
        [f"{ev},{freq[0]},{freq[1]},{freq[2]},{freq[3]}\n" for ev, freq in data]
    )
