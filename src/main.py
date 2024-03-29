from agents import Agent, ExpAgent
import time


agent = Agent(
    [1000, 1000, 1000],
    width=5,
    height=5,
    target_update_frequency=100,
    updates_per_step=1,
)

ITERATIONS = 10_000

data = []
elapsed_time = 0
completed_iterations = 0
TOTAL_ITERATIONS = ITERATIONS * 2


for model in ["control", "experimental"]:
    for i in range(ITERATIONS):

        start = time.time()

        agent.populate_replay(100)
        agent.train(100)
        ev, freq = agent.evaluate(50)
        data.append([model, i, ev, freq[0], freq[1], freq[2], freq[3]])

        end = time.time()

        elapsed = end - start
        elapsed_time += elapsed
        completed_iterations += 1

        avg_time = elapsed_time / completed_iterations
        estimated_left = (TOTAL_ITERATIONS - completed_iterations) * avg_time
        hours = estimated_left // 3600
        minutes = (estimated_left % 3600) // 60
        seconds = estimated_left % 60

        # print("Iter ", i, " ", ev, " ", freq, "\t", f"Est Left: {hours}h {minutes}m {seconds}s")
        print(
            "Iter {}, ({:.9}) [{:.2f},{:.2f},{:.2f},{:.2f}] Est. Left: {}h {}m {:.2f}s".format(
                i, ev, freq[0], freq[1], freq[2], freq[3], hours, minutes, seconds
            )
        )

    agent = agent.as_exp_agent()

with open("data.csv", "w") as file:
    file.write("model,iter,evaluation,up,down,right,left\n")
    file.writelines([",".join([str(i) for i in row]) + "\n" for row in data])
