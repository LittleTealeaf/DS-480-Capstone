from agent import ExpAgent
import time
import tensorflow as tf

agent = ExpAgent(
    layer_sizes=[500,500,250,125],
    width=3,
    height=3,
    target_update_interval=25,
    step_update_interval=2,
    create_training_seed=True,
    evaluate_on_training=True,
    gamma= lambda _: 0.7,
    seed=22
)

ITERATIONS = 10

data = []
elapsed_time = 0
completed_iterations = 0
TOTAL_ITERATIONS = ITERATIONS * 2

for model in ["control", "experimental"]:
    for i in range(ITERATIONS):
        start = time.time()

        agent.populate_replay(100)
        agent.train(500)
        ev, freq = agent.evaluate(50)
        data.append([model, i, ev, freq[0], freq[1], freq[2], freq[3]])
        if agent.has_nan_inf():
            print("NaN or Inf Found")
            exit()

        end = time.time()

        elapsed = end - start
        elapsed_time += elapsed
        completed_iterations += 1

        avg_time = elapsed_time / completed_iterations
        estimated_left = (TOTAL_ITERATIONS - completed_iterations) * avg_time
        hours = estimated_left // 3600
        minutes = (estimated_left % 3600) // 60
        seconds = estimated_left % 60

        fmt = "Iter {}, ({:.9f}) [{:.2f},{:.2f},{:.2f},{:.2f}] Est Left: {}h {}m {:.2f}s"
        print(
            fmt.format(
                i, ev, freq[0], freq[1], freq[2], freq[3], hours, minutes, seconds
            )
        )
    agent = agent.to_normal_agent()

with open("data.csv", "w") as file:
    file.write("model,iter,evaluation,up,down,right,left\n")
    file.writelines([",".join([str(i) for i in row]) + "\n" for row in data])
