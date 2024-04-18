from agent import ExpAgent
import time
from random import Random

MODEL_COUNT = 2
ITERATIONS = 20_000

data = []
elapsed_time = 0
completed_iterations = 0
TOTAL_ITERATIONS = ITERATIONS * 2 * MODEL_COUNT

base_seed = Random(923)

PRINT_LINE_CONFIG = (
    "{} {} Iter {}, {:.9f} [{:.2f},{:.2f},{:.2f},{:.2f}] Rem: {}h {}m {:.2f}s"
)

for id in range(MODEL_COUNT):

    seed = base_seed.random()



    agent = ExpAgent(
        layer_sizes=[1000,1000,1000,1000],
        width=3,
        height=3,
        target_update_interval=100,
        step_update_interval=20,
        create_training_seed=False,
        create_evaluation_seed=True,
        evaluate_on_training=False,
        gamma=lambda _: 0.97,
        seed=seed,
        max_replay=500,
        use_single_maze=True,
        train_on_maze_config=False,
        squish_on_update_target=False,
        swap_on_update_target=False
    )

    for model in ["experimental", "control"]:
        for i in range(ITERATIONS):
            start = time.time()

            while len(agent.replay) < 64:
                agent.populate_replay(10)
            agent.populate_replay(100)
            agent.train(64)
            ev, freq = agent.evaluate(1)
            data.append([model, id, i, ev, freq[0], freq[1], freq[2], freq[3]])
            if agent.has_nan_inf():
                print("NaN or Inf Found")
                exit()

            agent.print_policy()

            end = time.time()

            elapsed = end - start
            elapsed_time += elapsed
            completed_iterations += 1

            avg_time = elapsed_time / completed_iterations
            estimated_left = (TOTAL_ITERATIONS -
                              completed_iterations) * avg_time
            hours = estimated_left // 3600
            minutes = (estimated_left % 3600) // 60
            seconds = estimated_left % 60

            print(
                PRINT_LINE_CONFIG.format(
                    id,
                    model,
                    i,
                    ev,
                    freq[0],
                    freq[1],
                    freq[2],
                    freq[3],
                    hours,
                    minutes,
                    seconds,
                )
            )

        agent = agent.to_normal_agent()

with open("data.csv", "w") as file:
    file.write("model,id,iter,evaluation,up,down,right,left\n")
    file.writelines([",".join([str(i) for i in row]) + "\n" for row in data])
