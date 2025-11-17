import os

def write_config_txt():
    eps_set = [0.003, 0.005, 0.01, 0.02]
    repeat = 10
    with open('config.txt', 'w') as f:
        for i in range(repeat):
            for eps in eps_set:
                f.write(f"/usr/bin/python3 solver.py --eps {eps} --wandb-experiment {i}\n")
        f.close()

def write_sweep_config_txt():
    pass

if __name__ == "__main__":
    write_config_txt()

