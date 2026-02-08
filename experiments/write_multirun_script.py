import os
import numpy as np   
def write_txt_commands(
    num_players: int,
    depths: list[int],
    num_samples_per_depth: int,
):
    with open(f"commands_{num_players}p.txt", "w") as f:
        for depth in depths:
            for i in range(num_samples_per_depth):
                f.write(f"/usr/bin/python3 experiments/compute_eq.py --num-players {num_players} --depth {depth} --alpha 0.3 --output results_{num_players}.pkl\n")
        
    f.close()

if __name__ == "__main__":
    write_txt_commands(num_players=12, depths=np.arange(1, 20), num_samples_per_depth=300)