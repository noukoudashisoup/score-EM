import subprocess 
import time
import numpy as np
from itertools import product
from collections import OrderedDict
from copy import copy


parts = ["gpu"]
njob = 1

slurm = True
ps = []

single_conditions = OrderedDict(
seed = range(20,22),
dist = ["N"],
sigma = [1.0],
init_noise = [1.0],
pi = [0.1,0.2,0.3,0.4,0.5],
noise_interval = [1],
mean = [-10,10],
inner_steps = [3000],
alg = ["lang", "svgd", "spos"],
#alg = ["spos"],
noise_type = ["fixed", "decreasing"],
#noise_type = ["fixed"]
)

device = 0
all_conditions = list(product(*single_conditions.values()))

if __name__ == "__main__":
    for c in all_conditions:
        d = OrderedDict(zip(list(single_conditions.keys()), c))
        globals().update(d)

        command = "  ".join([f"--{k} {v}" for (k, v) in d.items()])
        print(command)

        job_str = "%.1f_%.1f_%d"%(pi, sigma, seed)

        base_command = "python run_svgd.py " + command + "\n"

        with open("run_job.sh", "w") as f:
                
                f.write("#!/bin/bash\n")
                f.write("#SBATCH --partition=%s\n" % parts[seed % len(parts)])
                f.write("#SBATCH --ntasks=%d\n"%4)
                f.write("#SBATCH --gres=gpu:1\n")
                #f.write("#SBATCH --nice=0%d\n"%4)
                f.write("#SBATCH --job-name=%s\n" % job_str)
                f.write("#SBATCH --output=slurm_output/%A \n")
                # f.write("#SBATCH --error=/nfs/ghome/live/kevinli/Code/cwgan/error/%s\n" % job_str)
                f.write("source activate score_EM\n")
                
                if slurm:
                    f.write(base_command)
                else:
                    f.write("CUDA_VISIBLE_DEVICES=%d " %(device % 3) + base_command)
                f.flush()
            
        device += 1
        if slurm:
            p = subprocess.Popen(["sbatch", "run_job.sh"])
        else:
            p = subprocess.Popen(["./run_job.sh"])

        time.sleep(0.1)
        ps.append(p)
        print("sdf")

        if not slurm:
            if len(ps) == njob:
                for p in ps:
                    p.wait()
                ps = []

