import wandb
api = wandb.Api()

# run = api.run("songlab/FACED_mlp_rawcode/r2_f8")
# if run.state == "finished":
#     for i, row in run.history().iterrows():
#       print(row["_timestamp"], row["accuracy"])

import numpy as np
acc = [82.5, 61.59722137451172, 71.9940414428711, 68.98809814453125,69.19642639160156, 61.43849182128906, 58.08531951904297, 50.079368591308594,58.55158615112305, 55.253971099853516]
print(np.mean(acc))
# 53.779762268066406  63.7734130859375