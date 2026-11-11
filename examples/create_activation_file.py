"""Create a tiny synthetic activation NPZ for exercising the data pipeline."""
import numpy as np
rng = np.random.default_rng(3)
n, d = 24, 16
task_id = np.array([f"task-{i//4}" for i in range(n)])
category = np.array(["looping"] * n)
role = np.array(["failing" if i%2==0 else "successful" for i in range(n)])
progress_bin = np.array([0] * n)
activations = rng.normal(size=(n,d)).astype("float32")
activations[role == "failing", 0] += 2.0
np.savez_compressed("activations.npz", task_id=task_id, category=category,
                    role=role, progress_bin=progress_bin, activations=activations)
