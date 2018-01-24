from npg_revisited.utils.gym_env import GymEnv
from npg_revisited.policies.gaussian_mlp import MLP
from npg_revisited.baselines.quadratic_baseline import FullQuadraticBaseline
from npg_revisited.baselines.mlp_baseline import MLPBaseline
from npg_revisited.baselines.quadratic_baseline import FullQuadraticBaseline
from npg_revisited.algos.TNPG import TNPG
from npg_revisited.utils.train_agent import train_agent
import time as timer
import mujoco_envs
SEED = 100

e = GymEnv('Hopper-v2')
policy = MLP(e.spec, hidden_sizes=(32,32), init_log_std=-0.1, seed=SEED)
baseline = MLPBaseline(e.spec, reg_coef=1e-4, epochs=1, batch_size=32)
#baseline = FullQuadraticBaseline(e.spec)
agent = TNPG(e, policy, baseline, normalized_step_size=0.05, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='mujoco_test_results',
            agent=agent,
            seed=SEED,
            niter=50,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=5,
            sample_mode='samples',
            num_samples=10000,
            save_freq=1)
print("time taken = %f" % (timer.time()-ts))
