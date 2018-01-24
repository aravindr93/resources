from npg_revisited.utils.gym_env import GymEnv
from npg_revisited.policies.autodiff_gaussian_mlp import MLP
from npg_revisited.baselines.quadratic_baseline import FullQuadraticBaseline
from npg_revisited.baselines.mlp_baseline import MLPBaseline
from npg_revisited.algos.TNPG import TNPG
from npg_revisited.utils.train_agent import train_agent
import time as timer
import mujoco_envs
SEED = 100

e = GymEnv('HalfCheetah-v2')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
#baseline = MLPBaseline(e.spec, reg_coef=1e-4, epochs=1, batch_size=32)
baseline = FullQuadraticBaseline(e.spec)
agent = TNPG(e, policy, baseline, normalized_step_size=0.05, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='mujoco_test_results_master',
            env_name=e.env_id,
            agent=agent,
            seed=SEED,
            niter=20,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=5,
            sample_mode='trajectories',
            min_traj=20,
            max_traj=20,
            save_freq=5)
print("time taken = %f" % (timer.time()-ts))
