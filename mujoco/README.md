# Setting up MuJoCo

## native mujoco
- Download the mujoco libraries from website https://www.roboti.us/index.html
- Obtain mujoco license from here https://www.roboti.us/license.html
- Unzip binaries to : `~/.mujoco/` so that you have a path like: `~/.mujoco/mjpro150/bin/`
- Place license keys in `~/.mujoco/mjkey.txt` and `~/.mujoco/mjpro150/bin/mjkey.txt`
- Check if mujoco is working by:
```
$ cd ~/.mujoco/mjpro150/bin
$ ./simulate ../model/humanoid.xml
```

## mujoco_py (OpenAI python wrapper)
- Do the above set-up to first get native mujoco working.
- Follow instructions in this repo: https://github.com/aravindr93/mujoco_envs_private. Talk to Aravind if you don't have access to this repo.

## mjulia (Kendall's Julia wrapper)
- TODO(Kendall): add instructions for this
- repo: https://github.com/klowrey/mjulia

## mjpy (Deepmind python wrapper)
- Not yet released. To be done.