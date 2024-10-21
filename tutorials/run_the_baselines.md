# Run the baselines
This year we provide several baselines:
* Manipulation track with [SB3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) and curriculum learning. 
* A reflex-based locomotion controller, see [here](https://myosuite.readthedocs.io/en/latest/baselines.html#myolegreflex-baseline).

These baselines will not give you good task performance or win the challenge for you, but they provide a nice starting point.

To run the sb3-baselines with hydra, you need to install:

``` bash
pip install stable-baselines3[extra]
pip install hydra-core==1.1.0 hydra-submitit-launcher submitit
#optional
pip install tensorboard wandb
```
Take a look [here](https://stable-baselines3.readthedocs.io/en/master/guide/install.html) if you run into issues.
The requirements for the reflex-based baseline are contained in the above link.

## Manipulation Track
This sb3-baseline will lift the cube upwards with the myoArm and grab it towards the MPL limb.

First, get the commands to run sb3 locally, or download the weights locally from Google Drive by navigating to the [myosuite/agents](https://github.com/MyoHub/myosuite/tree/main/myosuite/agents) repository:
```
sh train_myosuite.sh myochal local sb3
sh train_myosuite.sh myochal local sb3 baseline
```

You can now resume training from a previous checkpoint by adding +job_name=checkpoint.pt to the command line.
```
python hydra_sb3_launcher.py --config-path config --config-name hydra_myochal_sb3_ppo_config.yaml env=myoChallengeBimanual-v0 job_name=checkpoint.pt
```

A complete tutorial can be found here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AqC1Y7NkRnb2R1MgjT3n4u02EmSPem88?usp=sharing)


## Locomotion track
This deprl-baseline will try to stand around and slowly move across the quad.
``` python
import gym
import myosuite, deprl

env = gym.make('myoChallengeRunTrackP2-v0')
policy = deprl.load_baseline(env)

for ep in range(5):
    print(f'Episode: {ep} of 5')
    state = env.reset()
    while True:
        action = policy(state)
        # uncomment if you want to render the task
        # env.mj_render()
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done: 
            break
```


