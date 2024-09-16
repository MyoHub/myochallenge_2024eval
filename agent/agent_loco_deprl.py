import os
import pickle
import time
from utils import LocoRemoteConnection, DummyLocoEnv
import numpy as np
import copy

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym

def get_custom_observation(rc):
    """
    Use this function to create an observation vector from the 
    environment provided observation dict for your own policy.
    By using the same keys as in your local training, you can ensure that 
    your observation still works.
    """
    # example of obs_keys for deprl baseline
    obs_keys = [
      'terrain',        
      'internal_qpos',
      'internal_qvel',
      'grf',
      'torso_angle',
      'socket_force',
      'torso_angle',
      'model_root_pos',
      'model_root_vel',
      'muscle_length',
      'muscle_velocity',
      'muscle_force',
      'act',
      'act_dot'
    ]

    obs_dict = rc.get_obsdict()
    # add new features here that can be computed from obs_dict
    # obs_dict['qpos_without_xy'] = np.array(obs_dict['internal_qpos'][2:35].copy())

    return rc.obsdict2obsvec(obs_dict, obs_keys)

def generate_sample_OSL_param():
    """
    Example function to generate the dictionary for the
    """
    BODY_WEIGHT = 78.6689 * 9.81 # Total body mass (incl. OSL leg) * gravity

    temp_dict = {}
    temp_dict['e_stance'] = {}
    temp_dict['e_stance']['gain'] = {}
    temp_dict['e_stance']['threshold'] = {}
    temp_dict['e_stance']['gain']['knee_stiffness'] = 99.372
    temp_dict['e_stance']['gain']['knee_damping'] = 3.180
    temp_dict['e_stance']['gain']['knee_target_angle'] = np.deg2rad(5)
    temp_dict['e_stance']['gain']['ankle_stiffness'] = 19.874
    temp_dict['e_stance']['gain']['ankle_damping'] = 0
    temp_dict['e_stance']['gain']['ankle_target_angle'] = np.deg2rad(-2)
    temp_dict['e_stance']['threshold']['load'] = (0.25 * BODY_WEIGHT, 'above')
    temp_dict['e_stance']['threshold']['ankle_angle'] = (np.deg2rad(6), 'above')

    temp_dict['l_stance'] = {}
    temp_dict['l_stance']['gain'] = {}
    temp_dict['l_stance']['threshold'] = {}
    temp_dict['l_stance']['gain']['knee_stiffness'] = 99.372
    temp_dict['l_stance']['gain']['knee_damping'] = 1.272
    temp_dict['l_stance']['gain']['knee_target_angle'] = np.deg2rad(8)
    temp_dict['l_stance']['gain']['ankle_stiffness'] = 79.498
    temp_dict['l_stance']['gain']['ankle_damping'] = 0.063
    temp_dict['l_stance']['gain']['ankle_target_angle'] = np.deg2rad(-20)
    temp_dict['l_stance']['threshold']['load'] = (0.15 * BODY_WEIGHT, 'below')

    temp_dict['e_swing'] = {}
    temp_dict['e_swing']['gain'] = {}
    temp_dict['e_swing']['threshold'] = {}
    temp_dict['e_swing']['gain']['knee_stiffness'] = 39.749
    temp_dict['e_swing']['gain']['knee_damping'] = 0.063
    temp_dict['e_swing']['gain']['knee_target_angle'] = np.deg2rad(60)
    temp_dict['e_swing']['gain']['ankle_stiffness'] = 7.949
    temp_dict['e_swing']['gain']['ankle_damping'] = 0
    temp_dict['e_swing']['gain']['ankle_target_angle'] = np.deg2rad(25)
    temp_dict['e_swing']['threshold']['knee_angle'] = (np.deg2rad(50), 'above')
    temp_dict['e_swing']['threshold']['knee_vel'] = (np.deg2rad(3), 'below')

    temp_dict['l_swing'] = {}
    temp_dict['l_swing']['gain'] = {}
    temp_dict['l_swing']['threshold'] = {}
    temp_dict['l_swing']['gain']['knee_stiffness'] = 15.899
    temp_dict['l_swing']['gain']['knee_damping'] = 3.816
    temp_dict['l_swing']['gain']['knee_target_angle'] = np.deg2rad(5)
    temp_dict['l_swing']['gain']['ankle_stiffness'] = 7.949
    temp_dict['l_swing']['gain']['ankle_damping'] = 0
    temp_dict['l_swing']['gain']['ankle_target_angle'] = np.deg2rad(15)
    temp_dict['l_swing']['threshold']['load'] = (0.4 * BODY_WEIGHT, 'above')
    temp_dict['l_swing']['threshold']['knee_angle'] = (np.deg2rad(30), 'below')

    OSL_PARAM_LIST = {}
    for idx in np.arange(4):
        OSL_PARAM_LIST[idx] = {}
        OSL_PARAM_LIST[idx] = copy.deepcopy(temp_dict)

    return OSL_PARAM_LIST

osl_dict = generate_sample_OSL_param()

time.sleep(60)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    rc = LocoRemoteConnection("environment:8086")
else:
    rc = LocoRemoteConnection("localhost:8086")


# compute correct observation space
shape = get_custom_observation(rc).shape
rc.set_observation_space(shape)


################################################
## A -replace with your trained policy.
## HERE an example from a previously trained policy with deprl is shown (see https://github.com/MyoHub/myosuite/blob/main/docs/source/tutorials/4a_deprl.ipynb)
## additional dependencies such as gym and deprl might be needed
import deprl
policy = deprl.load_baseline(DummyLocoEnv(env_name='myoChallengeRunTrackP2-v0', stub=rc))
print('LOCO-OSL agent: policy loaded')
################################################



flag_completed = None # this flag will detect then the whole eval is finished
repetition = 0
while not flag_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    counter = 0
    repetition +=1
    while not flag_trial :

        if counter == 0:
            print('LOCO-OSL: Trial #'+str(repetition)+'Start Resetting the environment and get 1st obs')
            obs = rc.reset(osl_dict)

        ################################################
        ### B - HERE the action is obtained from the policy and passed to the remote environment
        obs = get_custom_observation(rc)
        action = policy(obs)
        ################################################

        ## gets info from the environment
        base = rc.act_on_environment(action)
        obs =  base["feedback"][0]

        flag_trial = base["feedback"][2]
        flag_completed = base["eval_completed"]

        print(f"LOCO-OSL: Agent Feedback iter {counter} -- trial solved: {flag_trial} -- task solved: {flag_completed}")
        print("*" * 100)
        counter +=1
