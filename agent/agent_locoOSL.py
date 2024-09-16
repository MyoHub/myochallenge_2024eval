import os
import pickle
import time

import copy
import numpy as np

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym


def pack_for_grpc(entity):
    return pickle.dumps(entity)

def unpack_for_grpc(entity):
    return pickle.loads(entity)

class EnvShell:

    def __init__(self, stub):

        action_len = unpack_for_grpc(
            stub.get_action_space(
                evaluation_pb2.Package(SerializedEntity=pack_for_grpc(None))
            ).SerializedEntity
        )

        obs_len = unpack_for_grpc(
            stub.get_observation_space(
                evaluation_pb2.Package(SerializedEntity=pack_for_grpc(None))
            ).SerializedEntity
        )
        self.observation_space = gym.spaces.Box(shape=(obs_len,), high=1e6, low=-1e6)
        self.action_space = gym.spaces.Box(shape=(action_len,), high=1.0, low=0.0)
        print("Action Space", self.action_space)
        print("Observation Space", self.observation_space)
        # TODO case for remapping of [-1 1] -> [0 1]


class Policy:

    def __init__(self, env):
        self.action_space = env.action_space

    def __call__(self, env):
        return self.action_space.sample()

def generateDict():
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

    # OSL_PARAM_LIST[1]['e_stance']['gain']['knee_stiffness'] = 1000

    return OSL_PARAM_LIST


time.sleep(60)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    channel = grpc.insecure_channel("environment:8086")
else:
    channel = grpc.insecure_channel("localhost:8086")

stub = evaluation_pb2_grpc.EnvironmentStub(channel)
env_shell = EnvShell(stub)
# policy = deprl.load_baseline(env_shell)
policy = Policy(env_shell)

osl_dict = generateDict() # Small test for agent

flat_completed = None
trial = 0
while not flat_completed:
    ret = 0
    print(f"LocoOSL: Start Resetting the environment and get 1st obs of iter {trial}")
    
    obs = unpack_for_grpc(
        stub.reset(
            evaluation_pb2.Package(SerializedEntity=pack_for_grpc(osl_dict))
        ).SerializedEntity
    )

    if trial == 0:
        mode = np.array([0])
    else:
        mode = np.array([1])
    
    stub.change_osl_mode(
        evaluation_pb2.Package(SerializedEntity=pack_for_grpc(mode))
    ).SerializedEntity

    flag_trial = None
    counter = 0
    for t in range(1000):
        print(
            f"Trial: {trial}, Iteration: {counter} flag_trial: {flag_trial} flat_completed: {flat_completed}"
        )

        action = env_shell.action_space.sample()
        base = unpack_for_grpc(
            stub.act_on_environment(
                evaluation_pb2.Package(SerializedEntity=pack_for_grpc(action))
            ).SerializedEntity
        )
        print(f" \t \t after step: {base['feedback'][1:4]}")
        obs = base["feedback"][0]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]
        print(
            f" \t \t after step: flag_trial: {flag_trial} flat_completed: {flat_completed}"
        )
        if flag_trial:
            print(f"Return was {ret}")
            print("*" * 100)
            break
        counter += 1
    trial += 1
