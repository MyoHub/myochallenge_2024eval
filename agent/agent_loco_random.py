import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import os
import pickle
import time
import copy

import numpy as np

time.sleep(60)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    channel = grpc.insecure_channel("environment:8086")
else:
    channel = grpc.insecure_channel("localhost:8086")



def pack_for_grpc(entity):
    return pickle.dumps(entity)


def unpack_for_grpc(entity):
    return pickle.loads(entity)


stub = evaluation_pb2_grpc.EnvironmentStub(channel)


print( unpack_for_grpc(
        stub.get_obsdict(
            evaluation_pb2.Package(SerializedEntity=pack_for_grpc(None))
        ).SerializedEntity
        ))
print(f'Len OBS: {len(unpack_for_grpc( stub.reset(evaluation_pb2.Package(SerializedEntity=pack_for_grpc(None))).SerializedEntity))}')

print("Original Ouput Obs keys:"+str(unpack_for_grpc(
        stub.get_obsdict(
            evaluation_pb2.Package(SerializedEntity=pack_for_grpc(None))
        ).SerializedEntity
        )))

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

flag_completed = None # this flag will detect then the whole eval is finished
repetition = 0

osl_dict = generate_sample_OSL_param()

while not flag_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    counter = 0
    repetition +=1
    while not flag_trial :

        if counter == 0:
            print('LOCO-OSL : Start Resetting the environment and get 1st obs')
            obs = unpack_for_grpc(
            stub.reset(
                evaluation_pb2.Package(SerializedEntity=pack_for_grpc(osl_dict))
            ).SerializedEntity
            )

        action = np.random.rand(80)

        ## stub gets info from the environment
        p = evaluation_pb2.Package(SerializedEntity=pack_for_grpc(action))
        s = stub.act_on_environment(p)
        ss = s.SerializedEntity
        base = unpack_for_grpc(ss)


        obs =  base["feedback"][0]

        flag_trial = base["feedback"][2]
        flag_completed = base["eval_completed"]

        print(f"LOCO-OSL : Random Agent Feedback iter {counter} -- solved: {flag_trial}")
        print("*" * 100)
        counter +=1
