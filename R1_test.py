import tensorflow as tf
import numpy as np
from ModelV3 import Agent, Buffer, AgentRestore
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
import os



def get_keys(obs, action):
    feature_vector = np.zeros(len(action))
    key = np.zeros((len(feature_vector), 2))
    for i in range(len(feature_vector)):
        if action[i] >= 0:
            feature_vector[i] = 1
        if feature_vector[i] == 1:
            if obs[i] >= np.mean(obs) + np.std(obs):
                key[i][0] = 1
                key[i][1] = 1
            elif obs[i] < np.mean(obs) + np.std(obs) and obs[i] > np.mean(obs) - np.std(obs):
                key[i][0] = 0
                key[i][1] = 1
            elif obs[i] <= np.mean(obs) - np.std(obs):
                key[i][0] = 1
                key[i][1] = 0
    key = np.array(key, dtype=int)
    key = np.reshape(key, (1, -1))[0]
    return key, feature_vector



def get_similarity_feature_value(key_alice, key_bob, feature_vector_alice, feature_vector_bob, check_key_uti):
    sim = 0
    feature=0
    idle = 0
    wrong=0
    key_alice_t = np.reshape(key_alice, (-1, 2))
    key_bob_t = np.reshape(key_bob, (-1, 2))
    for i in range(len(key_alice_t)):
        flag = (key_alice_t[i]==key_bob_t[i]).all()
        if flag and feature_vector_alice[i]== feature_vector_bob[i]:
            sim+=1
        # if flag and np.sum(key_alice_t[i])==0:
        #     idle+=1
        if flag and feature_vector_alice[i]==1 and feature_vector_bob[i]==1:
            feature+=1
        if flag == False or (feature_vector_alice[i] != feature_vector_bob[i]):
            wrong+=1
        if flag and feature_vector_alice[i]==0 and feature_vector_bob[i]==0 and check_key_uti[i]!=0:
            idle +=1

    return sim/int(len(feature_vector_bob)), feature/int(len(feature_vector_bob)), wrong/int(len(feature_vector_bob)), idle/int(len(feature_vector_bob))

def get_check(obs_alice, obs_bob):
    key = np.zeros((len(obs_bob), 2))
    uti = np.zeros(len(obs_bob))
    for i in range(len(obs_bob)):
        if obs_alice[i] >= np.mean(obs_alice)+np.std(obs_alice) and obs_bob[i] >= np.mean(obs_bob)+np.std(obs_bob):
            key[i][0] = 1
            key[i][1] = 1
            uti[i] = 1
        elif (obs_alice[i] < np.mean(obs_alice)+np.std(obs_alice) and obs_alice[i] > np.mean(obs_alice)-np.std(obs_alice)) and  (obs_bob[i] < np.mean(obs_bob)+np.std(obs_bob) and obs_bob[i] > np.mean(obs_bob)-np.std(obs_bob)):
            key[i][0] = 0
            key[i][1] = 1
            uti[i] = 1
        elif obs_alice[i] <= np.mean(obs_alice)-np.std(obs_alice) and obs_bob[i] <= np.mean(obs_bob)-np.std(obs_bob):
            key[i][0] = 1
            key[i][1] = 0
            uti[i] = 1
    key = np.reshape(key, (1, -1))
    key = np.array(key, dtype=int)[0]
    return key, uti

def get_reward(similarity, feature_value, wrong_value):
    # c1 = 0
    # for i in range(len(feature_vector)):
    #     if feature_vector[i]!=check_key_uti[i]:
    #         c1 +=1/int(len(feature_vector))
    c2 = 0
    if similarity>=1:
        c2 = feature_value-wrong_value
    r = feature_value-wrong_value
    return r



def run():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    K.set_session(tf.compat.v1.Session(config=config))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    tf.random.set_seed(1)
    np.random.seed(1)
    attribute_length = 10
    key_length = attribute_length
    num_agent = 2

    action_bound = 1

    df = pd.read_csv('uP0427.csv')
    rows = []
    for i in range(1, len(df)):
        row = df.values[i][0]
        row = str(row).split(';')[0:6]
        alice_temp, bob_temp = float(row[2]), float(row[4])
        rows.append([alice_temp, bob_temp])
    rows = np.array(rows[0:2000])
    alice_data = np.reshape(rows[:, 0], (-1, attribute_length))
    bob_data = np.reshape(rows[:, 1], (-1, attribute_length))
    step_length = len(alice_data)-attribute_length-1
    #step_length = 100
    epoch = 24900
    alice = AgentRestore(name='./R1/Alice', num_obs=attribute_length , num_act=key_length, num_agent=num_agent, epoch=epoch)
    bob = AgentRestore(name='./R1/Bob', num_obs=attribute_length , num_act=key_length, num_agent=num_agent, epoch=epoch)
    # Discount factor for future rewards
    reward_epoch = 0
    sim_epoch = 0
    fea_epoch = 0
    wrong_count = 0
    for step in range(step_length):
        alice_d = alice_data[step]
        bob_d = bob_data[step]
        alice_attribute = alice_d
        bob_attribute = bob_d
        obs_alice = (alice_attribute-min(alice_attribute))/(max(alice_attribute)-min(alice_attribute))
        obs_bob = (bob_attribute-min(bob_attribute))/(max(bob_attribute)-min(bob_attribute))
        state = np.vstack((obs_alice, obs_bob))
        act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
        act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
        key_alice, feature_vector_alice = get_keys(alice_attribute, act_alice)
        key_bob, feature_vector_bob = get_keys(bob_attribute, act_bob)

        check_key, check_key_uti = get_check(alice_attribute, bob_attribute)
        similarity, feature_value, wrong_value, idle_value = get_similarity_feature_value(key_alice, key_bob, feature_vector_alice,
                                                                                          feature_vector_bob,
                                                                                          check_key_uti)
        reward = get_reward(similarity, feature_value, wrong_value)

        sim_epoch += similarity
        fea_epoch += feature_value
        reward_epoch += reward
        # print('Alice Action:', act_alice)
        # print('Bob Action:', act_bob)
        print('step:', step)
        print('Ali:', key_alice, feature_vector_alice)
        print('Bob:', key_bob, feature_vector_bob)
        print('Chk:', check_key, check_key_uti)
        if similarity<=0.8:
            print('Wrong!')
            wrong_count+=1
        print('reward:%s, sim:%s, feature:%s, wrong:%s' % (reward, similarity, feature_value, wrong_value))
        print('\n')
    #print(reward_epoch / step_length)
    print('Ave Bar:', sim_epoch / step_length)
    print('Wrong Count:', wrong_count)
    print("\n")




if __name__ == '__main__':
    run()