import tensorflow as tf
import numpy as np
from ModelV3 import Agent, Buffer
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
import os



def get_keys(obs, action):
    #This function takes in two parameters obs and action, which are numpy arrays.
    # It then generates a feature vector based on the action input and creates a key based on the observation obs and the generated feature vector.
    # The function returns the key and the feature vector.
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
    #This function takes in five parameters key_alice, key_bob, feature_vector_alice, feature_vector_bob, and check_key_uti.
    # It compares the keys and feature vectors of key_alice, key_bob, feature_vector_alice, and feature_vector_bob.
    # Based on this comparison, it calculates the similarity between Alice's and Bob's keys, feature value, wrong value, and idle value.
    # The function returns these values as a tuple.
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
    #This function takes in two parameters obs_alice and obs_bob, which are numpy arrays.
    # It generates a key and a utility value based on the input observations obs_alice and obs_bob.
    # The function returns the key and the utility value.
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
    #This function takes in three parameters similarity, feature_value, and wrong_value.
    # It calculates a reward based on the similarity, feature value, and wrong value. The function returns the reward.
    # c1 = 0
    # for i in range(len(feature_vector)):
    #     if feature_vector[i]!=check_key_uti[i]:
    #         c1 +=1/int(len(feature_vector))
    c2 = 0
    if similarity>=1:
        c2 = feature_value-wrong_value
    r = feature_value-wrong_value
    return r

def get_reward2(feature_vector, check_key_uti):
    #This function takes in two parameters feature_vector and check_key_uti.
    # It calculates the reward based on the feature vector and check key utility values. The function returns the reward.
    s = 0
    w = 0
    for i in range(len(feature_vector)):
        if feature_vector[i]==check_key_uti[i]:
            s+=1
        else:
            w+=1
    reward = s/int(len(feature_vector))-100*w/int(len(feature_vector))
    return reward

def get_update(feature_vector, zero_count, one_count, step):
    #This function takes in four parameters feature_vector, zero_count, one_count, and step.
    # It updates the zero_count and one_count arrays based on the feature vector and the number of steps.
    # The function returns the updated zero_count and one_count.
    for i in range(len(feature_vector)):
        if feature_vector[i]==0:
            zero_count[i]+=1
        else:
            one_count[i]+=1
    return zero_count/int(step+1), one_count/int(step+1)



def run():
    #This function sets up the TensorFlow environment and initializes the required variables.
    # It then runs a training loop for a specified number of epochs and updates the model's weights using a Deep Q-learning algorithm.
    # During each epoch, it generates random keys and feature vectors for Alice and Bob and simulates a communication session between them.
    # It then updates the model's weights based on the simulated session and prints the loss and reward values.
    # Finally, it saves the trained model and plots the reward over time.
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    K.set_session(tf.compat.v1.Session(config=config))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    tf.random.set_seed(1)
    np.random.seed(1)
    training_epoch = 100000
    memory_size = 1000000
    batch_size = 128
    attribute_length = 10
    key_length = attribute_length
    num_agent = 2

    action_bound = 1

    df = pd.read_csv('uP0427.csv')
    idx = 0
    rows = []
    for i in range(1, len(df)):
        row = df.values[i][0]
        row = str(row).split(';')[0:6]
        alice_temp, bob_temp = float(row[2]), float(row[4])
        rows.append([alice_temp, bob_temp])
    rows = np.array(rows[0:2000])/100
    #rows = np.array(rows)/100
    alice_data = np.reshape(rows[:, 0], (-1, attribute_length))
    bob_data = np.reshape(rows[:, 1], (-1, attribute_length))
    step_length = len(alice_data)-attribute_length-1

    alice = Agent(name='Alice', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, action_bound=action_bound)
    bob = Agent(name='Bob', num_obs=attribute_length, num_act=key_length, num_agent=num_agent, action_bound=action_bound)
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    start_flag = False
    std_dev = 0.4
    memory = Buffer(buffer_capacity=memory_size, num_agent=num_agent, num_obs=attribute_length,
                    num_act=key_length)
    reward_all= []
    for epoch in range(training_epoch):
        reward_epoch = 0
        sim_epoch = 0
        ran_epoch = 0
        fea_epoch = 0
        alice_zero_count = np.zeros(key_length * 2)
        alice_one_count = np.zeros(key_length * 2)
        bob_zero_count = np.zeros(key_length * 2)
        bob_one_count = np.zeros(key_length * 2)
        for step in range(step_length):
            alice_d = alice_data[step]
            bob_d = bob_data[step]
            alice_attribute = alice_d
            bob_attribute = bob_d
            obs_alice = (alice_attribute-min(alice_attribute))/(max(alice_attribute)-min(alice_attribute))
            obs_bob = (bob_attribute-min(bob_attribute))/(max(bob_attribute)-min(bob_attribute))
            state = np.vstack((obs_alice, obs_bob))
            act_alice = tf.squeeze(alice.actor(tf.expand_dims(tf.convert_to_tensor(obs_alice), 0))).numpy()
            act_alice = np.clip(np.random.normal(act_alice, std_dev), -1, 1)
            act_bob = tf.squeeze(bob.actor(tf.expand_dims(tf.convert_to_tensor(obs_bob), 0))).numpy()
            act_bob = np.clip(np.random.normal(act_bob, std_dev), -1, 1)
            key_alice, feature_vector_alice = get_keys(alice_attribute, act_alice)
            key_bob, feature_vector_bob = get_keys(bob_attribute, act_bob)
            alice_zero_count, alice_one_count = get_update(feature_vector_alice, alice_zero_count, alice_one_count, step)

            check_key, check_key_uti = get_check(alice_attribute, bob_attribute)
            similarity, feature_value, wrong_value, idle_value = get_similarity_feature_value(key_alice, key_bob, feature_vector_alice, feature_vector_bob, check_key_uti)
            reward = get_reward(similarity, feature_value, wrong_value)
            c1=0
            if np.sum(check_key_uti)!=attribute_length and np.sum(feature_vector_alice)==attribute_length and np.sum(feature_vector_bob)==attribute_length:
                c1=-1
            reward = reward+c1
            sim_epoch += similarity
            fea_epoch += feature_value
            reward_epoch += reward
            alice_d_ = alice_data[step + 1]
            bob_d_ = bob_data[step + 1]
            alice_attribute_ = alice_d_
            bob_attribute_ = bob_d_
            obs_alice_ = (alice_attribute_-min(alice_attribute_))/(max(alice_attribute_)-min(alice_attribute_))
            obs_bob_ = (bob_attribute_-min(bob_attribute_))/(max(bob_attribute_)-min(bob_attribute_))
            state_ = np.vstack((obs_alice_, obs_bob_))
            action = [act_alice, act_bob]
            memory.store(state, action, [reward], state_)
            if memory.buffer_counter >= batch_size * 30:
                start_flag = True
                #if (epoch+1)%2==0:
                #std_dev *= 0.99995
                # Get sampling range
                record_range = min(memory.buffer_counter, memory.buffer_capacity)
                # Randomly sample indices
                idx = np.random.choice(record_range, batch_size)
                bt_state = tf.convert_to_tensor(memory.state_buffer[idx])
                bt_action = tf.convert_to_tensor(memory.action_buffer[idx])
                bt_reward = tf.convert_to_tensor(memory.reward_buffer[idx])
                bt_state_ = tf.convert_to_tensor(memory.next_state_buffer[idx])
                bt_alice_obs, bt_bob_obs = bt_state[:, 0, :], bt_state[:, 1, :]
                bt_alice_obs_, bt_bob_obs_ = bt_state_[:, 0, :], bt_state_[:, 1, :]
                bt_alice_act, bt_bob_act = bt_action[:, 0, :], bt_action[:, 1, :]
                bt_alice_act_, bt_bob_act_ = alice.actor_target(bt_alice_obs_, training=True), bob.actor_target(bt_bob_obs_, training=True)
                bt_reward = tf.cast(bt_reward, tf.float32)

                with tf.GradientTape() as tape:
                    Q = alice.critic([bt_state, alice.actor(bt_alice_obs), bt_bob_act], training=True)
                    actor_loss = -tf.math.reduce_mean(Q)
                actor_grad = tape.gradient(actor_loss, alice.actor.trainable_variables)
                alice.actor_optimizer.apply_gradients(zip(actor_grad, alice.actor.trainable_variables))

                with tf.GradientTape() as tape:
                    Q = bob.critic([bt_state, bob.actor(bt_bob_obs), bt_alice_act], training=True)
                    actor_loss = -tf.math.reduce_mean(Q)
                actor_grad = tape.gradient(actor_loss, bob.actor.trainable_variables)
                bob.actor_optimizer.apply_gradients(zip(actor_grad, bob.actor.trainable_variables))

                with tf.GradientTape() as tape:
                    y = bt_reward + gamma * alice.critic_target([bt_state_, bt_alice_act_, bt_bob_act_], training=True)
                    critic_value = alice.critic([bt_state, bt_alice_act, bt_bob_act], training=True)
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                critic_grad = tape.gradient(critic_loss, alice.critic.trainable_variables)
                alice.critic_optimizer.apply_gradients(zip(critic_grad, alice.critic.trainable_variables))

                with tf.GradientTape() as tape:
                    y = bt_reward + gamma * bob.critic_target([bt_state_, bt_bob_act_, bt_alice_act_], training=True)
                    critic_value = bob.critic([bt_state, bt_bob_act, bt_alice_act], training=True)
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                critic_grad = tape.gradient(critic_loss, bob.critic.trainable_variables)
                bob.critic_optimizer.apply_gradients(zip(critic_grad, bob.critic.trainable_variables))

                alice.target_update()
                bob.target_update()
        reward_all.append(reward_epoch / step_length)
        print('epoch:%s, start:%s, reward:%s, sim:%s, fea:%s' % (epoch, start_flag, reward_epoch / step_length, sim_epoch / step_length, fea_epoch / step_length))
        if (epoch+1)%100==0 and start_flag:
            np.save('./R1/reward.npy', reward_all)
            file_path = './R1/'
            alice.save_model(file_path, epoch + 1)
            bob.save_model(file_path, epoch + 1)
            reward_epoch = 0
            sim_epoch = 0
            fea_epoch = 0
            alice_zero_count = np.zeros(key_length * 2)
            alice_one_count = np.zeros(key_length * 2)
            bob_zero_count = np.zeros(key_length * 2)
            bob_one_count = np.zeros(key_length * 2)
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
                alice_zero_count, alice_one_count = get_update(feature_vector_alice, alice_zero_count, alice_one_count, step)
                check_key, check_key_uti = get_check(alice_attribute, bob_attribute)
                similarity, feature_value, wrong_value, idle_value = get_similarity_feature_value(key_alice, key_bob, feature_vector_alice, feature_vector_bob, check_key_uti)
                reward = get_reward(similarity, feature_value, wrong_value)
                reward = reward

                sim_epoch += similarity
                fea_epoch += feature_value
                reward_epoch += reward
                # print('Alice Action:', act_alice)
                # print('Bob Action:', act_bob)
                print('Ali:', key_alice, feature_vector_alice)
                print('Bob:', key_bob, feature_vector_bob)
                print('Chk:', check_key, check_key_uti)
                print('reward:%s, sim:%s, feature:%s, wrong:%s' % (reward, similarity, feature_value, wrong_value))
                print('\n')
            print(reward_epoch/step_length)
            print(sim_epoch/step_length)
            print("\n")
            reward_one = np.array(reward_all)
            x = np.arange(len(reward_one))
            plt.plot(x, reward_one, label='Proposed FD2K')
            plt.xlabel('Epochs')
            plt.ylabel('Accumulated reward')
            plt.grid()
            plt.legend()
            plt.show()


if __name__ == '__main__':
    run()