import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

#multi-agent reinforcement learning algorithm that uses the Actor-Critic method with the Deep Deterministic Policy Gradient (DDPG) algorithm.
#code is written in Python using the TensorFlow library.

class Buffer:
    #The Buffer class is responsible for storing the experiences of the agents in a replay buffer,
    # which is used for training the neural networks.
    def __init__(self, buffer_capacity, num_agent, num_obs, num_act):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((buffer_capacity, num_agent, num_obs))
        self.action_buffer = np.zeros((buffer_capacity, num_agent, (num_agent-1)*num_act))
        self.reward_buffer = np.zeros((buffer_capacity, 1))
        self.next_state_buffer = np.zeros((buffer_capacity, num_agent, num_obs))

    # Takes (s,a,r,s') obervation tuple as input
    def store(self, state, action, reward, state_):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = state_
        self.buffer_counter += 1


class Agent:
    #he Agent class contains the neural networks for the actor and critic and defines the training procedure for the networks.
    def __init__(self, name, num_obs, num_act, num_agent, action_bound):
        self.name=name
        self.num_obs=num_obs
        self.num_act=num_act
        self.num_agent=num_agent
        self.action_bound = action_bound
        self.tau = 0.01

        def get_actor():
            #The get_actor() function creates a neural network for the actor,
            # which outputs the actions for the agent given its observations
            # Initialize weights between -3e-3 and 3-e3
            last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

            inputs = layers.Input(shape=(num_obs,))
            out = layers.Dense(100, activation="elu")(inputs)
            #out = layers.Dropout(0.2)(out)
            out = layers.Dense(100, activation="elu")(out)
            out = layers.Dense(100, activation="elu")(out)
            out = layers.Dense(100, activation="elu")(out)
            out = layers.Dense(100, activation="elu")(out)
            #out = layers.Dropout(0.2)(out)
            outputs = layers.Dense(num_act, activation="tanh", kernel_initializer=last_init)(out)
            # Our upper bound is 2.0 for Pendulum.

            outputs = outputs
            model = tf.keras.Model(inputs, outputs)
            return model

        def get_critic():
            #The get_critic() function creates a neural network for the critic,
            # which evaluates the quality of the actions taken by the actor given the observations.
            # State as input
            last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            state_input = layers.Input(shape=(num_agent, num_obs))
            state_output = layers.Dense(100, activation="elu")(state_input)
            #state_output = layers.Dense(100, activation="elu")(state_output)
            state_output = layers.Flatten()(state_output)
            # state_output = state_output[:, :, 0]

            # Action as input
            self_act_input = layers.Input(shape=(num_act * (num_agent - 1)))
            other_act_input = layers.Input(shape=(num_act * (num_agent - 1) * (num_agent - 1)))
            action_input = layers.Concatenate()([self_act_input, other_act_input])
            action_out = layers.Dense(100, activation="elu")(action_input)
            action_out = layers.Flatten()(action_out)
            # Both are passed through seperate layer before concatenating
            concat = layers.Concatenate()([state_output, action_out])


            action_out = layers.Dense(400, activation="elu")(concat)
            action_out = layers.Dense(300, activation="elu")(action_out)
            action_out = layers.Dense(100, activation="elu", kernel_initializer=last_init)(action_out)
            #action_out = layers.Dense(100, activation="elu")(action_out)
            outputs = layers.Dense(1)(action_out)
            # Outputs single value for give state-action
            model = tf.keras.Model([state_input, self_act_input, other_act_input], outputs)
            return model

        self.actor = get_actor()
        self.critic = get_critic()
        self.actor_target = get_actor()
        self.critic_target = get_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(1e-4, clipnorm=10)
        self.critic_optimizer = tf.keras.optimizers.Adam(1e-3, clipnorm=10)
        # self.actor_optimizer = tf.keras.optimizers.SGD(1e-4)
        # self.critic_optimizer = tf.keras.optimizers.SGD(1e-3)
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())


    def save_model(self, fila_path, epoch):
        # saves the actor neural network to a file.
        self.actor.save(filepath=fila_path+ self.name+'_actor'+'_'+str(epoch))



    def target_update(self):
        #which updates the target networks for the actor and critic.
        # This is done by soft updating the weights of the target networks with the weights of the main networks.
        for (a, b) in zip(self.actor_target.variables, self.actor.variables):
            a.assign(b * self.tau + a * (1 - self.tau))
        for (a, b) in zip(self.critic_target.variables, self.critic.variables):
            a.assign(b * self.tau + a * (1 - self.tau))



# The AgentRestore class is used to load a saved actor neural network from a file.
class AgentRestore:
    def __init__(self, name, num_obs, num_act, num_agent, epoch):
        self.name=name
        self.num_obs = num_obs
        self.num_act = num_act
        self.num_agent = num_agent
        self.tau = 0.01
        self.epoch = epoch

        def restore_actor():
            model = keras.models.load_model(self.name+'_actor_'+str(epoch))
            return model

        self.actor = restore_actor()
















