import numpy as np
import math
import keras
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Lambda, Activation
from keras.models import Sequential, Model
from keras.layers.merge import Add
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 72
HIDDEN2_UNITS = 48

'''
criticNetwork = criticNetwork(sess, state_dim, action_dim)

criticNetwork.create_training_method()
criticNetwork.create_network()
criticNetwork.create_target_network()
criticNetwork.update_target()
criticNetwork.train()
criticNetwork.q_value()
criticNetwork.target_q_value()
criticNetwork.gradients()

#parameters:

criticNetwork.sess
criticNetwork.state_dim
criticNetwork.action_dim

criticNetwork.state_input
criticNetwork.action_input
criticNetwork.q_value_output
criticNetwork.net

criticNetwork.target_state_input
criticNetwork.target_action_input
criticNetwork.target_q_value_output
criticNetwork.target_update
criticNetwork.target_net
'''



class criticNetwork(object):
    def __init__(self, sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_dim = action_dim
        self.state_dim = state_dim

        K.set_session(sess)
        K.set_learning_phase(1)
        
        self.model, self.action, self.state = self.create_network(state_dim, action_dim)
        print("Critic network created")  
        self.target_model, self.target_action, self.target_state = self.create_network(state_dim, action_dim)  
        print("Target Critic network created")
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]    
    
    def create_network(self, state_dim, action_dim):
        print("Now we build the model")
        S = Input(shape=[state_dim])
        #S_n = BatchNormalization()(S)
        A = Input(shape=[action_dim],name='action2')
        #A_n = BatchNormalization()(A)
        w1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_initializer='glorot_uniform')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer='glorot_uniform')(w1)
        #norm_1_layer = BatchNormalization()(w1)
        a1 = Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer='glorot_uniform')(A)
        #norm_a_layer = BatchNormalization()(a1) 
        #norm_2_layer = BatchNormalization()(h1)
        h2 = Add()([h1,a1]) 
        h3 = Dense(HIDDEN1_UNITS, activation='relu', kernel_initializer='glorot_uniform')(h2)
        #norm_3_layer = BatchNormalization()(h3)
        V = Dense(1,activation='relu', kernel_initializer=keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None))(h3)   
        model = Model(inputs=[S,A],outputs=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)
