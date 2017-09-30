import numpy as np
import math
import keras
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Lambda
from keras.layers.merge import Add
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS=72
HIDDEN2_UNITS=48

'''
ACTOR NETWORK - overview

actorNetwork=actorNetwork(sess, state_dim, action_dim, action_bound)

actorNetwork.create_training_method()
actorNetwork.create_network()
actorNetwork.create_target_network()
actorNetwork.update_target()
actorNetwork.train()
actorNetwork.action()
actorNetwork.target_action()

#parameters:

actorNetwork.sess
actorNetwork.state_dim
actorNetwork.action_dim
actorNetwork.action_bound

actorNetwork.state_input
actorNetwork.action_output
actorNetwork.net

actorNetwork.target_state_input
actorNetwork.target_action_output
actorNetwork.target_update
actorNetwork.target_net

'''

class actorNetwork(object):
    
    def __init__(self, sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE): #action_bound):
        self.sess=sess
        self.BATCH_SIZE=BATCH_SIZE
        self.TAU=TAU
        self.LEARNING_RATE=LEARNING_RATE
        self.action_dim = action_dim
        self.state_dim = state_dim

        K.set_session(sess)
        K.set_learning_phase(1)

        #Now create the model
        self.model, self.weights, self.state = self.create_network(state_dim, action_dim)
        print("Actor network created")
        self.target_model, self.target_weights, self.target_state = self.create_network(state_dim, action_dim)
        print("Target Actor network created")
        self.action_gradient = tf.placeholder(tf.float32,[None, action_dim])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)
        
    def create_network(self, state_dim, action_dim):
        print("Now we build the model")
        S=Input(shape=[state_dim])
        #norm_input=BatchNormalization()(S)
        h0=Dense(HIDDEN1_UNITS, activation='relu', kernel_initializer='glorot_uniform')(S)
        #norm_1_layer=BatchNormalization()(h0)
        h1=Dense(HIDDEN2_UNITS, activation='relu', kernel_initializer='glorot_uniform')(h0)
        #norm_2_layer=BatchNormalization()(h1)
        cart=Dense(action_dim,activation='tanh',kernel_initializer=keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None))(h1)
        model=Model(inputs=S, outputs=cart)
        model.compile(loss="mse", optimizer='sgd')


        return model, model.trainable_weights, S
    
    