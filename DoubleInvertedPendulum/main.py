import tensorflow as tf

import numpy as np
import random
import gym
import argparse
import time

from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam


import traceback
import datetime
import csv
import timeit
import json

from actor_network import actorNetwork
from critic_network import criticNetwork
from buffer import replayBuffer
from OU_process import OrnsteinUhlenbeckNoise


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)
    episode_loss = tf.Variable(0.)
    tf.summary.scalar("Loss", episode_loss)
    
    summary_vars = [episode_reward, episode_ave_max_q, episode_loss]
    summary_ops = tf.summary.merge_all()
    
    return summary_ops, summary_vars

def ddpg(train_indicator):
    
    BUFFER_SIZE=150000
    BATCH_SIZE=64
    GAMMA=0.99
    EXPLORE = 50000.
    TAU=0.001       #Target Network HyperParameters
    LRA=0.0001      #LEARNING RATE ACTOR
    LRC=0.001       #LEARNING RATE CRITIC


    MAX_EPISODES=50000
    MAX_STEPS=2000

    ENV_NAME = 'InvertedDoublePendulum-v1'

    reward = 0
    terminal = False
    epsilon = 1
    epsilon_decay = 0.995

    #-----------------#
    #       CSV       #
    #-----------------#

    # nome_file = 'reward.csv'
    # csv_file = open(nome_file, 'w')
    # data_writer = csv.writer(csv_file, delimiter=',')
    # data_writer.writerow(['episode','timestep','reward'])


    np.random.seed(1337)
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
       
    sess = tf.Session()
    from keras import backend as K
    K.set_session(sess)

    actor = actorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = criticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    memory = replayBuffer(BUFFER_SIZE)

    #Now load the weight
    print("Now we load the weight...")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actortargetmodel.h5")
        critic.target_model.load_weights("critictargetmodel.h5")
        print("Weight load successfully...")
    except:
        print("Cannot find the weight!")

    print("Double Inverted Pendulum __ Experiment Starting...")

    #--------------------#
    #     Tensorboard    #
    #--------------------#

    summary_ops, summary_vars = build_summaries()
    writer = tf.summary.FileWriter('./results', sess.graph)

    max_reward_ever = 0
    actor_noise = OrnsteinUhlenbeckNoise(action_dim)
    for i in range(MAX_EPISODES):
        if i%500==0:
            print("Training episode : " + str(i) + " Replay Buffer " + str(memory.count()))
    
        s = env.reset()
        ep_reward = 0.
        ep_ave_max_q = 0.

        actor_noise.reset()
        step=0
        for j in range(MAX_STEPS):
            loss=0
            start_time = time.time()
            #epsilon -= 1.0/EXPLORE
            epsilon *= epsilon_decay

            if train_indicator==1 and np.random.random() < epsilon:
            	a=actor.model.predict(s.reshape(1, s.shape[0])) + actor_noise.add_noise()
            else:
            	a=actor.model.predict(s.reshape(1, s.shape[0]))
            
            s_next, reward, terminal, info = env.step(a)


            # data_writer.writerow([i,j,reward])
            # csv_file.flush()
            
            memory.add(np.reshape(s, actor.state_dim),np.reshape(a, actor.action_dim),reward, terminal,np.reshape(s_next, actor.state_dim))
            
            #keep adding experiences to the memory until there are at least minibatch size samples
            if memory.count() > BATCH_SIZE:
                memory.sample_batch(BATCH_SIZE)
                #return memory.sample_batch(BATCH_SIZE)
                s_batch, a_batch, r_batch, t_batch, s2_batch = memory.sample_batch(BATCH_SIZE)
                #calculate targets
                a_batch = a_batch.reshape(BATCH_SIZE, action_dim)
                #a2_batch = actor.target_action(s2_batch)
                a2_batch = actor.target_model.predict(s2_batch)
                target_q = critic.target_model.predict([s2_batch, a2_batch])
                y_i=[]
                for k in range(BATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA*target_q[k])

                if (train_indicator):
                    loss += critic.model.train_on_batch([np.array(s_batch), np.array(a_batch)], np.array(y_i))
                    #loss += history.history['loss'][0]
                    a_for_grad = actor.model.predict(s_batch)
                    grads = critic.gradients(s_batch, a_for_grad)
                    predicted_q_value=critic.model.predict([s_batch, a_batch])
                    ep_ave_max_q += np.amax(predicted_q_value)
                    actor.train(s_batch, grads)
                    actor.target_train()
                    critic.target_train()
                else:
                    env.render()
                
            s = s_next
            ep_reward += reward
            end_time = time.time()-start_time


            step +=1
            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0] : ep_reward,
                    summary_vars[1] : ep_ave_max_q / float(j),
                    summary_vars[2] : loss
                })
                writer.add_summary(summary_str, i)
                writer.flush()
                break
                
        if ep_reward>max_reward_ever:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                actor.target_model.save_weights("actortargetmodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.target_model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

                critic.target_model.save_weights("critictargetmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.target_model.to_json(), outfile)
                print("Critic model saved")
                print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(ep_reward))
                print("Total steps in equilibrium: " + str(step))
                print("")
                max_reward_ever = ep_reward
    print("Finish.")
    sess.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')
    parser.add_argument('-t', help='train=1, test=0',  dest='train_indicator', type=int, default=1)
    # parser.add_argument('-m', help='name of the model',      dest='model_name', type=str,   default='')
    # parser.add_argument('-l', help='load',     dest='load', type=int,   default=0)
    # parser.add_argument('-lb', help='name file buffer to be loaded, if "" do not load ', dest='name_load_buffer', type=str,   default='')
    args = parser.parse_args()

    try:
        ddpg(train_indicator=args.train_indicator)
    except Exception as e:
        print("Exception occured!")
        traceback.print_stack()
        print("__________________")
        traceback.print_exc()