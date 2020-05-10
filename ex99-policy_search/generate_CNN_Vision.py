import numpy as np
import random
import multiprocessing as mp
import gym
from gym.envs.box2d import LunarLander
import os
from sklearn.utils.extmath import softmax

import time
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 13.0
SIDE_ENGINE_POWER  =  0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY   = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

_SAVE_FOLDER = './data_CNN_Vision'
_BATCH_SIZE = 600
_NUM_BATCHES = 60
_TIME_STEPS = 2000
_RENDER = True

def random_initial_position(env):
    
    env.reset()

    pos_x = VIEWPORT_H/SCALE * random.random()
    lander_angle = np.pi/2 * random.random() - np.pi/4
    print (pos_x)
    env.lander = env.world.CreateDynamicBody(
        position = (pos_x, VIEWPORT_H/SCALE),
        angle = lander_angle,
        fixtures = fixtureDef(
            shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in LANDER_POLY ]),
            density=5.0,
            friction=0.1,
            categoryBits=0x0010,
            maskBits=0x001,  # collide only with ground
            restitution=0.0) # 0.99 bouncy
        )
    env.lander.color1 = (0.5,0.4,0.9)
    env.lander.color2 = (0.3,0.3,0.5)
    env.lander.ApplyForceToCenter( (
        env.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
        env.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        ), True)

    env.legs = []
    for i in [-1,+1]:
        leg = env.world.CreateDynamicBody(
            position = (pos_x - i*LEG_AWAY/SCALE, VIEWPORT_H/SCALE),
            angle = lander_angle + (i*0.05),
            fixtures = fixtureDef(
                shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                density=1.0,
                restitution=0.0,
                categoryBits=0x0020,
                maskBits=0x001)
            )
        leg.ground_contact = False
        leg.color1 = (0.5,0.4,0.9)
        leg.color2 = (0.3,0.3,0.5)
        rjd = revoluteJointDef(
            bodyA=env.lander,
            bodyB=leg,
            localAnchorA=(0, 0),
            localAnchorB=(i*LEG_AWAY/SCALE, LEG_DOWN/SCALE),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=LEG_SPRING_TORQUE,
            motorSpeed=+0.3*i  # low enough not to jump back into the sky
            )
        if i==-1:
            rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
            rjd.upperAngle = +0.9
        else:
            rjd.lowerAngle = -0.9
            rjd.upperAngle = -0.9 + 0.5
        leg.joint = env.world.CreateJoint(rjd)
        env.legs.append(leg)
        
    env.drawlist = [env.lander] + env.legs


def generate_action(env, prev_action, observation, w, b):
#    index = np.random.randn(env.action_space.n)    
#    index[2] = np.abs(index[2])     # Favor main engine over the others:
#    return np.argmax(index)
    prob_weights = softmax([w.dot(observation)+b]).ravel()
    action = np.random.choice(range(len(prob_weights)), p=prob_weights)
    return action

def normalize_observation(observation_img):
    return observation_img.astype('float32') / 255.

def simulate_batch(batch_num):
    import cv2
    
    env = LunarLander()

    obs_data = []
    action = env.action_space.sample()
    
#    for i_episode in range(_BATCH_SIZE):
    while len(obs_data) < _BATCH_SIZE:
        random_initial_position (env) #        observation = env.reset()

        mw = np.random.random_sample((4,8))
        mb = np.random.random_sample(4)
        
        for t in range(_TIME_STEPS):

            observation_img = env.render(mode='rgb_array')
            observation, reward, done, info = env.step(action)
            action = generate_action(env, action, observation, mw, mb)
            
#            if not t % 3:
                # Checks if the shuttle is inside the screen
            if -0.90 <= observation[0] <= 0.90 and -0.90 <= observation[1] <= 0.90:
                if np.random.random() > 0.7:
                    res_img = cv2.resize(observation_img, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
                    obs_data.append([normalize_observation(res_img), observation])
#                if np.abs(observation[1]-400/30/2) <= np.abs(np.random.normal(0, 400/30/2/3)) or np.abs(observation[0]-600/30/2) <= np.abs(np.random.normal(0, 600/30/2/3)):
#                    obs_data.append([normalize_observation(observation_img), observation])
            else:
                print ('Observation ignored')
                
            print (env.lander.position, observation[0:2])
            if done:
                break
            

    print("Saving dataset for batch {}".format(batch_num))
    if not os.path.isdir(_SAVE_FOLDER):
        os.mkdir(_SAVE_FOLDER)
        
    np.save(os.path.join(_SAVE_FOLDER,'obs_data_VAE_{}'.format(batch_num)), obs_data)    
    env.close()

def main():
    print("Generating data for Auto-Enconder")

    with mp.Pool( 8 ) as p:
        p.map(simulate_batch, range(_NUM_BATCHES))

if __name__ == "__main__":
    main()# -*- coding: utf-8 -*-
