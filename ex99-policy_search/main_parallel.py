import numpy as np
from matplotlib import pyplot as plt
from functools import reduce
import operator
import gym
from sklearn.utils.extmath import softmax
from PSO import Particle_Swarm_Opt
from CMAES import CMAES
import pickle
import deap
import multiprocessing

import warnings
warnings.filterwarnings("ignore")

gym.logger.set_level(40)

# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers 
# in state vector. Reward for moving from the top of the screen to landing pad and 
# zero speed is about 100..140 points. If lander moves away from landing pad it loses 
# reward back. Episode finishes if the lander crashes or comes to rest, receiving 
# additional -100 or +100 points. Each leg ground contact is +10. Firing main engine 
# is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. 
# Fuel is infinite, so an agent can learn to fly and then land on its first attempt. 
# Four discrete actions available: do nothing, fire left orientation engine, fire 
# main engine, fire right orientation engine.

#state = [
#        (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
#        (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_W/SCALE/2),
#        vel.x*(VIEWPORT_W/SCALE/2)/FPS,
#        vel.y*(VIEWPORT_H/SCALE/2)/FPS,
#        self.lander.angle,
#        20.0*self.lander.angularVelocity/FPS,
#        1.0 if self.legs[0].ground_contact else 0.0,
#        1.0 if self.legs[1].ground_contact else 0.0
#        ]


#%% Reinforcement Learning
    
def discount_and_normalize_rewards(episode_rewards, gamma, norm=False):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    if norm:
        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    return discounted_episode_rewards

def parallel_eval_fitness_simulate(part):

    env = gym.make('LunarLander-v2')
    # Reinforcement Learning parameters
    Nx0=10
    Kmax = 1500
    Nmc  = 1
    gamma=1

    return eval_fitness_simulate(env, part, Nx0=Nx0, Kmax=Kmax, Nmc=Nmc, gamma=gamma, show=False)

def eval_fitness_simulate(env, part, Nx0=5, Kmax = 1000, Nmc  = 1, gamma=0.95, show=False):
    # Nx0  = number of episodes with different initial states to simulate each individual 
    # Kmax = max episode timestep
    # Nmc  = number of episodes for the same individual state for a stochastic policy (for deterministic Nmc=1)
    # gamma= discount factor
    
    # part = [i for i in range(36)]  
    Nx0_rewards = []
    for i_Nx0 in range(Nx0):
        # observation = env.reset()
        # save_point_env = np.copy.deepcopy(env)
        Nmc_rewards = []
        for i_Nmc in range(Nmc):
            # env = save_point_env
            observation = env.reset()
            t_rewards = []
            for t in range(Kmax):
                if show:
                    env.render()
                part_matrix_w = np.reshape(part[:-4],(4,8))
                part_matrix_b = part[-4:]
                prob_weights = softmax([part_matrix_w.dot(observation)+part_matrix_b]).ravel()
                action = np.random.choice(range(len(prob_weights)), p=prob_weights)
                observation, reward, done, info = env.step(action)
                t_rewards.append(reward)
                if done:
                    Nmc_rewards.append( discount_and_normalize_rewards( t_rewards, gamma, norm=False )[0] )
                    break
        Nx0_rewards.append( np.mean(Nmc_rewards) )
    return (np.mean(Nx0_rewards),)

def plot_logbook(logbook):
    plt.figure()
    plt.plot( np.arange(len(logbook))+1, [l['max'] for l in logbook] ,label='max')
    plt.plot( np.arange(len(logbook))+1, [l['min'] for l in logbook] ,label='min')
    plt.plot( np.arange(len(logbook))+1, [l['avg'] for l in logbook] ,label='avg')
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.show(block=True)

#def eval_fit(part):
#    Nx0=5
#    Kmax = 2000
#    Nmc  = 1
#    gamma=1
#    show=False
#    return eval_fitness_simulate(env, part, Nx0, Kmax, Nmc, gamma, show)

#%% MAIN
if __name__ == "__main__":
    
    # Start gym environment
    load = False
    save = False

    gamma = 1.0
    
    version = '18_06_13_PolicySearch_CMAES_noCNN_parallel_2'
    
    # Evolution Strategy algorithm parameters
    population= 50
    GEN=10
    loop_GEN = 20

    env = gym.make('LunarLander-v2')
    size= env.observation_space.shape[0] * env.action_space.n + env.action_space.n # 36

    disp=True
    
 
    
    if load:
        deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMax)
        with open(version+'.pkl', 'rb') as f:
            halloffame, logbook = pickle.load(f)
    else:
        eval_fit = parallel_eval_fitness_simulate
        cmaes = CMAES(eval_fit, lambda_=population, size=size)
        pool = multiprocessing.Pool(2)
        cmaes.toolbox.register("map", pool.map)
        halloffame = []
        for _ in range(loop_GEN):
            cmaes.optimize(GEN, disp)
            halloffame.append( {'nGen':GEN, 'gene':cmaes.hof.items[0], 'fitness_value':cmaes.hof.keys[0].values[0]} )
        logbook = cmaes.logbook
        if save:    
            with open(version+'.pkl', 'wb') as output:
                save_object = [halloffame, logbook]
                pickle.dump(save_object, output, pickle.HIGHEST_PROTOCOL)
    
    plot_logbook(logbook)

#    env.reset()
    env.render()
    import time
    time.sleep(20)
    
    eval_fitness_simulate(env, np.random.random(36), Nx0=1, Kmax=1000, Nmc=1, gamma=gamma, show=True)
    
    # Simulates the best individual
    for i_hof, hof in enumerate(halloffame):
        actual_nGen =  np.sum([h['nGen'] for h in halloffame[0:i_hof+1]])
        print('Generations: {0}   fitness_value: {1}'.format(actual_nGen, hof['fitness_value']) )
        eval_fitness_simulate(env, hof['gene'], Nx0=1, Kmax=1000, Nmc=1, gamma=gamma, show=True)
    env.close()
    
           
    #pso = Particle_Swarm_Opt(eval_fit, population=100, size=36, phi1=2.0, phi2=2.0, pmin=-6, pmax=6, smin=-3, smax=3)
    #for _ in range(20):
    #    pop, logbook, best = pso.optimize(GEN=25, disp=True)
    #pso.plot()
    #eval_fitness_simulate(env, best, Nx0=1, Kmax = 2000, Nmc = 1, gamma=1, show=True)
