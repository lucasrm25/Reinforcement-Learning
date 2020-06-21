import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time

env = gym.make('MountainCar-v0')

# define grid size to discretize state space
ngridpoints = 20
dim = env.observation_space.shape[0]

# discretize state space
state_grid = np.linspace( env.observation_space.low, env.observation_space.high, ngridpoints ).T
states = np.array(list(product(*state_grid)))

env.observation_space.n = ngridpoints**dim

def getStateIdx(s):
    return np.argmin(np.linalg.norm(s-states,axis=1))

def qlearning_lambda(env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=1.0, lamb=0.95, num_ep=int(1e4), ep_length=200):
    ''' Implementation of the Q-learning algorithm (off-policy TD control) with Eligibility traces.
    The continuous state space is discretized using state aggregation.
    '''
    render_every_X_ep = 100
    print_every_X_ep = 100

    n_states  = env.observation_space.n
    n_actions = env.action_space.n

    # init arbitrarily Q(s; a)
    Q = np.random.random((n_states,n_actions)) #np.zeros((n_states,n_actions))
    # init eligibility traces
    eligibility = np.zeros((n_states,n_actions))

    nbr_succ = []
    nbr_step = []

    for i in range(num_ep):
        # init state
        s_cont = env.reset()
        s = getStateIdx(s_cont)

        r_total = 0
        for j in range(ep_length):

            if not i % render_every_X_ep and i > 0: 
                env.render()
            
            # choose action from behaviour policy: epsilon-greedy
            a = np.argmax( Q[s,:] ) if not np.random.binomial(1,epsilon) else np.random.randint(n_actions)

            # take action a from behaviour policy, observe R, S'
            s_cont_, r, done, info = env.step(a)
            s_ = getStateIdx(s_cont_)

            # target policy: greedy
            a_ = np.argmax(Q[s_,:])
            # TD error  
            td_error = r + gamma * Q[s_,a_] - Q[s,a]
            # decay eligibility of all states and increase the recently visited one
            eligibility *= lamb * gamma
            eligibility[s,a] += 1

            # update every state's Q-value estimate according to their eligibilities
            # here we are sharing with all the visited states which value function they will see in the future
            Q += alpha * td_error * eligibility
            
            # update state and action
            s = s_

            r_total += r
            if done: 
                break

        nbr_succ.append(s_cont_[0] >= 0.5)
        if len(nbr_succ)>200: nbr_succ.pop(0)
        nbr_step.append(j+1)      
        if len(nbr_step)>200: nbr_step.pop(0)

        # decay epsilon value (exploration->exploitation)
        epsilon *= epsilon_decay

        if not i % print_every_X_ep and i > 0: 
            print(f'episode: {i:06d} - epsilon: {epsilon} - success rate: {np.mean(nbr_succ)} - avg steps/ep {np.mean(nbr_step)}')

    return Q


def main():
    env.reset()
    Q = qlearning_lambda(env, alpha=0.1, gamma=0.9, epsilon=0.9, epsilon_decay=1-1e-3, lamb=0.95, num_ep=int(5e3), ep_length=200)
    env.close()
    plot_V(env, Q)
    plt.show()

if __name__ == "__main__":
    main()
