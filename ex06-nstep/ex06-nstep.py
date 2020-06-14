''' Implementation of n-step SARS Algorithm to find optimal policy for the
FrozenLake environment
'''

import multiprocessing as mp
from itertools import chain, product

import gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import dill


def terminals(env):
    """ This is a helper function that returns terminal states 
    """
    return ((env.desc == b'H') | (env.desc == b'G') ).flatten()


def print_policy(Q, env):
    """ This is a helper function to print a nice policy from the Q function"""
    moves = [u'←', u'↓',u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    policy = np.chararray(dims, unicode=True)
    policy[:] = ' '
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        policy[idx] = moves[np.argmax(Q[s])]
        if env.desc[idx] in [b'H', b'G']:
            policy[idx] = u'·'
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row]) 
        for row in policy]))


def plot_V(Q, env):
    """ This is a helper function to plot the state values from the Q function"""
    fig = plt.figure()
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    V = np.zeros(dims)
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        V[idx] = np.max(Q[s])
        if env.desc[idx] in [b'H', b'G']:
            V[idx] = 0.
    plt.imshow(V, origin='upper', 
               extent=[0,dims[0],0,dims[1]], vmin=.0, vmax=.6, 
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y+0.5, dims[0]-x-0.5, '{:.3f}'.format(V[x,y]),
                horizontalalignment='center', 
                verticalalignment='center')
    plt.xticks([])
    plt.yticks([])


def plot_Q(Q, env):
    """ This is a helper function to plot the Q function """
    from matplotlib import colors, patches
    fig = plt.figure()
    ax = fig.gca()

    # plt.ion()
    # plt.show()

    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape

    up = np.array([[0, 1], [0.5, 0.5], [1,1]])
    down = np.array([[0, 0], [0.5, 0.5], [1,0]])
    left = np.array([[0, 0], [0.5, 0.5], [0,1]])
    right = np.array([[1, 0], [0.5, 0.5], [1,1]])
    tri = [left, down, right, up]
    pos = [[0.2, 0.5], [0.5, 0.2], [0.8, 0.5], [0.5, 0.8]]

    cmap = plt.cm.RdYlGn
    norm = colors.Normalize(vmin=.0,vmax=.6)

    ax.imshow(np.zeros(dims), origin='upper', extent=[0,dims[0],0,dims[1]], vmin=.0, vmax=.6, cmap=cmap)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        x, y = idx
        if env.desc[idx] in [b'H', b'G']:
            color = 'k' if env.desc[idx] == b'H' else 'darkgreen'
            ax.add_patch(patches.Rectangle((y, dims[0]-1-x), 1, 1, color=color)) #cmap(.0)
            plt.text(y+0.5, dims[0]-x-0.5, '{:.2f}'.format(.0),
                horizontalalignment='center', 
                verticalalignment='center')
            continue
        for a in range(len(tri)):
            ax.add_patch(patches.Polygon(tri[a] + np.array([y, dims[0]-1-x]), color=cmap(Q[s][a]) ))
            plt.text(y+pos[a][0], dims[0]-1-x+pos[a][1], '{:.2f}'.format(Q[s][a]), 
                        horizontalalignment='center', verticalalignment='center',
                    fontsize=9, fontweight=('bold' if Q[s][a] == np.max(Q[s]) else 'normal'))

    plt.xticks([])
    plt.yticks([])


def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay = 0.999, num_ep=int(1e4), verbose=False):
    ''' Implementation of the n-step SARSA algorithm (on-policy TD control) 
    '''
    n_states = env.observation_space.n
    n_actions =  env.action_space.n

    # Initialize arbitrarily Q(s; a)
    Q = np.random.random((n_states,  n_actions))
    # terminal states must be initialized to zero
    Q[terminals(env),:] = 0

    # for all episodes
    for i in range(int(num_ep)):
        if not i%100 and verbose: print(f'iter:{i}') 

        stored_actions = {}
        stored_rewards = {}
        stored_states  = {}

        # init state S0 != terminal
        s = env.reset()
        # choose action from behaviour policy: epsilon-greedy
        a = np.argmax( Q[s,:] ) if np.random.binomial(1,1-epsilon) == 1 else np.random.randint(n_actions)
        
        # store
        stored_states[0] = s
        stored_actions[0] = a

        T = np.inf  # maximum number of time steps
        tau = 0     # tau is the time whose state’s estimate is updated
        t = 0       # current episode time step
        while tau < T-1:

            # evaluate if state is not terminal or maximum number of time steps have been reached
            if t < T:
                
                # take action a from behaviour policy, observe Rt+1, St+1
                s_, r_, done, _ = env.step(int(a))

                # block the algorithm from entering to this section if state is terminal => t > T
                if done: 
                    T = t + 1
                else:
                    # choose action from behaviour policy: epsilon-greedy
                    a_ = np.argmax( Q[s_,:] ) if np.random.binomial(1,1-epsilon) == 1 else np.random.randint(n_actions)

                # store action a_t, reward R_t+1, state S_t+1
                stored_rewards[t+1] = r_
                stored_states[t+1]  = s_
                stored_actions[t+1] = a_

            tau = t - n + 1     # tau is the time whose state’s estimate is updated

            # we have recorded at least n states or we reached the end of the episode
            # we are now able to evaluate and update the policy
            if tau >= 0:

                # G_{t:t+n}
                G = np.sum([
                    gamma**(i-tau-1) * stored_rewards[i]
                    for i in range(tau+1, min(tau+n,T)+1)
                ])
                # terminal state implies that Q(s'=terminal,a')=0
                G += gamma**n * Q[stored_states[tau+n], stored_actions[tau+n]] if tau + n < T else 0
        
                # evaluate policy - update Q-value function
                s_tau = stored_states[tau]
                a_tau = stored_actions[tau]
                Q[s_tau,a_tau] += alpha * ( G - Q[s_tau,a_tau] )

            # update state and action
            s, a = s_, a_
            # decay epsilon value (exploration->exploitation)
            epsilon *= epsilon_decay
            # increase current time step
            t += 1
    return Q

def V_iteration(env, maxiter=1e3, gamma = 0.8, eps = 1e-8):
    ''' Calculate optimal policy using the Value iteration algorithm

        NOTE: env.P[state][action] gives tuples (p, ns, r, is_terminal), which tells the
        probability p that we end up in the next state ns and receive reward r -> p(s',r|s,a)
    '''
    n_states = env.observation_space.n
    n_actions =  env.action_space.n

    V = np.zeros(n_states)  # init values as zero

    # iterate value function until convergence
    for _ in range(int(maxiter)):

        # value function for the next iteration
        V_ = np.array([
            np.max( [
                np.sum( [ p*(r+gamma*V[s_]) for p, s_, r, _ in env.P[s][a] ] )  # Bellman equation
                for a in range(n_actions)
            ])
            for s in range(n_states)
        ])

        if np.max(V_-V) <= eps: 
            break
        else: 
            V = V_

    # evaluate best policy (Bellman equation)
    optPolicy = np.array([
            np.argmax( [
                    np.sum( [ p*(r+gamma*V[s_]) for p, s_, r, _ in env.P[s][a] ] )
                    for a in range(n_actions)
            ])
            for s in range(n_states)
        ])
    
    return V, optPolicy

def Q_iteration(env, maxiter=1e3, gamma = 0.8, eps = 1e-8):
    ''' Calculate optimal policy using the Q iteration algorithm

        NOTE: env.P[state][action] gives tuples (p, ns, r, is_terminal), which tells the
        probability p that we end up in the next state ns and receive reward r -> p(s',r|s,a)
    '''
    n_states = env.observation_space.n
    n_actions =  env.action_space.n

    Q = np.zeros((n_states,n_actions))  # init values as zero

    # iterate value function until convergence
    for _ in range(int(maxiter)):

        # value function for the next iteration
        Q_ = np.array([
            [
                np.sum( [ p*(r+gamma*np.max(Q[s_,:])) for p, s_, r, _ in env.P[s][a] ] )  # Bellman equation
                for a in range(n_actions)
            ]
            for s in range(n_states)
        ])

        if np.max(Q_-Q) <= eps: 
            break
        else: 
            Q = Q_
    return Q


# V = {(1,2):1, (1,4):10}
# c = 1
# print( [V[c,i] for i in [2,4]] )


if __name__ == '__main__':

    map_name = ["4x4","8x8"][1]
    is_slippery=False
    alpha=0.1
    gamma=0.9

    # epsilon=0.9            # lots of exploration in the beginning
    # epsilon_decay = 0.999  # relativelly high decay to exploit soon    # decay**k * epsilon = fraction * epsilon   ->  k = np.log(f)/np.log(d)
    epsilon=0.1    
    epsilon_decay = 1

    num_ep=1e1

    env=gym.make('FrozenLake-v0', is_slippery=is_slippery,  map_name=map_name)

    # True Q and V functions evaluated using dynamic programming
    Q_true    = Q_iteration(env, maxiter=1e4, gamma=gamma, eps=1e-15)
    V_true, _ = V_iteration(env, maxiter=1e4, gamma=gamma, eps=1e-15)
    assert not np.any( np.max(Q_true,axis=1) - V_true )
    plot_Q(Q_true,env)
    plt.show()


    # define hyper-parameters to evaluate
    n = [1,2,4,8,16]
    alpha = [0.1,0.2,0.4,0.6,0.8,1.0]
    params = list(product(
        [env],n,alpha,[gamma],[epsilon],[epsilon_decay],[num_ep]
    ))

    # # test algorithm
    # Q = nstep_sarsa(env, n=3, alpha=0.2, num_ep=1e3, verbose=True)
    # print_policy(Q,env)
    # plot_Q(Q,env)
    # plt.show()
    
    # wrapper to allow using tqdm with multiprocessing
    def nstep_sarsa_wrapper(arglist):
        return nstep_sarsa(*arglist)
    # evaluate
    with mp.Pool(processes= int(mp.cpu_count()) ) as pool:
        Qmp = list(tqdm(pool.imap(nstep_sarsa_wrapper, params), total=len(params)))

    # convert list to dict
    Q = {}
    for i, (ni, alphai) in enumerate( np.vstack(params)[:,1:3] ):
        Q[ni,alphai] = Qmp[i]

    Qrmse = {}
    for k, Qi in Q.items():
        ni, alphai = k
        Qrmse[k] = np.sqrt(mean_squared_error( Qi.flatten() , Q_true.flatten() ))

    # with open('./rldata.dat', "wb") as f:
    #     dill.dump({'Q':Q,'Qrmse':Qrmse},f)

    # with open('./rldata.dat', "rb") as f:
    #     data = dill.load(f)
    # Q = data['Q']
    # Qrmse = data['Qrmse']

    plt.figure()
    rmse_n = np.zeros((len(n),len(alpha)))
    for i, ni in enumerate(n):
        for j, ai in enumerate(alpha):
            rmse_n[i,j] = Qrmse[ni,ai]
        plt.plot( alpha, rmse_n[i], '-*', label=f'n={ni}' )
    plt.legend()
    plt.show()
