import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def terminals(env):
    """ This is a helper function that returns terminal states 
    """
    # terms = []
    # for s in range(env.observation_space.n):
    #     # terminal is when we end with probability 1 in terminal:
    #     if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
    #         terms.append(s)
    # return terms
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


def sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay = 0.999, num_ep=int(1e4)):
    ''' Implementation of the SARSA algorithm (on-policy TD control) 
    '''
    n_states = env.observation_space.n
    n_actions =  env.action_space.n

    # Initialize arbitrarily Q(s; a)
    Q = np.random.random((n_states,  n_actions))
    # terminal states must be initialized to zero
    Q[terminals(env),:] = 0

    # This is some starting point performing random walks in the environment:
    for i in range(num_ep):
        if not i%100: print(f'iter:{i}') 

        # init state and first epsilon-greedy policy
        s = env.reset()
        a = np.argmax( Q[s,:] ) if np.random.binomial(1,1-epsilon) == 1 else np.random.randint(n_actions)
        
        done = False
        while not done:

            # take action a from behaviour policy, observe R, S'
            s_, r, done, _ = env.step(a)
            
            # behaviour policy == target policy == epsilon-greedy
            a_ = np.argmax( Q[s_,:] ) if np.random.binomial(1,1-epsilon) == 1 else np.random.randint(n_actions)
            
            # evaluate policy - update Q-value function
            # target policy: epsilon-greedy
            Q[s,a] += alpha * ( r + gamma * Q[s_,a_] - Q[s,a] )
            
            # update state and action
            s, a = s_, a_

        # decay epsilon value (exploration->exploitation)
        epsilon *= epsilon_decay
    return Q


def qlearning(env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay = 0.999, num_ep=int(1e4)):
    ''' Implementation of the Q-learning algorithm (off-policy TD control) 
    '''
    n_states = env.observation_space.n
    n_actions =  env.action_space.n

    # Initialize arbitrarily Q(s; a)
    Q = np.random.random((n_states,  n_actions))
    # terminal states must be initialized to zero
    Q[terminals(env),:] = 0

    # This is some starting point performing random walks in the environment:
    for i in range(num_ep):
        if not i%100: print(f'iter:{i}') 

        # init state and first epsilon-greedy policy
        s = env.reset()
        
        done = False
        while not done:
            
            # behaviour policy: epsilon-greedy
            a = np.argmax( Q[s,:] ) if np.random.binomial(1,epsilon) == 0 else np.random.randint(n_actions)

            # take action a from behaviour policy, observe R, S'
            s_, r, done, _ = env.step(a)

            # evaluate policy - update Q-value function
            # target policy: greedy
            Q[s,a] += alpha * ( r + gamma * np.max(Q[s_,:]) - Q[s,a] )
            
            # update state and action
            s = s_
        
        # decay epsilon value (exploration->exploitation)
        epsilon *= epsilon_decay
    return Q



is_slippery=True
map_name = ["4x4","8x8"][1]
alpha=0.1
gamma=0.9
epsilon=0.9            # lots of exploration in the beginning
epsilon_decay = 0.999  # relativelly high decay to exploit soon    # decay**k * epsilon = fraction * epsilon   ->  k = np.log(f)/np.log(d)
num_ep=int(1e4)


# env=gym.make('FrozenLake-v0')
env=gym.make('FrozenLake-v0', is_slippery=is_slippery,  map_name=map_name)
#env=gym.make('FrozenLake-v0', map_name="8x8")
env.render()

print("Running sarsa...")
Q = sarsa(env, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, num_ep=num_ep)
plot_V(Q, env)
plot_Q(Q, env)
print_policy(Q, env)
plt.show()

print("Running qlearning")
Q = qlearning(env, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, num_ep=num_ep)
plot_V(Q, env)
plot_Q(Q, env)
print_policy(Q, env)
plt.show()
