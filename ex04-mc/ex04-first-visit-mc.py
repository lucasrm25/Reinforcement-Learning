import gym
import numpy as np
import matplotlib.pyplot as plt

''' First-visit Monte Carlo sampling of value function, under some policy pi
'''

def main():

    env = gym.make('Blackjack-v0')
    
    # Init some useful variables:
    n_states = [s.n for s in env.observation_space]   # [players current sum, dealer's one showing card, player holds a usable ace]
    n_actions = env.action_space.n

    maxiter = int(100*1e3)

    # Initialize given policy to be evaluated: stick if sum >= 20, else hit
    policy = np.zeros(n_states, dtype=int)
    policy[:21,:,:] = 1
    policy[21:,:,:] = 0
    
    # Initialize arbitrary state-value function
    V = np.zeros(n_states)
    
    # empty list for all s \in S
    returns = np.frompyfunc(list, 0, 1)(np.empty(n_states, dtype=object))


    for i in range(maxiter):

        if not i%1000:
            print(f'it:{i}')

        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        s = tuple([int(si) for si in obs])
        done = False

        # dictionary state:cummulative_reward to store cummulative rewards for each episode
        G = {}

        # Generate episode using given policy
        while not done:
            
            # evaluate policy and play
            obs, reward, done, _ = env.step( policy[s] )
            # decode next state
            sn = tuple([int(si) for si in obs])

            # consider only the first state, in case the same state appears in the same trajectory
            if not s in G:
                G[s] = 0

            # accumulates the future reward for all states in past state
            for si in G.keys():
                G[si] += reward
            
            s = sn

        # for all state in episode:
        #   1. append cummulative reward to Returns(s)
        #   2. Update value function V(s) = average(Returns(s))
        for si,G_si in G.items():
            returns[si].append(G_si)
            V[si] = np.average(returns[si])


    fig = plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.imshow( V[12:22,1:,0], vmin=-1, vmax=1, extent=[1,11,21,12] )
    plt.colorbar()
    plt.xlabel('Dealer first card')
    plt.ylabel('Player sum')
    plt.title('Value - Expected cummulative reward \n(NO usable ace)')

    plt.subplot(122)
    plt.imshow( V[12:22,1:,1], vmin=-1, vmax=1, extent=[1,11,21,12] )
    plt.colorbar()
    plt.xlabel('Dealer first card')
    plt.ylabel('Player sum')
    plt.title('Value - Expected cummulative reward \n(usable ace)')
    plt.show()


if __name__ == "__main__":
    main()
