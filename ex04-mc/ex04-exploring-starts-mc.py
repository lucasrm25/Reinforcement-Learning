import gym
import numpy as np
import matplotlib.pyplot as plt

''' Use Monte Carlo ES (exploring starts) to learn action-state-value function Q(s,a) and at the same time improve policy

Alternate between policy evaluation and policy improvement:
- MC policy iteration step: policy evaluation using MC methods
- Policy improvement step: greed w.r.t. to action{value

'''

def main():

    env = gym.make('Blackjack-v0')
    
    # Init some useful variables:
    n_states = tuple([s.n for s in env.observation_space])   # [players current sum, dealer's one showing card, player holds a usable ace]
    n_actions = tuple([env.action_space.n])

    maxiter = int(500*1e3)

    # Initialize random suboptimal policy
    policy = np.random.randint(0,n_actions,n_states)
    
    # Initialize arbitrary state-action-value function Q(s,a).
    Q = np.zeros( n_states + n_actions )
    # number of samples that has been drawn for each (s,a) pair to estimate Q(s,a)
    n = np.ones( n_states + n_actions )
    
    # Returns(s,a) <- empty list
    # returns = np.frompyfunc(list, 0, 1)(np.empty(n_states + n_actions, dtype=object))

    for i in range(maxiter):
        if not i%1000: print(f'it:{i}')

        # sample some random initial state and action
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        s = tuple([int(si) for si in obs])
        a = tuple(np.random.randint(0,n_actions,1))

        # dictionary state_action:cummulative_reward to store cummulative rewards for each episode
        G = {}

        # Generate episode starting from (s0, a0) following π
        done = False
        while not done:
            
            # evaluate policy and play
            obs, reward, done, _ = env.step(a[0])

            # consider only the first state, in case the same state appears in the same trajectory
            if not s+a in G:
                G[s+a] = 0

            # accumulates the future reward for all states in past state
            for s_a_idx in G.keys():
                G[s_a_idx] += reward
            
            # evaluate policy: improve state-action value function Q(s,a)
            for s_a_idx, G_s_a in G.items():
                
                # NOTE: it takes too much memory to store all MC samples
                # returns[s_a_idx].append(G_s_a)
                # Q[s_a_idx] = np.average(returns[s_a_idx])

                Q[s_a_idx] = Q[s_a_idx] + 1/n[s_a_idx] * ( G_s_a - Q[s_a_idx] )
                n[s_a_idx] += 1

            # improve policy
            policy[s] = np.argmax( Q[s][:] )

            # decode next state-action
            s = tuple([int(si) for si in obs])
            a = (policy[s],)

    fig = plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.imshow( policy[12:22,1:,0], vmin=0, vmax=1, extent=[1,11,21,12] )
    plt.colorbar()
    plt.xlabel('Dealer first card')
    plt.ylabel('Player sum')
    plt.title('Optimal Policy \n(NO usable ace)')

    plt.subplot(122)
    plt.imshow( policy[12:22,1:,1], vmin=0, vmax=1, extent=[1,11,21,12] )
    plt.colorbar()
    plt.xlabel('Dealer first card')
    plt.ylabel('Player sum')
    plt.title('Optimal Policy \n(usable ace)')
    plt.show()


if __name__ == "__main__":
    main()
