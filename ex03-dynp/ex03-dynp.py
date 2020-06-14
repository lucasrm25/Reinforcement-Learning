import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)



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

def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    V, policy = V_iteration(env)
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break


if __name__ == "__main__":
    main()
