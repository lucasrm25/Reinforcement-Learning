import gym
import numpy as np

# Init environment
env = gym.make("FrozenLake-v0")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration(maxiter=100):
    ''' Calculate optimal policy using the Value iteration algorithm

        NOTE: env.P[state][action] gives tuples (p, ns, r, is_terminal), which tells the
        probability p that we end up in the next state ns and receive reward r -> p(s',r|s,a)
    '''

    V = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8

    # iterate value function until convergence
    for _ in range(maxiter):

        # value function for the next iteration
        Vn = np.array([
            np.max( [
                    np.sum( [ p*(r+gamma*V[sn]) for p, sn, r, _ in env.P[s][a] ] )  # Bellman equation
                    for a in range(n_actions)
            ])
            for s in range(n_states)
        ])

        if np.max(Vn-V) <= theta: break
        else: V = Vn

    # evaluate best policy (Bellman equation)
    optPolicy = np.array([
            np.argmax( [
                    np.sum( [ p*(r+gamma*V[sn]) for p, sn, r, _ in env.P[s][a] ] )
                    for a in range(n_actions)
            ])
            for s in range(n_states)
        ])
    
    return optPolicy


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
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
