import gym
import numpy as np

# Init environment
# Lets use a smaller 3x3 custom map for faster computations
custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
env = gym.make("FrozenLake-v0", desc=custom_map3x3)
# TODO: Uncomment the following line to try the default map (4x4):
#env = gym.make("FrozenLake-v0")

# Uncomment the following lines for even larger maps:
#random_map = generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)

# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

r = np.zeros(n_states) # the r vector is zero everywhere except for the goal state (last state)
r[-1] = 1.


def trans_matrix_for_policy(policy):
    ''' Returns the transition probability matrix P for a policy
    Returns:
        P(s,s'): Transition matrix P_{i,j} = p(s'=j|s=i)
               = p(s'|s) = Sum_a pi(a|s)*p(s'|s,a)
    '''
    transitions = np.zeros((n_states, n_states))
    for s in range(n_states):
        probs = env.P[s][policy[s]]         # env.P(s,a) = env.P(s,pi(s)) -> [ p(s'|s,a), s', r(s',s,a)=E[r|s',a,s], isterminal ]
        for el in probs:                    # P(s,s') = p(s'|s) = Sum_a p(s'|s,a)
            transitions[s, el[1]] += el[0]
    return transitions


def terminals():
    """ This is a helper function that returns terminal states 
    """
    terms = []
    for s in range(n_states):
        # terminal is when we end with probability 1 in terminal:
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
            terms.append(s)
    return terms


def value_policy(policy, gamma=0.8):
    '''
    V(s) = R(s) + P(s,s') @ gamma*V(s')   ->  V = (I - gamma * P)^-1 @ R
         = r(s) + Sum_{s'}  p(s'|s) * gamma * V(s')

    where  P(s,s') = p(s'|s) = Sum_a pi(a|s)*p(s'|s,a)
    '''
    P = trans_matrix_for_policy(policy)
    # calculate and return v. (P, r and gamma already given)
    I = np.eye( P.shape[0] )
    V_pi = np.linalg.inv(I - gamma * P) @ r
    return V_pi


def bruteforce_policies():
    from itertools import product

    terms = terminals()
    
    # here we consider all possible policies. But for terminal states we dont care about the policy, 
    # so we just consider always the policy pi(s=terminalstate) = 0
    possibleactions = [ list(range(n_actions)) if s not in terms else [0] for s in range(n_states)]
    allpolicies = list(product(*possibleactions)) # allpolicies = [ele for ele in product(range(n_actions), repeat=n_states)]

    # calculate value function for all possible policies and 
    # select the one that has higher value function for all states
    optimalvalue = np.zeros(n_states)
    for policy in allpolicies:
        value = value_policy(policy)        
        if np.all( value >= optimalvalue ): # if value[0] > optimalvalue[0]:
            optimalvalue = value
            optimalpolicy = policy

    optimalpolicies = [optimalpolicy]

    print ("Optimal value function:")
    print(optimalvalue)
    print ("number optimal policies:")
    print (len(optimalpolicies))
    print ("optimal policies:")
    print (np.array(optimalpolicies))
    return optimalpolicies



def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # Here a policy is just an array with the action for a state as element
    policy_left = np.zeros(n_states, dtype=np.int)  # 0 for all states
    policy_right = np.ones(n_states, dtype=np.int) * 2  # 2 for all states

    # Value functions:
    print("Value function for policy_left (always going left):")
    print (value_policy(policy_left))
    print("Value function for policy_right (always going right):")
    print (value_policy(policy_right))

    optimalpolicies = bruteforce_policies()


    # "rollout" a policy in the environment:
    print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(optimalpolicies[0][state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break


if __name__ == "__main__":
    main()
