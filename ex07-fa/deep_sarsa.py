''' 
TODO:
    - [ ] experience replay
'''

import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('MountainCar-v0')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, s):
        # x = torch.cat((s, a), 1)
        x = self.fc1(s)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

net = Net()
# print(net)
# print(list(net.parameters()))
# test network
s_test = torch.randn(1, 2)
net(s_test)



def deep_sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay = 1.0, num_ep=int(1e4), ep_length=200):
    ''' Implementation of the SARSA algorithm (on-policy TD control) 
    '''
    render_every_X_ep = 100
    print_every_X_ep = 100

    n_actions =  env.action_space.n

    # Initialize arbitrarily Q(s; a)
    Q = Net()

    nbr_succ = []
    nbr_step = []

    optimizer = torch.optim.SGD(net.parameters(), lr=alpha)

    for i in range(num_ep):

        # init state
        s = env.reset()
        s = torch.tensor(s.astype(np.float32)).unsqueeze(0)
        # choose action from behaviour policy: epsilon-greedy
        a = torch.argmax(Q(s)) if np.random.binomial(1,1-epsilon) else torch.tensor(np.random.randint(n_actions))
        
        r_total = 0
        for j in range(ep_length):

            if not i % render_every_X_ep and i > 0: 
                env.render()
        
            optimizer.zero_grad()

            # take action a from behaviour policy, observe R, S'
            with torch.no_grad():
                s_, r, done, _ = env.step(a.numpy())
                s_ = torch.tensor(s_.astype(np.float32)).unsqueeze(0)
            
            # behaviour policy: epsilon-greedy
            a_ = torch.argmax(Q(s_)) if np.random.binomial(1,1-epsilon) else torch.tensor(np.random.randint(n_actions))
            
            # evaluate policy - update Q-value function
            # target policy a_ == behaviour policy: epsilon-greedy
            loss = F.mse_loss( Q(s)[:,a] , r + gamma * Q(s_)[:,a_] )  # Q[s,a] += alpha * ( r + gamma * Q[s_,a_] - Q[s,a] )
            loss.backward()
            optimizer.step()

            # update state and action
            s, a = s_, a_

            r_total += r
            if done: 
                break

        nbr_succ.append(s_[0,0] >= 0.5)
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
    Q = deep_sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.9, epsilon_decay=1-1e-3, num_ep=int(10e3), ep_length=200)
    play_episode(env, Q)
    env.close()


if __name__ == "__main__":
    main()
