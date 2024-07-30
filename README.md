![image](https://github.com/user-attachments/assets/e7ca6527-f8a5-4229-bc28-d23a4fa46374)
# Training AI to play chrome dino, using Reinforcement Learning  
## Introduction  
**Reinforcement Learning** (RL) is one of the three forms of machine learning, along with supervised learning - where input data is labelled and the model tries to map input to the labels, and unsupervised learning, where no labels are given and it is left to the model to figure out underlying patterns in the data. RL however, is different from these. RL involves training the model (called the agent) by letting it choose from a set of actions at random, such as pressing a button, or clicking, and rewarding it when it reaches a favourable outcome (such as staying alive, or reaching the target) and can optionally be penalised for an unfavourable outcome. The model thus learns which actions to perform at what time to maximise the reward gained.  
## How RL works  
An RL model is an example of a Markov decision process. It involves the agent in a series of states and actions, and the future of the agent is decided only by the present state of the agent.  
![image](https://github.com/user-attachments/assets/2231c610-3376-45aa-b622-20c6a62e0ff3)
Take the (very simplified) example of a space rover navigating the Martian terrain, with 6 states and two actions, moving either left or right. One of the states, state 1, is the most desirable, kind of like the main base. It is where we want the rover to go. State 6 is also a favourable state, something like a smaller base in case the rover is low on fuel and cannot reach state 1. The rest of the states are of no particular importance. Using these, we assign a 'reward' to each state, 1 being the highest with a reward of 100, 6 having a reward of 40 and the rest with a reward of 0. The goal of the agent is to maximise the reward, so it will always try to reach state 1. The agent can change its state by either moving left or moving right from its current state. 
### Discount rate  
In many cases, we would want to include a "time aspect" to this reward situation, so to speak. For example, the rover has limited fuel, so it would like to get the best bang for its buck, getting best possible reward while managing fuel. Or in case of financial trading, a trade possible today is better than a similar trade a couple of days in the future. This is possible using the discount rate. The reward formula with a discount rate α is:  
`reward = r_i + α*r_(i+1) + (α^2)*r_(i+2) + ...`  
where `r_i` is the reward of the `i`th state. 
### Policy  
The policy is a series of actions that, according to the model, is optimal to get the best reward.  
### Q-value function / state action value function `Q(s,a)` 
The Q-value is function with two parameters, a state s and action a. It is defined by the reward gained if you start in state s, take the action a once, and then behave optimally after that. 
What does behaving optimally mean here?  
Suppose we start from state 5. We can either go right straight to state 6 for the 40 reward, or keep going left to state 1 for the 100 reward. If there's no discount, then clearly going left is the way to go since we get a larger reward. If we do consider a discount rate, however, we need to calculate the discounted reward for each of these paths. Assuming a discount rate of `0.5`, the rewards for each of these paths are:  
* Going right to 6: `r = 0 + 0.5*40 = 20`
* Going left to 1: `r = 0 + 0.5*0 + 0.5^2*0 + 0.5^3*0 + 0.5^4*100 = 6.67`
We can see that moving right is the optimal action in this case. This is what is meant by moving optimally.

Now that we figured out the optimal policy for state 5, we can find out the Q-value Q(s,a) for s = 6 and a = left or right. Q(6,left) is equal to moving left from state 6 to state 5, and then 
## Description of the project  
This project is very suitable as a beginner level project in RL, as it is relatively easy to implement in the short time frame I had, yet covers all the basics of RL - creating a custom environment, implementing the reward system, application of computer vision, training the model through a deep Q-learning neural network (DQN) and testing it. I learnt a lot from the project and gained some idea of what we can do using RL algorithms. Some other applications of RL could be in designing a rover or a drone to navigate terrain, or in algorithmic trading systems to identify good trades. I hope to take on such projects in the future.
