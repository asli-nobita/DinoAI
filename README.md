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
The policy is a series of actions that the model takes. An optimal policy is that series of actions which, given a state, will yield the best possible reward. 
### Q-value function / state action value function `Q(s,a)` 
The Q-value is function with two parameters, a state s and action a. It is defined by the reward gained if you start in state s, take the action a once, and then behave optimally after that. 
What does behaving optimally mean here?  
Suppose we start from state 5. We can either go right straight to state 6 for the 40 reward, or keep going left to state 1 for the 100 reward. If there's no discount, then clearly going left is the way to go since we get a larger reward. If we do consider a discount rate, however, we need to calculate the discounted reward for each of these paths. Assuming a discount rate of `0.5`, the rewards for each of these paths are:  
* Going right to 6: `r = 0 + 0.5*40 = 20`
* Going left to 1: `r = 0 + 0.5*0 + 0.5^2*0 + 0.5^3*0 + 0.5^4*100 = 6.25`
We can see that moving right is the optimal action in this case. This is what is meant by moving optimally.

Now that we figured out the optimal policy for state 5, we can find out the Q-value Q(s,a) for s = 5 and a = left or right. Q(5,left) is equal to moving left from state 5 to state 4, and then moving optimally from there. First we find the optimal path from state 4:  
* left to 1: `r = 0 + 0.5*0 + 0.5^2*0 + 0.5^3*100 = 12.5`
* right to 6: `r = 0 + 0.5*0 + 0.5^2*40 = 10`
So the optimal path from state 4 is moving left to 1. Now we can calculate Q(5,left):
`Q(5, left) = 0 + 0.5*0 + 0.5^2*0 + 0.5^3*0 + 0.5^4*100 = 6.25`
and Q(5, right):
`Q(5, right) = 0 + 0.5*40 = 20`
Q(5, right) is greater than Q(5, left). Thus the optimal policy from state 5 is to move right to 6.
Note that the optimal policy and the Q value depend on the rewards assigned to each state and also the discount rate. If the reward for state 6 was 10 for example, the optimal policy would be to go left to 1 even at state 5.
### Bellman equation  
The Bellman equation is the most important equation in RL. It states that, for a discount rate α,  
`Q(s,a) = R(s) + α*max(Q(s',a')`  
that is, the Q-value of state s with action a is equal to the reward assigned to state s + α times the maximum possible Q-value at s', the state reached upon by performing action a from state s, that can be achieved using a possible action a'. In our mars rover example, max(Q(5,a')) is 20 for a'=right. The maximum value of Q function is also the maximum reward achievable from that state.  
## Description of the project  
This project is very suitable as a beginner level project in RL, as it is relatively easy to implement in the short time frame I had, yet covers all the basics of RL - creating a custom environment, implementing the reward system, application of computer vision, training the model through a deep Q-learning neural network (DQN) and testing it. I learnt a lot from the project and gained some idea of what we can do using RL algorithms. Some other applications of RL could be in designing a rover or a drone to navigate terrain, or in algorithmic trading systems to identify good trades. I hope to take on such projects in the future.
### Creating the environment  
I had to create a custom environment to play the Chrome dino game, using the `Box` method from the **Gymnasium** library. I used **mss** to capture screenshots of a small region just in front of the dino, scaled the resolution down and converted the image to greyscale, using **OpenCV**. Since I needed a way to signal to the model when a run ended and it had reached the game over screen, I captured the 'GAME OVER' text and performed OCR on the image using **pytesseract** library, passing the extracted text to a method in the environment class and checking the text contained the word 'GAME'. I arbitrarily assigned a reward value of 1 for each frame the agent stays alive.  
### Training the model  
The model was trained using a DQN from **stable-baselines3** to learn the Q values. 
## Results and scope of improvement  
The program works as intended. We can check the mean reward rate per episode during training increases as number of timesteps increase. In the game, our dino manages to avoid the first few obstacles. That being said, there are some shortcomings in the project that I have noticed and attempted to rectify:  
* fps/timesteps per second observed during training is quite low (around 2 fps). This means the model takes a really long time to cover, say 10,000 timesteps (or long enough to actually get a finely tuned model). Now this might be in part due to hardware limitations, but one major reason could be the implementation of the environment itself. Maybe the OCR at the end of every run is slowing the algorithm down. I tried to fix this by focusing on a very tiny region near the upper left corner of the text. This region would be white and turn to grey once a run is over.
* The `game_location` and `done_location` defined in the environment are in absolute pixels, meaning that resizing the game window can cause errors in the program.
* The agent keeps performing random actions. Ideally, the agent should only perform an action to avoid an obstacle and do nothing when no obstacle is in sight. I tried to fix this by allotting a small reward for no action. <!--- mention issue faced doing this -->
* `pydirectinput` works only on windows. There is a library called `pyautogui` that works cross platform on Mac and Linux as well, but when I tried to use it in my project it throws up all sorts of errors.
## Conclusion  
Overall, it was a fun project to make. I managed to learn quite a few things and gained some idea of where RL can be applied. A chess playing bot or tetris playing bot. An RL based movie/book recommendation system. Navigation of drones/robots. Financial trading algorithm. These are much more complex and require knowledge of many things other than RL as well. All of them seem exciting though, and I'd be looking forward to working on one of these projects in the future together with the club.
