<div align="center">
<img src="./docs/img/icing-on-the-rl.png"/>
<h1>Icing on the RL</h1>
<p>Feel free to configure environment and hyper-parameter easily</p>
</div>

## Introduce

Icing on the RL is library that helps to config environmet, train, test, plot results and also learning reinforcement learning  
There are various good RL libraries to study RL methods, but I have a little bit confuse and complex to analyze inside code for studying. So I implement RL methods from scratch for learning and feel pleasure of writing and testing RL methods  

The goal of thest project is easy to train, test, evaluate RL methods and also easy to learning RL methods from inside codes  
  
### Title
- [How to use](#how-to-use)
- [Methods](#methods)

## How to use
### DQL
We will implement DQL same as [pytorch-DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) in irl  
First, we need to wrapper gymnasium to irl env

```python
from irl.env import ActionType, Env

class GymWrapEnv(Env):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__()
        self.env = gym.make(name, **kwargs)

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return obs

    def step(self, action: ActionType) -> tuple[np.ndarray, SupportsFloat, bool, bool]:
        obs, reward, terminated, truncated, _ = self.env.step(action)
        return obs, reward, terminated, truncated
    
    def select_random_action(self) -> ActionType:
        return self.env.action_space.sample()

env = GymWrapEnv("CartPole-v1")
```

Next, configure DQL agent

```python
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
net = DQN(4, 2).to(device)
agent = DQLAgent(net, device)
```
Next, configure DQL environment and training configuration
```python
# configure DQL environment (allow dql agent to train and eval on env)
dql = DQL(agent, env, device)

from irl.replay_memory.replay_memory import ReplayMemory
from irl.sampling.egreedy import ExponentialEgreedyPolicy

# configure loss function, optimizer, sampling policy, replay memory
loss = nn.SmoothL1Loss()
optimizer = optim.AdamW(net.parameters(), lr=learning_rate, amsgrad=True)
sampling_policy = ExponentialEgreedyPolicy(0.9, 1000, 0.05)
replay_memory = ReplayMemory(10000)
```

Next, configure hook when training

```python
import matplotlib
import matplotlib.pyplot as plt
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

from irl.methods.hook import BaseHook

class CustomHook(BaseHook):
    def __init__(self) -> None:
        self.episode_durations = []
        self.step = 0

    def after_step(self, step_n: int, obs: np.ndarray, reward: SupportsFloat) -> None:
        self.step += 1
    
    def after_episode(self, epi_n: int, score: float) -> None:
        self.episode_durations.append(self.step)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)

        plt.clf()
        plt.title('Training')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if is_ipython:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        self.step = 0
```

Next, we define optimizer and train agent

```python
dqn_optimzier = DQNOptimizer(
    net=net,
    criterion=loss,
    optimizer=optimizer,
    device=device,
    gamma=0.99,
    target_net=deepcopy(net),
    tau=0.005,
)

dql.train(
    episode=600,
    batch_size=128,
    dqn_optimizer=dqn_optimzier,
    sampling_policy=sampling_policy,
    replay_memory=replay_memory,
    hook=CustomHook(),
)
```

Also, you can evaluate after training

```python
# change env render mode to see result
env.env = gym.make("CartPole-v1", render_mode="human")
dql.eval(episode=3)
```


## Methods
Follow table shows implemented RL methods  
:heavy_check_mark: - implemented, :hourglass_flowing_sand: - proceeding, :x: - not yet  
|method|implemented|code reference|paper
|---|:---:|---|---|
|DQL (Deep Q Learning)|:heavy_check_mark:|https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html|[Human-level control through deep reinforcement learning (Nature - 2015)](https://www.nature.com/articles/nature14236)[^1]|
|PPO|:x:| | |


[^1] Playing Atari with Deep Reinforcement Learning is first introduction of DQL algorithm and next paper Human-level control throgh deep reinforcement learning proposes some advanced techniques


## Issue
If there is any issue in algorithm, code or docs etc. Feel free to make issue  