# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given Reinforcement Learning environment using Q-Learning and comparing the state values with the First Visit Monte Carlo method.

## PROBLEM STATEMENT
For the given frozen lake environment, find the optimal policy applying the Q-Learning algorithm and compare the value functions obtained with that of First Visit Monte Carlo method. Plot graphs to analyse the difference visually.

## Q LEARNING ALGORITHM
# Step 1:
Store the number of states and actions in a variable, initialize arrays to store policy and action value function for each episode. Initialize an array to store the action value function.
# Step 2: 
Define function to choose action based on epsilon value which decides if exploration or exploitation is chosen.
# Step 3:
Create multiple learning rates and epsilon values.
# Step 4: 
Run loop for each episode, compute the action value function but in Q-Learning the maximum action value function is chosen instead of choosing the next state and next action's value. 
# Step 5:
Return the computed action value function and policy. Plot graph and compare with Monte Carlo results.
## Q LEARNING FUNCTION
### Name: HIRUTHIK SUDHAKAR
### Register Number: 212223240054

```PY
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
      state, done=env.reset(), False
      while not done:
        action=select_action(state, Q, epsilons[e])
        next_state, reward, done, _ = env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q, axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```




## OUTPUT:
### Mention the optimal policy, optimal value function , success rate for the optimal policy.
<BR>
<img width="369" height="289" alt="image" src="https://github.com/user-attachments/assets/efd74bfd-897a-4ecf-aa01-5ddc6549c8a3" />
<BR>
<img width="805" height="619" alt="image" src="https://github.com/user-attachments/assets/07842942-3f9c-49b5-ab89-344a451a3cfe" />
<BR>
<img width="590" height="129" alt="image" src="https://github.com/user-attachments/assets/9b727d19-bff7-48a5-b3dc-9082d7f22109" />
<BR>


### Include plot comparing the state value functions of Monte Carlo method and Qlearning.
<BR>
<img width="1787" height="777" alt="image" src="https://github.com/user-attachments/assets/6994163d-af02-431e-8fbd-dc54e2b28286" />
<BR>
<img width="1832" height="777" alt="image" src="https://github.com/user-attachments/assets/59e2ad65-3092-4439-b9df-e04950fab9f0" />
<BR>


## RESULT:

Therefore, python program to find optimal policy using Q-Learning is developed and state value function obtained is compared with first visit monte carlo.
