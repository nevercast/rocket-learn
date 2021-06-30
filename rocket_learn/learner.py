import os
import pickle
import time

import numpy as np

from redis import Redis
import msgpack

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import multiprocessing as mp

import rocket_learn.worker
from rocket_learn.experience_buffer import ExperienceBuffer



#example pytorch stuff, delete later
actor = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, action_dim),
    nn.Softmax(dim=-1)
)

# critic
critic = nn.Sequential(
    nn.Linear(state_dim, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

class learner:
    def __init__(self):
        self.logger = SummaryWriter("log_directory")
        self.algorithm = PPO(actor, critic, self.logger)

        self.buffer = ExperienceBuffer()

        #**DEFAULT NEEDS TO INCORPORATE BASIC SECURITY, THIS IS NOT SUFFICIENT**
        self.redis = Redis(host='127.0.0.1', port=6379)



        # might be better to move this "main work step" to an external class
        # <-- build workers either here or externally to avoid linkage
        # should we use rolv's sb3 code or can we do better not being tied to sb3?
        # ROLV COMMENT:
        # SB3 has a bunch of useful stuff even if their PPO impl is a little strange.
        # Could make our own PPO by subclassing SB3 stuff, but may place unknown restrictions on us at some stage



        #what's a good model number scheme and do you account for continuing training from a saved model?
        model_version = 1

        # this is an ugly way of doing it
        TRAJ_ROUNDS = 10
        traj_count = 0
        while True:
            self.buffer.add_step(self.recieve_worker_data())
            traj_count += 1

            if traj_count >= TRAJ_ROUNDS:
                self.calculate(self.buffer)

                self.buffer.clear()
                traj_count = 0

                # SOREN COMMENT:
                # how do you want to combine the actor/critic dictionary for transmission?
                worker.update_model(self.redit, <<add state dict dump>>, model_version)
                model_version += 1

    def recieve_worker_data(self):
        while True:
            item = self.redis.lpop(ROLLOUTS)
            if item is not None:
                rollout = msgpack.loads(item)
                yield rollout
            else:
                time.sleep(10)

    def calculate(self):
        #apply PPO now but separate so we can refactor to allow different algorithm types
        self.algorithm.calculate(self.buffer)


#this should probably be in its own file
class PPO:
    def __init__(self, actor, critic, logger, n_rollouts = 36, lr_actor = 3e-4, lr_critic = 3e-4, gamma = 0.9, epochs = 1):
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer

        self.logger = logger

        #hyperparameters
        self.epochs = epochs
        self.gamma = gamma
        self.gae_lambda = 0
        self.batch_size = 512
        self.clip_range = .2
        self.ent_coef = 1
        self.vf_coef = 1
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])


    def calculate(self, buffer: ExperienceBuffer):
        values = self.critic(buffer)
        buffer_size = buffer.size()

        #totally stole this section from
        #https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
        #I am not attached to it, make it better if you'd like
        returns = []
        gae = 0
        for i in reversed(range(buffer_size)):
            delta = buffer.rewards[i] + self.gamma * values[i + 1] * buffer.dones[i] - values[i]
            gae = delta + self.gamma * lmbda * buffer.dones[i] * gae
            returns.insert(0, gae + values[i])

        advantages = np.array(returns) - values[:-1]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        #returns is also called references?

        for e in range(self.epochs):
            # ROLV COMMENT:
            # Is there some overview of different methods [of PPO] (pros/cons)?
            # If there is something that is closer to our task that should be preferred.
            # SOREN COMMENT:
            # I don't know enough to strongly favor one so I'm going with SB3 cause we'll have
            # a reference

            # this is mostly pulled from sb3
            for i, rollout in enumerate(self.buffer.generate_rollouts(self.batch_size)):
                #this should probably be consolidated into buffer
                adv = self.advantages[i:i+batch_size]

                # SOREN COMMENT:
                # need to figure out the right way to do this. agent is a part of worker
                # and agent has the distribution. Return as a part of rollouts?
                <<<<CONTINUE HERE, NEED TO GET THESE>>>>
                log_prob = XXXXX
                old_log_prob = XXXXX
                entropy = XXXXXX

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # If we want value clipping, add it here
                value_loss = F.mse_loss(returns, values)

                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()

                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()


                #self.logger write here to log results

