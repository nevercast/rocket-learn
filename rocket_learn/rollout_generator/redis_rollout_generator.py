import functools
import itertools
import os
import random
import zlib
from concurrent.futures import ProcessPoolExecutor
from threading import Thread

import cloudpickle as pickle
import time
from collections import Counter
from typing import Iterator, Callable, List
from uuid import uuid4

import msgpack
import msgpack_numpy as m
import numpy as np
# import matplotlib.pyplot  # noqa
import psutil
import wandb
# from matplotlib.axes import Axes
# from matplotlib.figure import Figure
from gym.vector.utils import CloudpickleWrapper
from redis import Redis
from redis.exceptions import ResponseError
from rlgym.utils import ObsBuilder, RewardFunction
from rlgym.utils.action_parsers import ActionParser
from trueskill import Rating, rate, SIGMA
import plotly.graph_objs as go

from rlgym.envs import Match
from rlgym.gamelaunch import LaunchPreference
from rlgym.gym import Gym
from rlgym.utils.gamestates import GameState
from rocket_learn.backends.redis_backend import RedisBackend
from rocket_learn.experience_buffer import ExperienceBuffer
from rocket_learn.rollout_generator.base_rollout_generator import BaseRolloutGenerator
from rocket_learn.utils import util
from rocket_learn.utils.batched_obs_builder import BatchedObsBuilder
from rocket_learn.utils.util import encode_gamestate, probability_NvsM, softmax

# Constants for consistent key lookup
QUALITIES = "qualities"
N_UPDATES = "num-updates"
SAVE_FREQ = "save-freq"

MODEL_LATEST = "model-latest"
VERSION_LATEST = "model-version"

ROLLOUTS = "rollout"
OPPONENT_MODELS = "opponent-models"
WORKER_IDS = "worker-ids"
CONTRIBUTORS = "contributors"
_ALL = (
    QUALITIES, N_UPDATES, SAVE_FREQ, MODEL_LATEST, VERSION_LATEST, ROLLOUTS, OPPONENT_MODELS,
    WORKER_IDS, CONTRIBUTORS)

m.patch()


# Helper methods for easier changing of byte conversion
def _serialize(obj):
    return zlib.compress(msgpack.packb(obj))


def _unserialize(obj):
    return msgpack.unpackb(zlib.decompress(obj))


def _serialize_model(mdl):
    device = next(mdl.parameters()).device  # Must be a better way right?
    mdl_bytes = pickle.dumps(mdl.cpu())
    mdl.to(device)
    return mdl_bytes


def _unserialize_model(buf):
    agent = pickle.loads(buf)
    return agent


def encode_buffers(buffers: List[ExperienceBuffer], strict=False, send_rewards=True):
    if strict:
        states = np.asarray([encode_gamestate(info["state"]) for info in buffers[0].infos])
        actions = np.asarray([buffer.actions for buffer in buffers])
        log_probs = np.asarray([buffer.log_probs for buffer in buffers])
        if send_rewards:
            rewards = np.asarray([buffer.rewards for buffer in buffers])
            return states, actions, log_probs, rewards
        return states, actions, log_probs
    else:
        return [
            (buffer.meta, buffer.observations, buffer.actions, buffer.rewards, buffer.dones, buffer.log_probs)
            for buffer in buffers
        ]


def decode_buffers(enc_buffers, versions, encoded, obs_build_factory=None, rew_func_factory=None,
                   act_parse_factory=None):
    if encoded:
        if len(enc_buffers) == 3:
            game_states, actions, log_probs = enc_buffers
            rewards = None
        elif len(enc_buffers) == 4:
            game_states, actions, log_probs, rewards = enc_buffers
        else:
            raise ValueError

        obs_builder = obs_build_factory()
        act_parser = act_parse_factory()

        if isinstance(obs_builder, BatchedObsBuilder):
            assert rewards is not None
            obs = obs_builder.batched_build_obs(game_states[:-1])
            prev_actions = act_parser.parse_actions(actions.reshape((-1,) + actions.shape[2:]).copy(), None).reshape(
                actions.shape[:2] + (8,))
            prev_actions = np.concatenate((np.zeros((actions.shape[0], 1, 8)), prev_actions[:, :-1]), axis=1)
            obs_builder.add_actions(obs, prev_actions)
            dones = np.zeros_like(rewards, dtype=bool)
            dones[-1, :] = True
            buffers = [
                ExperienceBuffer(observations=[obs[i]], actions=actions[i], rewards=rewards[i], dones=dones[i],
                                 log_probs=log_probs[i])
                for i in range(len(obs))
            ]
            return buffers

        game_states = [GameState(gs.tolist()) for gs in game_states]
        rew_func = rew_func_factory()
        obs_builder.reset(game_states[0])
        rew_func.reset(game_states[0])
        buffers = [
            ExperienceBuffer(infos=[{"state": game_states[0]}])
            for _ in range(len(game_states[0].players))
        ]

        env_actions = [
            act_parser.parse_actions(actions[:, s, :].copy(), game_states[s])
            for s in range(actions.shape[1])
        ]

        obss = [obs_builder.build_obs(p, game_states[0], np.zeros(8))
                for i, p in enumerate(game_states[0].players)]
        for s, gs in enumerate(game_states[1:]):
            assert len(gs.players) == len(versions)
            final = s == len(game_states) - 2
            old_obs = obss
            obss = []
            i = 0
            for version in versions:
                if version == 'na':
                    continue  # don't want to rebuild or use prebuilt agents
                player = gs.players[i]

                # IF ONLY 1 buffer is returned, need a way to say to discard bad version

                obs = obs_builder.build_obs(player, gs, env_actions[s][i])
                if rewards is None:
                    if final:
                        rew = rew_func.get_final_reward(player, gs, env_actions[s][i])
                    else:
                        rew = rew_func.get_reward(player, gs, env_actions[s][i])
                else:
                    rew = rewards[i][s]
                buffers[i].add_step(old_obs[i], actions[i][s], rew, final, log_probs[i][s], {"state": gs})
                obss.append(obs)
            i += 1

        return buffers
    else:
        buffers = []
        for enc_buffer in enc_buffers:
            meta, obs, actions, rews, dones, log_probs = enc_buffer
            buffers.append(
                ExperienceBuffer(meta=meta, observations=obs, actions=actions,
                                 rewards=rews, dones=dones, log_probs=log_probs)
            )
        return buffers


class RedisRolloutGenerator(BaseRolloutGenerator):
    """
    Rollout generator in charge of sending commands to workers via redis
    """

    def __init__(
            self,
            redis: Redis,
            obs_build_factory: Callable[[], ObsBuilder],
            rew_func_factory: Callable[[], RewardFunction],
            act_parse_factory: Callable[[], ActionParser],
            save_every=10,
            logger=None,
            clear=True,
            mmr_min_episode_length=150,
            max_age=0
    ):
        self.tot_bytes = 0
        self.redis = redis
        self.logger = logger

        # TODO saving/loading
        if clear:
            self.redis.delete(*_ALL)
            self.redis.set(N_UPDATES, 0)
        else:
            if self.redis.exists(ROLLOUTS) > 0:
                self.redis.delete(ROLLOUTS)
            self.redis.decr(VERSION_LATEST, 2)  # In case of reload from old version, don't let current seep in

        self.redis.set(SAVE_FREQ, save_every)
        self.contributors = Counter()  # No need to save, clears every iteration
        self.obs_build_func = obs_build_factory
        self.rew_func_factory = rew_func_factory
        self.act_parse_factory = act_parse_factory
        self.mmr_min_episode_length = mmr_min_episode_length
        self.pretrained_agents = {}
        self.max_age = max_age

    @staticmethod
    def _process_rollout(rollout_bytes, latest_version, obs_build_func, rew_build_func, act_build_func, max_age):
        rollout_data, versions, uuid, name, result, encoded = _unserialize(rollout_bytes)

        v_check = [v for v in versions if isinstance(v, int)]

        if any(version < 0 and abs(version - latest_version) > max_age for version in v_check):
            return

        buffers = decode_buffers(rollout_data, versions, encoded, obs_build_func, rew_build_func, act_build_func)
        return buffers, versions, uuid, name, result

    def _update_ratings(self, name, versions, buffers, latest_version, result):
        ratings = []
        relevant_buffers = []
        for version, buffer in itertools.zip_longest(versions, buffers):
            if version == 'na':
                continue  # no need to rate pretrained agents
            elif version < 0:
                if abs(version - latest_version) <= self.max_age:
                    relevant_buffers.append(buffer)
                    self.contributors[name] += buffer.size()
                else:
                    return []
            else:
                rating = Rating(*_unserialize(self.redis.lindex(QUALITIES, version)))
                ratings.append(rating)

        # Only old versions, calculate MMR
        if len(ratings) == len(versions) and len(buffers) == 0:
            blue_players = sum(divmod(len(ratings), 2))
            blue = tuple(ratings[:blue_players])  # Tuple is important
            orange = tuple(ratings[blue_players:])

            # In ranks lowest number is best, result=-1 is orange win, 0 tie, 1 blue
            r1, r2 = rate((blue, orange), ranks=(0, result))

            # Some trickery to handle same rating appearing multiple times, we just average their new mus and sigmas
            ratings_versions = {}
            for rating, version in zip(r1 + r2, versions):
                ratings_versions.setdefault(version, []).append(rating)

            for version, ratings in ratings_versions.items():
                avg_rating = Rating((sum(r.mu for r in ratings) / len(ratings)),
                                    (sum(r.sigma ** 2 for r in ratings) ** 0.5 / len(ratings)))  # Average vars
                self.redis.lset(QUALITIES, version, _serialize(tuple(avg_rating)))

        return relevant_buffers

    def generate_rollouts(self) -> Iterator[ExperienceBuffer]:
        while True:
            latest_version = int(self.redis.get(VERSION_LATEST))
            data = self.redis.blpop(ROLLOUTS)[1]
            self.tot_bytes += len(data)
            res = self._process_rollout(
                data, latest_version,
                self.obs_build_func, self.rew_func_factory, self.act_parse_factory,
                self.max_age
            )
            if res is not None:
                buffers, versions, uuid, name, result = res
                versions = [version for version in versions if version != 'na']  # don't track humans or hardcoded

                relevant_buffers = self._update_ratings(name, versions, buffers, latest_version, result)
                yield from relevant_buffers

        # futures = []
        # cpus = psutil.cpu_count(logical=False)
        # with ProcessPoolExecutor(cpus) as ex:
        #     while True:
        #         # Kinda scuffed ngl
        #         if len(futures) > 0 and futures[0].done():
        #             res = futures.pop(0).result()
        #             if res is not None:
        #                 latest_version = int(self.redis.get(VERSION_LATEST))
        #                 buffers, versions, uuid, name, result = res
        #                 relevant_buffers = self._update_ratings(name, versions, buffers, latest_version, result)
        #                 yield from relevant_buffers
        #         elif len(futures) < 2 * cpus:
        #             latest_version = int(self.redis.get(VERSION_LATEST))
        #             data = self.redis.blpop(ROLLOUTS)[1]
        #             self.tot_bytes += len(data)
        #             futures.append(ex.submit(
        #                 RedisRolloutGenerator._process_rollout,
        #                 data,
        #                 latest_version,
        #                 CloudpickleWrapper(self.obs_build_func),
        #                 CloudpickleWrapper(self.rew_func_factory),
        #                 CloudpickleWrapper(self.act_parse_factory)
        #             ))

    def _plot_ratings(self, ratings):
        if len(ratings) <= 0:
            return
        mus = np.array([r.mu for r in ratings])
        mus = mus - mus[0]
        sigmas = np.array([r.sigma for r in ratings])
        # sigmas[1:] = (sigmas[1:] ** 2 + sigmas[0] ** 2) ** 0.5

        x = np.arange(len(mus))
        y = mus
        y_upper = mus + 2 * sigmas  # 95% confidence
        y_lower = mus - 2 * sigmas

        fig = go.Figure([
            go.Scatter(
                x=x,
                y=y,
                line=dict(color='rgb(0,100,80)'),
                mode='lines',
                name="mu",
                showlegend=False
            ),
            go.Scatter(
                x=np.concatenate((x, x[::-1])),  # x, then x reversed
                y=np.concatenate((y_upper, y_lower[::-1])),  # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',  # TODO same color as wandb run?
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name="sigma",
                showlegend=False
            ),
        ])

        fig.update_layout(title="Rating", xaxis_title="Iteration", yaxis_title="TrueSkill")

        self.logger.log({
            "qualities": fig,
        }, commit=False)

    def _add_opponent(self, agent):
        # Add to list
        self.redis.rpush(OPPONENT_MODELS, agent)
        # Set quality
        ratings = [Rating(*_unserialize(v)) for v in self.redis.lrange(QUALITIES, 0, -1)]
        if ratings:
            quality = Rating(ratings[-1].mu, SIGMA)
        else:
            quality = Rating(0, 1)  # First (basically random) agent is initialized at 0
        self.redis.rpush(QUALITIES, _serialize(tuple(quality)))

    def update_parameters(self, new_params):
        """
        update redis (and thus workers) with new model data and save data as future opponent
        :param new_params: new model parameters
        """
        model_bytes = _serialize_model(new_params)
        self.redis.set(MODEL_LATEST, model_bytes)
        self.redis.decr(VERSION_LATEST)

        print("Top contributors:\n" + "\n".join(f"{c}: {n}" for c, n in self.contributors.most_common(5)))
        self.logger.log({
            "contributors": wandb.Table(columns=["name", "steps"], data=self.contributors.most_common())},
            commit=False
        )
        self._plot_ratings([Rating(*_unserialize(v)) for v in self.redis.lrange(QUALITIES, 0, -1)])
        tot_contributors = self.redis.hgetall(CONTRIBUTORS)
        tot_contributors = Counter({name: int(count) for name, count in tot_contributors.items()})
        tot_contributors += self.contributors
        if tot_contributors:
            self.redis.hset(CONTRIBUTORS, mapping=tot_contributors)
        self.contributors.clear()

        self.logger.log({"rollout_bytes": self.tot_bytes}, commit=False)
        self.tot_bytes = 0

        n_updates = self.redis.incr(N_UPDATES) - 1
        save_freq = int(self.redis.get(SAVE_FREQ))

        if n_updates % save_freq == 0:
            # self.redis.set(MODEL_N.format(self.n_updates // self.save_every), model_bytes)
            self._add_opponent(model_bytes)
            try:
                self.redis.save()
            except ResponseError:
                print("redis manual save aborted, save already in progress")


class RedisRolloutWorker:
    """
    Provides RedisRolloutGenerator with rollouts via a Redis server
    """

    def __init__(self, redis: Redis, name: str, match: Match,
                 past_version_prob=.2, evaluation_prob=0.01, sigma_target=1,
                 streamer_mode=False, send_gamestates=True, pretrained_agents=None, human_agent=None,
                 deterministic_old_prob=0.5, force_paging=False):
        # TODO model or config+params so workers can recreate just from redis connection?
        self.name = name

        self.backend = RedisBackend(redis)

        self.pretrained_agents = {}
        self.pretrained_total_prob = 0
        if pretrained_agents is not None:
            self.pretrained_agents = pretrained_agents
            self.pretrained_total_prob = sum([self.pretrained_agents[key] for key in self.pretrained_agents])

        self.human_agent = human_agent

        if human_agent and pretrained_agents:
            print("** WARNING - Human Player and Pretrain Agents are in conflict. **")
            print("**           Pretrained Agents will be ignored.                **")

        self.streamer_mode = streamer_mode

        self.current_agent = self.backend.get_latest_model()
        self.past_version_prob = past_version_prob
        self.evaluation_prob = evaluation_prob
        self.sigma_target = sigma_target
        self.send_gamestates = send_gamestates
        self.deterministic_old_prob = deterministic_old_prob

        # TODO: Wtf is the point in worker ids, does the API need this? :D

        if not self.streamer_mode:
            print("Started worker on host", redis.connection_pool.connection_kwargs.get("host"),
                  "under name", name)  # TODO log instead
        else:
            print("Streaming mode set. Running silent.")

        self.match = match
        self.env = Gym(match=self.match, pipe_id=os.getpid(), launch_preference=LaunchPreference.EPIC,
                       use_injector=True, force_paging=force_paging)
        self.n_agents = self.match.agents
        self.total_steps_generated = 0

    def run(self):  # Mimics Thread
        """
        begin processing in already launched match and push to redis
        """
        n = 0
        latest_version = None
        t = Thread()
        t.start()
        while True:
            self.current_agent, latest_version = self.backend.get_latest_model()
            if self.current_agent is None:
                print("No agent found, waiting for new agent")
                time.sleep(1)
                continue

            n += 1
            pretrained_choice = None

            match_teams = self.backend.get_match_versions()
            versions = [v for team in match_teams for v in team]
            
            agents = []
            for version in versions:
                if version == -1:
                    agents.append(self.current_agent)
                else:
                    selected_agent = self.backend.get_opponent(version)
                    if np.random.random() < self.deterministic_old_prob:
                        selected_agent.deterministic = True
                    agents.append(selected_agent)
            versions = [v if v != -1 else latest_version for v in versions]

            encode = self.send_gamestates
            if all(isinstance(v, int) for v in versions) and all(v >= 0 for v in versions) \
                    and not self.streamer_mode and self.human_agent is None:
                print("Running evaluation game with versions:", versions)
                result = util.generate_episode(self.env, agents, evaluate=True)
                rollouts = []
                print("Evaluation finished, goal differential:", result)
                encode = False
            else:
                version_info = []
                for v in versions:
                    if pretrained_choice is not None and v == 'na':  # print name but don't send it back
                        version_info.append(str(type(pretrained_choice).__name__))
                    elif v == 'na':
                        version_info.append('Human_player')
                    else:
                        version_info.append(str(v))

                if not self.streamer_mode:
                    print("Generating rollout with versions:", version_info)

                rollouts, result = util.generate_episode(self.env, agents, evaluate=False)
                if len(rollouts[0].observations) <= 1:
                    rollouts, result = util.generate_episode(self.env, agents, evaluate=False)

                if len(rollouts[0].observations) <= 1:
                    print(" ** Rollout Generation Error: Restarting Generation ** ")
                    continue

                state = rollouts[0].infos[-2]["state"]
                goal_speed = np.linalg.norm(state.ball.linear_velocity) * 0.036  # kph
                str_result = ('+' if result > 0 else "") + str(result)
                self.total_steps_generated += len(rollouts[0].observations) * len(rollouts)
                post_stats = f"Rollout finished after {len(rollouts[0].observations)} steps ({self.total_steps_generated} total steps), result was {str_result}"
                if result != 0:
                    post_stats += f", goal speed: {goal_speed:.2f} kph"

                if not self.streamer_mode:
                    print(post_stats)

            if not self.streamer_mode:
                rollout_data = encode_buffers(rollouts, strict=encode)  # TODO change
                # sanity_check = decode_buffers(rollout_data, encode,
                #                               lambda: self.match._obs_builder,
                #                               lambda: self.match._reward_fn,
                #                               lambda: self.match._action_parser)
                rollout_bytes = _serialize((rollout_data, versions, self.uuid, self.name, result,
                                            encode))
                
                self.backend.push_rollout(rollout_bytes)
