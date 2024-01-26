from collections import defaultdict
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn.functional as F
from Baseline_graph_direct_hill_regulation.causal_hill_regulation_5nodes import CausalGraph


def one_hot(length, idx):
    one_hot = np.zeros(length)
    one_hot[idx] = 1
    return one_hot


class CBNEnv(Env):
    def __init__(self,
                 agent_type='default',
                 info_phase_length=50,
                 reward_scale=[1.0, 1.0],
                 train=True):
        """Create a stable_baselines-compatible environment to train policies on"""
        self.action_space = Box(0, np.inf, (1,), dtype=np.float64)
        self.observation_space = Box(-np.inf, np.inf, (7,))  # 5(o_t) + 1(a_t-1) + 1(r_t-1)
        if train:
            self.state = TrainEnvState(info_phase_length)  # 一个继承了 EnvState class 的类
        else:
            self.state = TestEnvState(info_phase_length)
        self.reward_scale = reward_scale
        self.logger = None
        self.log_data = defaultdict(int)
        self.agent_type = agent_type
        self.ep_rew = 0
        self.goal = [[9, 10]]
        self.vertex = [4]
        self.goal_vertex = [1]

    @classmethod
    def create(cls, n_env, info_phase_length, reward_scale):
        return DummyVecEnv(
            [lambda: cls(info_phase_length=info_phase_length,
                         reward_scale=reward_scale)
             for _ in range(n_env)])

    def reward(self, val):

        max_state = torch.from_numpy(np.array(val))  # 转为tensor
        score_offset = abs(F.relu(self.goal[0][0] - max_state) + F.relu(max_state - self.goal[0][1]))
        score_center_plus = abs(
            min(self.goal[0][1], max(self.goal[0][0], max_state)) - (self.goal[0][0] + self.goal[0][1]) / 2)
        score_center_plus = 24 - score_offset - 0.2 * score_center_plus

        if 0 < self.state.info_steps < 25:
            # 用餐时间消除波动惩罚
            r2 = 0.0
        else:
            temp_reward = self.vertex[0]
            r2 = - 1.0 * abs(self.state.get_value(temp_reward) - self.state.get_last_value(temp_reward))
        return self.reward_scale[0] * score_center_plus + self.reward_scale[1] * r2

    def step(self, action):
        """

        :param action: 注意是 list 格式，只有1个维度，表示干预node为何值
        :return:
        """
        info = dict()
        self.state.intervene(self.vertex[0], action[0])  # 干预2

        if 10 < self.state.info_steps < 16:
            self.state.increase(0, 10.0)  # 干预0

        self.state.calculate()
        observed_vals = self.state.sample_all()  # 干预完了后获得可观测node的value
        r = self.reward(self.state.graph.get_value(self.goal_vertex[0]))  # 使得 node 3 接近 10

        if self.state.info_steps == self.state.info_phase_length:  # 最后一步
            info["episode"] = dict()
            info["episode"]["r"] = self.ep_rew  # episode的reward
            info["episode"]["l"] = self.state.info_steps  # episode的长度
            self.ep_rew = 0.0
            done = True
        else:
            self.ep_rew = self.ep_rew + r  # 计算累计奖励
            done = False

        # concatenate all data that goes into an observation
        obs_tuple = (observed_vals, self.state.prev_action, self.state.prev_reward)
        obs = np.concatenate(obs_tuple)
        # step the environment state
        new_prev_action = np.array(action)
        self.state.step_state(new_prev_action, np.array([r]))
        # print("Step:", self.state.info_steps, "observed_vals:", observed_vals, "Reward:", r)
        return obs, r, done, info

    def log_callback(self):
        for k, v in self.log_data.items():
            self.logger.logkv(k, v)
        self.log_data = defaultdict(int)

    def reset(self):
        self.state.reset()
        observed_vals = self.state.sample_all()
        obs_tuple = (observed_vals, self.state.prev_action, self.state.prev_reward)
        obs = np.concatenate(obs_tuple)
        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=94566):
        np.random.seed(seed)


class EnvState(object):
    def __init__(self, info_phase_length=4):
        """Create an object which holds the state of a CBNEnv"""
        self.info_phase_length = info_phase_length

        self.info_steps = None  # 用来记录当前是第几步
        self.prev_action = None
        self.prev_reward = None
        self.graph = None

        self.reset()

    def step_state(self, new_prev_action, new_prev_reward):
        self.prev_action = new_prev_action
        self.prev_reward = new_prev_reward
        if self.info_steps == self.info_phase_length:  # 最后一步
            self.info_steps = 0
        else:
            self.info_steps += 1

    def intervene(self, node_idx, intervene_val):  # 干预
        self.graph.intervene(node_idx, intervene_val)

    def increase(self, node_idx, increase_val):  # 累加
        self.graph.increase(node_idx, increase_val)

    def calculate(self):
        self.graph.calculate()

    def sample_all(self):
        return self.graph.sample_all()

    def get_value(self, node_idx):
        return self.graph.get_value(node_idx)

    def get_last_value(self, node_idx):
        return self.graph.get_last_value(node_idx)

    def get_graph(self):
        raise NotImplementedError()

    def reset(self):
        self.info_steps = 0
        self.prev_action = np.zeros(1)
        self.prev_reward = np.zeros(1)
        self.graph = self.get_graph()
        print("EnvState reset:", self.graph.print_graph())


class TrainEnvState(EnvState):
    def get_graph(self):
        return CausalGraph(train=True)


class TestEnvState(EnvState):
    def get_graph(self):
        return CausalGraph(train=False)


class DebugEnvState(EnvState):
    def __init__(self):
        super().__init__()
        self.reward_data = None
