from collections import defaultdict
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
import copy
# from src.causal import CausalGraph
from virtual_patient.src_HER_Baseline_MLP_PPO.insulin_causal_x4_x8_simpler import CausalGraph
import torch
import torch.nn.functional as F

import operator
from functools import reduce

def one_hot(length, idx):
    one_hot = np.zeros(length)
    one_hot[idx] = 1
    return one_hot


class PreCBN(Env):
    def __init__(self,
                 agent_type='default',
                 info_phase_length=1440,
                 action_range=[-np.inf,np.inf],
                 train=True,
                 goal=[[70*1.886280201, 180*1.886280201]],
                 reward_scale=[1.0, 1.0],
                 vertex=[3],
                 last_vertex=[12],
                 patient_ID='adult#004'):
        """Create a stable_baselines-compatible environment to train policies on"""
        # vg = 1.886280201
        self.logger = None
        self.log_data = defaultdict(int)
        self.agent_type = agent_type
        self.ep_rew = 0
        self.vertex = vertex  # 这一次干预的顶点(list),即第几个顶点，为一个列表值
        self.last_vertex = last_vertex  # 上一次干预的顶点
        self.goal = goal # 最大最小目标值
        self.reward_scale = reward_scale  # 用来平衡 r1和r2
        self.patient_ID = patient_ID

        if train:
            self.state = TrainEnvState(self.vertex, self.last_vertex, info_phase_length, self.patient_ID)  # 一个继承了 EnvState class 的类
        else:
            self.state = TestEnvState(self.vertex, self.last_vertex, info_phase_length, self.patient_ID)
        self.action_space = Box(action_range[0], action_range[1], (len(self.vertex),), dtype=np.float64)
        # print("*************************",self.state.graph.len_obe + 2 * len(self.goal))
        self.observation_space = Box(-np.inf, np.inf, (self.state.graph.len_obe + 2 * len(self.goal),))  # 得到子图的节点数

    @classmethod
    def create(cls, goal, vertex, last_vertex, reward_scale, info_phase_length, action_range, n_env):
        return DummyVecEnv([lambda: cls(goal=goal, vertex=vertex, last_vertex=last_vertex, action_range=action_range, reward_scale = reward_scale,
                                        info_phase_length=info_phase_length)
                            for _ in range(n_env)])

    def reward(self, val, observed_valse):
        if len(val) == 1:
            max_state = torch.from_numpy(np.array(val))  # 转为tensor
            score_offset = abs(F.relu(self.goal[0][0] - max_state[0]) + F.relu(max_state[0] - self.goal[0][1]))
            score_center_plus = abs(
                min(self.goal[0][1], max(self.goal[0][0], max_state[0])) - (self.goal[0][0] + self.goal[0][1]) / 2)
            score_center_plus = 24 - score_offset - 0.2 * score_center_plus

        elif len(val) == 2:
            max_state = torch.from_numpy(np.array(val))
            score_offset = abs(F.relu(self.goal[0][0] - max_state[0]) + F.relu(max_state[0] - self.goal[0][1]))
            score_center_plus = abs(
                min(self.goal[0][1], max(self.goal[0][0], max_state[0])) - (self.goal[0][0] + self.goal[0][1]) / 2)
            score_center_plus = 24 - score_offset - 0.2 * score_center_plus
            # 第一个被干预节点的reward
            score_offset_ = abs(F.relu(self.goal[1][0] - max_state[1]) + F.relu(max_state[1] - self.goal[1][1]))
            score_center_plus_ = abs(
                min(self.goal[1][1], max(self.goal[1][0], max_state[1])) - (self.goal[1][0] + self.goal[1][1]) / 2)
            score_center_plus_ = 24 - score_offset_ - 0.2 * score_center_plus_
            # 第二个被干预节点的reward
            score_center_plus += score_center_plus_  # 两者相加

        if (self.state.info_steps >= 360 and self.state.info_steps <= 420) \
                or (self.state.info_steps >= 660 and self.state.info_steps <= 720) \
                or (self.state.info_steps >= 1080 and self.state.info_steps <= 1140):
            # 用餐时间消除波动惩罚
            r2 = 0.0
        else:
            temp_reward = 0
            if 4 in self.vertex and 8 in self.vertex:
                temp_reward = 4
            elif (7 in self.vertex and 4 in self.vertex):
                temp_reward = 7
            elif (6 in self.vertex):
                temp_reward = 6
            else:
                temp_reward = self.vertex[0]
            r2 = - 1.0 * abs(self.state.get_value(temp_reward) - self.state.get_last_value(temp_reward))

        return self.reward_scale[0]*score_center_plus + self.reward_scale[1] * r2

    def step(self, action):
        """
        :param action: 注意是 list 格式，只有1个维度，表示干预node为何值
        :return:
        """
        info = dict()
        self.state.intervene(action)  # action是一个列表
        observed_vals, now_insulin = self.state.sample_all()  # 干预完了后获得可观测node的value

        r = self.reward(now_insulin, observed_vals).numpy()  # 使得 insulin 在 70 到 180 范围内
        # print("Step:", self.state.info_steps, "Action:", action, "Now_insulin:", now_insulin, "Reward:", r, 'state:', observed_vals)
        # print('\n')

        if self.state.info_steps == self.state.info_phase_length:  # 最后一步
            self.ep_rew = self.ep_rew + r  # 计算累计奖励
            info["episode"] = dict()
            info["episode"]["r"] = self.ep_rew  # episode的reward
            info["episode"]["l"] = self.state.info_steps  # episode的长度
            # print("ep_rew：", self.ep_rew)
            self.ep_rew = 0.0
            done = True
        else:
            self.ep_rew = self.ep_rew + r  # 计算累计奖励
            done = False

        # concatenate all data that goes into an observation
        obs_tuple = np.hstack([observed_vals, reduce(operator.add, self.goal)])
        ####################################################### 这里考虑 1）observed_vals； 2）observed_vals, self.state.prev_action, self.state.prev_reward，再加上上一时刻状态
        obs = obs_tuple
        # step the environment state
        # new_prev_action = one_hot(8, action)
        new_prev_action = np.array(action)
        # print("Step:", self.state.info_steps, "Reward:", reward, "obs:", obs)
        self.state.step_state(new_prev_action, np.array([r]),obs)
        return obs, r, done, info

    def log_callback(self):
        for k, v in self.log_data.items():
            self.logger.logkv(k, v)
        self.log_data = defaultdict(int)

    def reset(self):
        self.state.reset()
        observed_vals, now_insulin = self.state.sample_all()
        obs_tuple = np.hstack((observed_vals, reduce(operator.add, self.goal)))
        obs = obs_tuple
        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=94566):
        np.random.seed(seed)


class EnvState(object):
    def __init__(self, vertex, last_vertex, info_phase_length=50, patient_ID='adult#004'):
        """Create an object which holds the state of a CBNEnv"""
        self.info_phase_length = info_phase_length
        self.info_steps = None  # 用来记录当前是第几步
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.graph = None
        self.vertex = vertex
        self.last_vertex = last_vertex
        self.patient = patient_ID

        self.reset()

    def step_state(self, new_prev_action, new_prev_reward, new_prev_state):
        self.prev_action = new_prev_action
        self.prev_reward = new_prev_reward
        self.prev_state = copy.deepcopy(new_prev_state)
        if self.info_steps == self.info_phase_length:  # 最后一步
            self.info_steps = 0
        else:
            self.info_steps += 1

    def intervene(self, intervene_val):  # 干预
        # print("intervene_val:",intervene_val)
        self.graph.intervene(intervene_val)

    def sample_all(self):
        # return self.graph.sample_all()[:-1]   # 这里应该是设置最后一个node不可见
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
        self.prev_state, _ = self.sample_all()


class TrainEnvState(EnvState):
    def get_graph(self):
        return CausalGraph(train=True, vertex=self.vertex, last_vertex=self.last_vertex, patient_ID=self.patient_ID)


class TestEnvState(EnvState):
    def get_graph(self):
        return CausalGraph(train=False, vertex=self.vertex, last_vertex=self.last_vertex, patient_ID=self.patient_ID)


class DebugEnvState(EnvState):
    def __init__(self):
        super().__init__()
        self.reward_data = None
