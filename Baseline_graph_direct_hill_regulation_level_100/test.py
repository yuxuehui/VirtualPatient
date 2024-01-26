"""
    测试 x3-x12
    运行实例：python ./test2.py -vertex 3 -reward_weight 1.0 -model_dir "test_x3_x12_reward_copy"
"""
import sys
import os
# 获取当前文件路径
import torch
import pickle
import copy
import argparse
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
print(father_path)
sys.path.append(father_path)    # /home/yuxuehui/yxhfile/causal-metarl-master/src
father_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
sys.path.append(father_path)    # /home/yuxuehui/yxhfile/causal-metarl-master
print(father_path)

import numpy as np
import matplotlib.pyplot as plt
# from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from Baseline_graph_direct_hill_regulation_level_100.env_5nodes_level import CBNEnv
from src.a2c import A2C


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        # print("**********************self.locals", self.locals)
        self.logger.record('random_value', value)
        return True


def train(args):
    env = CBNEnv.create(
        info_phase_length=60,
        action_range=args.action_range,
        vertex=args.vertex,
        reward_scale=args.reward_scale,
        list_last_vertex=[
            # {  # 前1阶段
            #     "vertex": [2],
            #     "dir": father_path + "/Model_Baseline_DDE/Baseline_hill_8nodes_x2_x1_1.zip",
            #     "info_phase_length": 60,
            #     "action_range": [0, np.inf],
            #     "reward_scale": [1.0, 1.0],
            #     "last_vertex": [1]},
            # {  # 前2阶段
            #     "vertex": [3],
            #     "dir": father_path + "/Model/her_low_action_3_goal_12_2reward3_copy2.zip",
            #     "info_phase_length": 1440,
            #     "action_range": [-np.inf,
            #                      np.inf],
            #     "reward_scale": [1.0, 1.0],
            #     "last_vertex": [12]}
        ],
        n_env=1
    )

    current_path = os.path.abspath(__file__)
    father_path1 = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    father_path = os.path.abspath(os.path.dirname(father_path1) + os.path.sep + ".")
    path = os.path.join(
        father_path,
        'Model_Baseline_DDE',
        args.model_dir,
    )



    model = A2C.load(path, env=env)

    ### 测试，统计Action
    obs = env.envs[0].reset()
    obs_list = []
    action_list = []
    low_ep_reward = 0.0
    for j in range(60):
        action, _states = model.predict(obs)
        if j == 0:
            for i in range(len(action)):action_list.append([])
        for i in range(len(action)):action_list[i].append(action[i])
        # for item in range(len(action)):
        #     action[item] = action[item] * 10.0
        obs, rewards, dones, info = env.envs[0].step(action)
        print("Action:", action, obs, rewards)
        obs_list.append(copy.deepcopy(obs))
        low_ep_reward = low_ep_reward + rewards
    return_action = []
    for j in range(len(action_list)):
        action_list[j].sort()
        return_action.append([action_list[j][6], action_list[j][54]])

    print("x3取值范围：", return_action)
    obs_list = np.array(obs_list)
    obs_list = obs_list.T
    x = range(60)
    plt.figure()
    plt.plot(x, obs_list[0], label="x0")
    plt.plot(x, obs_list[1], label="x1", linestyle="--")
    plt.plot(x, obs_list[2], label="x2", linestyle="--")
    plt.legend(loc='upper left')
    plt.savefig('./test_hill.jpg')

    # ### 测试，统计Action
    # obs = env.envs[0].reset()
    # obs_list = []
    # action_list = []
    # low_ep_reward = 0.0
    # for j in range(100):
    #     obs = [10, 10, 10, 10, 10, 10, 10, 10, 10]
    #     action, _states = model.predict(obs)
    #     if j == 0:
    #         for i in range(len(action)): action_list.append([])
    #     for i in range(len(action)): action_list[i].append(action[i])
    #     # obs, rewards, dones, info = env.envs[0].step(action)
    #     # print(obs, rewards)
    #     obs_list.append(copy.deepcopy(action))
    #     # low_ep_reward = low_ep_reward + rewards
    # return_action = []
    # for j in range(len(action_list)):
    #     action_list[j].sort()
    #     return_action.append([action_list[j][10], action_list[j][90]])
    #
    # print("x3取值范围：", return_action)
    # obs_list = np.array(obs_list)
    # obs_list = obs_list.T
    # x = range(len(obs_list[0]))
    # plt.plot(x, obs_list[0], label="action")
    # # plt.plot(x, obs_list[1], label="x12", linestyle="--")
    # plt.legend(loc='upper left')
    # plt.savefig('./test3.jpg')
    # # plt.show()

if __name__ == "__main__":
    # 创建一个参数解析实例
    parser = argparse.ArgumentParser()
    # 添加参数解析
    parser.add_argument("-vertex", nargs='+', type=int, help="Now inserve")
    parser.add_argument("-action_range", nargs='+', type=int, help=" ", default=[0, np.inf])  # 如果步输入的话就是np.inf
    parser.add_argument("-reward_weight", type=float, help=" ", default=1.0)  # 如果步输入的话就是1.0
    parser.add_argument("-model_dir", type=str, help="Save dir")
    parser.add_argument("-reward_scale", nargs='+', type=float, help=" ", default=[1.0, 1.0])
    parser.add_argument("-n_step", type=int, help="n_step", default=5)
    parser.add_argument("-seed", type=int, help="seed", default=94566)
    parser.add_argument("-max_grad_norm", type=float, help="seed", default=10000)
    parser.add_argument("-total_timesteps", type=int, help="seed", default=int(3e7))
    # 开始解析
    args = parser.parse_args()
    train(args)
