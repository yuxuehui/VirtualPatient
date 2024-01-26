import argparse
import numpy as np
# task1 : 01_train_2reward_PPO
# task2 : 01_train2_5nodes
# task3 : 01_train2_hill_regulation_5nodes
# task4 : 01_train2_level_contain1
from virtual_patient.envs.env_do_all_her_2reward import CBNEnv as CBNEnv_1
from Baseline_graph_direct_hill_regulation.env_5nodes import CBNEnv as CBNEnv_2
from Baseline_graph_direct_hill_regulation.env_hill_regulation_5nodes import CBNEnv as CBNEnv_3
from Baseline_graph_direct_hill_regulation_level_100.env_5nodes_level import CBNEnv as CBNEnv_4


def get_env(task_name, args):
    if task_name == "1":
        env = CBNEnv_1.create(
            info_phase_length=2,
            action_range=args.action_range,
            vertex=args.vertex,
            # reward_weight=args.reward_weight,
            reward_scale=args.reward_scale,
            n_env=1,
            patient_ID=args.patient_ID,
            list_last_vertex=[],
            flag=args.flag,
            meal_time=args.meal_time,
            default_meal=args.default_meal,
            default_insulin=args.default_insulin
        )
    if task_name == "2":
        env = CBNEnv_2.create(
            info_phase_length=100,
            # action_range = args.action_range,
            # vertex = args.vertex,
            reward_scale=args.reward_scale,
            # list_last_vertex = [
            #   # {  # 前1阶段
            #   #     "vertex": [4, 8],
            #   #     "dir": father_path + "/Model_CHO2_add/her_low_action_48_goal_3_mlp_range_0_600_2.zip",
            #   #     "info_phase_length": 1440,
            #   #     "action_range": [0, 600],
            #   #     "reward_scale": [1.0, 1.0],
            #   #     "last_vertex": [3]},
            #   # {  # 前2阶段
            #   #     "vertex": [3],
            #   #     "dir": father_path + "/Model/her_low_action_3_goal_12_2reward3_copy2.zip",
            #   #     "info_phase_length": 1440,
            #   #     "action_range": [-np.inf,
            #   #                      np.inf],
            #   #     "reward_scale": [1.0, 1.0],
            #   #     "last_vertex": [12]}
            #   ],
            n_env=1)
    if task_name == "3":
        env = CBNEnv_3.create(
            info_phase_length=100,
            # action_range = args.action_range,
            # vertex = args.vertex,
            reward_scale=args.reward_scale,
            # list_last_vertex = [
            #   # {  # 前1阶段
            #   #     "vertex": [4, 8],
            #   #     "dir": father_path + "/Model_CHO2_add/her_low_action_48_goal_3_mlp_range_0_600_2.zip",
            #   #     "info_phase_length": 1440,
            #   #     "action_range": [0, 600],
            #   #     "reward_scale": [1.0, 1.0],
            #   #     "last_vertex": [3]},
            #   # {  # 前2阶段
            #   #     "vertex": [3],
            #   #     "dir": father_path + "/Model/her_low_action_3_goal_12_2reward3_copy2.zip",
            #   #     "info_phase_length": 1440,
            #   #     "action_range": [-np.inf,
            #   #                      np.inf],
            #   #     "reward_scale": [1.0, 1.0],
            #   #     "last_vertex": [12]}
            #   ],
            n_env=1
        )
    if task_name == "4":
        env = CBNEnv_4.create(
            info_phase_length=100,
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
    return env


if __name__ == "__main__":

    task_name = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("-reward_scale", nargs='+', type=float, help=" ", default=[1.0, 1.0])
    if task_name == "1":
        parser.add_argument("-reward_scale", nargs='+', type=float, help=" ", default=[1.0, 1.0])
        parser.add_argument("-vertex", nargs='+', type=int, help="Now inserve", default=[10])
        parser.add_argument("-patient_ID", type=str, help="Patient ID", default='adult#004')
        parser.add_argument("-flag", type=int, help="flag", default=1)  # flag
        parser.add_argument("-meal_time", nargs='+', type=int, help="meal_time", default=[300, 600, 1000])  # meal_time
        parser.add_argument("-default_meal", nargs='+', type=int, help="default_meal",
                            default=[50, 100, 100])  # default_meal
        parser.add_argument("-default_insulin", type=int, help="default_insulin", default=0.0)  # default_insulin
        # parser.add_argument("-action_range", nargs='+', type=int, help=" ", default=[0, 3000])  # flag=0
        parser.add_argument("-action_range", nargs='+', type=int, help=" ", default=[0, 300])  # flag=1
        # parser.add_argument("-action_range", nargs='+', type=int, help=" ", default=[np.array([0,0]),np.array([3000,300])])  #flag=2
    if task_name == "2":
        parser.add_argument("-vertex", nargs='+', type=int, help="Now inserve", default=[4])
        parser.add_argument("-action_range", nargs='+', type=int, help=" ", default=[0, np.inf])  # 如果步输入的话就是np.inf
    if task_name == "3":
        parser.add_argument("-vertex", nargs='+', type=int, help="Now inserve", default=[4])
        parser.add_argument("-action_range", nargs='+', type=int, help=" ", default=[0, np.inf])  # 如果步输入的话就是np.inf
    if task_name == "4":
        parser.add_argument("-vertex", nargs='+', type=int, help="Now inserve", default=[5])
        parser.add_argument("-action_range", nargs='+', type=int, help=" ", default=[0, np.inf])  # 如果步输入的话就是np.inf

    args = parser.parse_args()
    env = get_env("1", args)
