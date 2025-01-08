![VirtualPatients](img/virtualpatient_logo.png)
![Teaser](img/virtualpatient_sim.png)

# VirtualPatients

<!-- ## 🔥 News

- [2024-12-25] Opensourced offline dataset
- [2024-01-26] Release VritualPatients V1.0 -->

VirtualPatients is a physics platform that enables users to design and test treatment protocols for in-silico subjects with Diabetes Mellitus. It is simultaneously multiple things:

1. A **lightweight**, **Pythonic**, and **user-friendly** patient simulation platform.
2.  A universal physics engine capable of simulating a wide range of individual differences and biophysiological phenomena.
3.  A tool to simulate and test treatment protocols identical to proposed clinical studies.
4.  A system to measure the impact on diabetes management and treatment.

## Overview

### Key Features

- 🐍 100% Python, both front-end interface and back-end physics engine, all natively developed in python.
- 📐 Differentiability: VirtualPatients is designed to be fully compatible with differentiable simulation. 
- 💥 A unified framework that supports various state-of-the-art physics solvers, modeling a vast range of individual differences and biophysiological phenomena.

### IN Silico Population
- Embodies the biophysiological parameters of the FDA accepted in silico population
- 19 adults, 10 adolescents, 10 children

### Basic User-Defined Simulation Input
- Meal profiles (CHO amount, timing and duration of a meal)
- Insulin treatment (amount and timing of basal/bolus insulin doses)
- Time of simulation and regulation (length & time of day)

<img src="img/bg_level.png" alt="drawing" width="400" class="center"/>

### Subject-Specific Data to Fine-Tune Treatment
- Individualised, intra-personal results 
- Inter-personal differences are highlighted across the spectrum of human variation  
- Population results are calculated from individual results, as in a clinical trial  
- Age, Body Weight (kg), subject-specific optimal basal insulin rate (u/hr), individual carbohydrate ratio (CR, g/U), total daily insulin, and insulin sensitivity (maximum drop in glucose mg/dl per unit of insulin [MD])  
- Metabolic testing results may be simulated for individual subjects and incorporated into treatment plans beforehand  

### Simulation Results Data (Per Subject and Population)
- Blood glucose (BG) values and simulated sensor readings (mg/dl per minute)
- Basal/bolus insulin injections (pmol/minute)
- User-specified data from controller
- System states, carbohydrate intake and more ...


## Differentiable Simulation Process

The operation of the VirtualPatients simulation is shown in the figure below:

![differential](img/virtualpatient_differential.png)

The difference $\Delta$ is used to update the patient’s state $\textbf{x}$, calculated using the CGD step as shown in the following equation:

$$
\Delta_0 = -\textcolor[rgb]{0.27,0.4,0.37}{k_{max}} \cdot x_0 + CHO \cdot 1000
$$

$$
\textcolor[rgb]{0.27,0.4,0.37}{kgut} \in [\textcolor[rgb]{0.5,0.32,0.43}{k_{min}}, \textcolor[rgb]{0.5,0.32,0.43}{k_{max}}]
$$

$$
\Delta_1 = \textcolor[rgb]{0.5,0.32,0.43}{k_{max}} \cdot x_0 - \textcolor[rgb]{0.27,0.4,0.37}{kgut} \cdot x_1
$$

$$
\Delta_2 = \textcolor[rgb]{0.27,0.4,0.37}{kgut} \cdot x_1 - \textcolor[rgb]{0.5,0.32,0.43}{kabs} \cdot x_2
$$

$$
\Delta_3 = \max(\textcolor[rgb]{0.5,0.32,0.43}{kp_1} - \textcolor[rgb]{0.5,0.32,0.43}{kp_2} \cdot x_3 - \textcolor[rgb]{0.5,0.32,0.43}{kp_3} \cdot x_8, 0) - 1 + \textcolor[rgb]{0.5,0.32,0.43}{\frac{f \cdot kabs}{BW}} \cdot x_2 - \textcolor[rgb]{0.5,0.32,0.43}{ke_1} \cdot \text{Relu}(x_3 - \textcolor[rgb]{0.5,0.32,0.43}{ke_2}) - \textcolor[rgb]{0.5,0.32,0.43}{k_1} \cdot x_3 + \textcolor[rgb]{0.5,0.32,0.43}{k_2} \cdot x_4
$$

$$
\Delta_4 = -\frac{\textcolor[rgb]{0.5,0.32,0.43}{Vm_0} + \textcolor[rgb]{0.5,0.32,0.43}{Vm_x} \cdot x_6 \cdot x_4}{x_4 + \textcolor[rgb]{0.5,0.32,0.43}{Km_0}} + \textcolor[rgb]{0.5,0.32,0.43}{k_1} \cdot x_3 - \textcolor[rgb]{0.5,0.32,0.43}{k_2} \cdot x_4
$$

$$
\Delta_5 = -\textcolor[rgb]{0.5,0.32,0.43}{m_2 + m_4} \cdot x_5 + \textcolor[rgb]{0.5,0.32,0.43}{m_1} \cdot x_9 + \textcolor[rgb]{0.5,0.32,0.43}{ka_1} \cdot x_{10} + \textcolor[rgb]{0.5,0.32,0.43}{ka_2} \cdot x_{11}
$$

$$
\Delta_6 = -\textcolor[rgb]{0.5,0.32,0.43}{p2u} \cdot x_6 + \textcolor[rgb]{0.5,0.32,0.43}{p2u} \cdot \left( \frac{x_5}{\textcolor[rgb]{0.5,0.32,0.43}{Vi}} - \textcolor[rgb]{0.5,0.32,0.43}{Ib} \right)
$$

$$
\Delta_7 = -\textcolor[rgb]{0.5,0.32,0.43}{ki} \cdot \left( \frac{x_7 - x_5}{\textcolor[rgb]{0.5,0.32,0.43}{Vi}} \right)
$$

$$
\Delta_8 = -\textcolor[rgb]{0.5,0.32,0.43}{ki} \cdot (x_8 - x_7)
$$

$$
\Delta_9 = -\textcolor[rgb]{0.5,0.32,0.43}{m_1 + m_{30}} \cdot x_9 + \textcolor[rgb]{0.5,0.32,0.43}{m_2} \cdot x_5
$$

$$
\Delta_{11} = \textcolor[rgb]{0.5,0.32,0.43}{kd} \cdot x_{10} - \textcolor[rgb]{0.5,0.32,0.43}{ka_2} \cdot x_{11}
$$

$$
\Delta_{12} = -\textcolor[rgb]{0.5,0.32,0.43}{ksc} \cdot x_{12} + \textcolor[rgb]{0.5,0.32,0.43}{ksc} \cdot x_3
$$

The $\textcolor[rgb]{0.5,0.32,0.43}{\text{purple variables}}$ correspond to the personalised learnable simulation parameters $\textbf{u}$. The feature variables $\textbf{u}$ for 39 patients were trained on the patients' offline data and are stored in the file [vpatient_params.csv](https://github.com/yuxuehui/VirtualPatient/blob/main/Data/vpatient_params.csv).

<!-- ### Simulated Environment
VirtualPatients simulates the users' blood glucoses changes, under the intervene of meal size, insulin dosage, and exercise intensity. 

Utilizing the VirtualPatients simulator, one can access a "live" environment just like the real online healthcare environment. This simulator generates virtual patients individually, each beginning with an initial physiological metric, such as blood glucose levels. Algorithms are then tasked with formulating treatment recommendations, encompassing aspects like meal size, insulin dosage, and exercise intensity. Following the implementation of these treatments, the virtual patient provides feedback, indicating how their physiological metrics have responded, akin to the responses one would expect from real-life patients. This feedback loop simulates real-world patient responses, offering valuable insights into the effectiveness of treatment strategies.

1) We provide 30 virtual patients, modified from *the glucose-insulin system* [1], which is the open source part of the DMMS.R and T1DM simulators developed by The Epsilon Group and has been certified by the US FDA. The glucose-insulin system is an existing dynamic model of glucose ingestion and absorption, which is obtained through an extremely challenging triple tracer meal protocol by tracking the glucose conversion dynamic in the meals of 204 normal individuals. 

2) Besides, we provide 9 real-wrold patients trained through dataset from Dnurse. How these nine patients were trained is described in:

3) Each patient (both 30 virtual patients and 9 real-world patients) is associated with 61 static attributes. Here, static/dynmaic means whether the features will change during an interactive process. The attributes information about involve patient age, patient gender, patient , etc.

4) We provide  -->

## How to Use VirtualPatients?
### Quick Installation
For the latest version, clone the repository and install locally:
```bash
git clone https://github.com/yuxuehui/VirtualPatient.git
cd VirtualPatients
pip install -e .
```

## Designing healthcare interventions through a decision algorithm

### VirtualPatients Intervention Schemes

**VirtualPatients** provides three intervention schemes to meet the personalised needs of different patients:  
1) **Dietary intervention**  
2) **Medication intervention**  
3) **Combined dietary and medication intervention**  

(An interface for exercise intervention will be made available soon.)  

The intervention scheme is configured using five parameters: `info_phase_length`, `flag`, `meal_time`, `default_meal`, and `default_insulin`. The details of each parameter are as follows:  

- **`flag`**: Determines the control mode with a value of 0, 1, or 2.  
  - `0`: Controls carbohydrates only.  
  - `1`: Controls insulin only.  
  - `2`: Controls both carbohydrates and insulin.  

- **`info_phase_length`**: the number of time steps per day.  
  - `3` when `flag=0`, three meals per day.  
  - `1440` for other `flag` values. Basal/bolus insulin injections are once per minute, so 1440 insulin doses are required in a natural day. 

- **`meal_time`**: Specifies meal times in a list with three elements (in minutes). Default value: `[360, 660, 1080]`  

- **`default_meal`**: Specifies the default meal size, used when `flag=0`. Formatted as a list with three elements. Default value: `[50, 50, 50]`  

- **`default_insulin`**: Specifies the default insulin dosage, used when `flag=1`. Formatted as a float. Default value: `0.01`  

Additionally, do not forget change the `action_range` according to different intervention schemes and the need of your control methods, which is used to specify the range of values for the action space.

You can use VirtualPatients to evaluate various types of control methods, such as supervised learning and reinforcement learning. Below is an example of a Python implementation using PPO:

```python
from virtual_patient.envs.env_do_all_her_2reward import CBNEnv
from stable_baselines3 import PPO
env = CBNEnv.create(    # initialise a environment
        info_phase_length=1440,
        action_range=args.action_range,
        vertex=args.vertex,
        reward_scale=args.reward_scale,
        n_env=1,
        patient_ID=args.patient_ID,
        list_last_vertex=[],
        flag=0,
        meal_time=[300,600,1000],
        default_meal=[50,100,100],
        default_insulin=0.02
    )
optimizer_kwargs = dict(
        alpha=0.95,
    )
policy_kwargs = dict(
        optimizer_kwargs=optimizer_kwargs,
        optimizer_class=RMSpropTFLike
    )

model = PPO("MlpPolicy", # initialise a PPO model
            env,
            n_steps=5,
            gae_lambda=0.95,
            gamma=0.9,
            n_epochs=10,
            ent_coef=0.0,
            learning_rate=float(1e-4),
            clip_range=0.2,
            use_sde=True,
            sde_sample_freq=4,
            verbose=1,
            tensorboard_log='../logs',
            seed=94566)

model.learn(    # train
    total_timesteps=int(3e7),
    callback=TensorboardCallback())

env.close()
```


# Associated Papers
VirtualPatients is the result of a collaborative effort between the [Web Intelligence Group](http://wi.hit.edu.cn/) at Harbin Institute of Technology and [Dnurse](https://www.dnurse.com/v2/en/). It is a large-scale effort that integrates various existing and ongoing research projects into a single system. Below is a list of papers that have contributed to the VirtualPatients project:

- Yu, Xuehui, et al. "Causal prompting model-based offline reinforcement learning." arXiv preprint arXiv:2406.01065 (2024).
- Yu, Xuehui, et al. "KaDGT: How to Survive in Online Personalisation with Highly Low-quality Offline Datasets." arXiv preprint arXiv:2335.20405 (2024).
- Liu, Liangliang, et al. "An interactive food recommendation system using reinforcement learning." Expert Systems with Applications (2024): 124313.
- Yu, Xuehui, et al. "ARLPE: A meta reinforcement learning framework for glucose regulation in type 1 diabetics." Expert Systems with Applications 228 (2023): 120156.
- Yu, Xuehui, et al. "Causal Coupled Mechanisms: A Control Method with Cooperation and Competition for Complex System." 2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2022.

# Reference
[1] Dalla Man C, Rizza R A, Cobelli C. Meal simulation model of the glucose-insulin system[J]. IEEE Transactions on biomedical engineering, 2007, 54(10): 1740-1749.