# VirtualPatient
VirtualPatient provides diabetics smulators, which not only be built based on real-data of an online healthcare application, Dnurse, but also provide interpretable dynamic transitions of a variety of patients. VirtualPatient is a near-realistic patient simulator, which can be helpful for testing decision support system, recommendation system and interpretable reasoning in real-world application, in which contains uncertainty, noise and user heterogeneity. Our research aims to foster the development of intelligent decision support studies for online applications, promoting the advancement of robust, personalized algorithms.

## Main Features
### Simulated Environment
VirtualPatient simulates the users' blood glucoses changes, under the intervene of meal size, insulin dosage, and exercise intensity. 

Utilizing the VirtualPatient simulator, one can access a "live" environment just like the real online healthcare environment. This simulator generates virtual patients individually, each beginning with an initial physiological metric, such as blood glucose levels. Algorithms are then tasked with formulating treatment recommendations, encompassing aspects like meal size, insulin dosage, and exercise intensity. Following the implementation of these treatments, the virtual patient provides feedback, indicating how their physiological metrics have responded, akin to the responses one would expect from real-life patients. This feedback loop simulates real-world patient responses, offering valuable insights into the effectiveness of treatment strategies.

1) We provide 30 virtual patients, modified from *the glucose-insulin system* [1], which is the open source part of the DMMS.R and T1DM simulators developed by The Epsilon Group and has been certified by the US FDA. The glucose-insulin system is an existing dynamic model of glucose ingestion and absorption, which is obtained through an extremely challenging triple tracer meal protocol by tracking the glucose conversion dynamic in the meals of 204 normal individuals. 

2) Besides, we provide 9 real-wrold patients trained through dataset from Dnurse. How these nine patients were trained is described in:

3) Each patient (both 30 virtual patients and 9 real-world patients) is associated with 61 static attributes. Here, static/dynmaic means whether the features will change during an interactive process. The attributes information about involve patient age, patient gender, patient , etc.

4) We provide 

## Installation
### Install using pip
Install the VirtualPatient package:
```
pip install -e .
```

## Usage for Supervised Learning

## Usage for Reinforcement Learning
How create a virtual patient?
```
from virtual_patient.envs.env_do_all_her_2reward import CBNEnv
from stable_baselines3 import PPO
env = CBNEnv.create(
        info_phase_length=1440,
        action_range=[0, 3000],
        vertex=[10],
        reward_scale=[1.0, 1.0],
        n_env=1,
        patient_ID=patient_ID
    )
optimizer_kwargs = dict(
        alpha=0.95,
    )
policy_kwargs = dict(
        optimizer_kwargs=optimizer_kwargs,
        optimizer_class=RMSpropTFLike
    )
model = PPO("MlpPolicy",
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
model.learn(
    total_timesteps=int(3e7),
    callback=TensorboardCallback())

env.close()
```



## Acknowledgement
This project is an outcome of a joint work of [Web Intelligence Group](http://wi.hit.edu.cn/), Harbin Insititute Technology and [Dnurse: Prescription Digital Therapeutics Companies](https://www.dnurse.com/v2/en/).

# Reference
[1] Dalla Man C, Rizza R A, Cobelli C. Meal simulation model of the glucose-insulin system[J]. IEEE Transactions on biomedical engineering, 2007, 54(10): 1740-1749.