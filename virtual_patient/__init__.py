from gym.envs.registration import register

register(
    id='virtual_patient/CBNEnv-v0',
    entry_point='virtual_patient.envs:CBNEnv',
)