# VirtualPatient
VirtualPatient provides diabetics smulators, which not only be built based on real-data of an online healthcare application, Dnurse, but also provide interpretable dynamic transitions of a variety of patients. VirtualPatient is a near-realistic patient simulator, which can be helpful for testing decision support system, recommendation system and interpretable reasoning in real-world application, in which contains uncertainty, noise and user heterogeneity. Our research aims to foster the development of intelligent decision support studies for online applications, promoting the advancement of robust, personalized algorithms.

## Main Features
### Simulated Environment
VirtualPatient simulates the users' blood glucoses changes, under the intervene of meal size, insulin dosage, and exercise intensity.

1) We provide 30 virtual patients, modified from *the glucose-insulin system* [1], which is the open source part of the DMMS.R and T1DM simulators developed by The Epsilon
Group and has been certified by the US FDA.


## Installation
### Install using pip
Install the VirtualPatient package:
```
pip install -e .
```


## Usage for Supervised Learning

## Usage for Reinforcement Learning


## Acknowledgement
This project is an outcome of a joint work of [Web Intelligence Group](http://wi.hit.edu.cn/), Harbin Insititute Technology and [Dnurse: Prescription Digital Therapeutics Companies](https://www.dnurse.com/v2/en/).

# Reference
[1] Dalla Man C, Rizza R A, Cobelli C. Meal simulation model of the glucose-insulin system[J]. IEEE Transactions on biomedical engineering, 2007, 54(10): 1740-1749.