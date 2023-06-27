
<p align="center">
    <img width="100" src="/Assets/rlai.png" />
</p>
<br>
<h2 align=center>LC-ETD: Loosely Consistent Emphatic Temporal-Difference Learning</h2>

This repository includes the code for [this paper](https://openreview.net/forum?id=Jx7LBaXM3M).
<br>

## Table of Contents
- **[Specification of Dependencies](#specifications)**
- **[Algorithms](#algorithms)**
    - **TD**: [Off-policy TD](#td)
    - **Emphatic-TD family**   : [Emphatic TD](#etd), [Emphatic TDβ](#etdb)  
    - **LC-ETD family** : [LC-ETD1](#lcetd1), [LC-ETD2](#lcetd2), [LC-ETD3](#lcetd3)
    - **[Algorithm Glossary](#glossary)**
- **[Environments](#environment)** : [Baird's counterexample](#baird), [Two-state environment](#two_state), [Four Room Grid World](#four_room_grid_world)
- **[How to run the code](#how-to-run)**: [Learning.py](#learning.py), [Job Buidler](#job_builder)

<a name='specifications'></a>
## Specification of Dependencies
This code requires python 3.5 or above. Packages that are required for running the code are all in the `requirements.txt`
file. To install these dependencies, run the following command if your pip is set to `python3.x`:
```text
pip install requirements.txt
```
otherwise, run:
```text
pip3 install requirements.txt
```




<a name='algorithms'></a>
## Algorithms

<a name='td'></a>
### Off-policy TD

**Paper** [Off-Policy Temporal-Difference Learning with Function Approximation](
https://www.cs.mcgill.ca/~dprecup/publications/PSD-01.pdf)<br>
**Authors** Doina Precup, Richard S. Sutton, Sanjoy Dasgupta<br>

<a name='full_is_td'></a>
### Full-IS-TD

**Paper** [Eligibility Traces for Off-Policy Policy Evaluation]()<br>
**Authors** Doina Precup, Richard S. Sutton, Satinder Singh<br>

### Emphatic-TD algorithms

<a name='etd'></a>
#### Emphatic TD

**Paper** [An Emphatic Approach to the Problem of Off-policy Temporal-Difference Learning](
https://jmlr.org/papers/volume17/14-488/14-488.pdf)<br>
**Authors** Richard S. Sutton, A. Rupam Mahmood, Martha White<br>

<a name='etdb'></a>
#### Emphatic TDβ

**Paper** [Generalized Emphatic Temporal Difference Learning: Bias-Variance Analysis](
https://ojs.aaai.org/index.php/AAAI/article/view/10227/10086)<br>
**Authors** Assaf Hallak, Aviv Tamar, Remi Munos, Shie Mannor<br>

### LC-ETD algorithms

**Paper** Loosely Consistent Emphatic Temporal-Difference Learning<br>
**Authors** Jiamin He, Fengdi Che, Yi Wan, A. Rupam Mahmood<br>
**Algorithms** LC-ETD1, LC-ETD2, LC-ETD3

<a name='lcetd1'></a>

<a name='lcetd2'></a>

<a name='lcetd3'></a>


<a name='glossary'></a>
### Algorithm Glossary
Here, we briefly explain all the symbols and variables names that we use in our implementation.

#### meta-parameters
- Common parameters of all algorithms:
  - alpha (α): is the step size that defines how much the weight vector [**w**](#var_w) is updated at each time step.
  - lambda (λ): is the bootstrapping parameter.
- beta (β): is the parameter used by the [**ETDβ**](#etdb) algorithm that defines how much the product of importance sampling ratios
from the past affects the current update.

#### Algorithms variables
<a name='var_w'></a>
- **w**: is the main weight vector being learned. ```init: w=0```.
<a name='var_v'></a>
- **v**: is the secondary weight vector learned by Gradient-TD algorithms.  ```init: v=0```.
<a name='var_z'></a>
- **z**: is the eligibility trace vector.  ```init: z=0```.
<a name='var_delta'></a>
- delta (𝛿): is the td-error, which in the full bootstrapping case, is equal to the reward plus the value of the next 
  state minus the value of the current state.
<a name='var_s'></a>
- s: is the current state (scalar).
<a name='var_x'></a>
- **x**: is the feature vector of the current state.
<a name='var_s_p'></a>
- s_p: is the next state (scalar).
<a name='var_x_p'></a>
- **x_p**: is the feature vector of the next state. 
<a name='var_r'></a>
- r: is the reward.
<a name='var_rho'></a>
- rho (ρ): is the importance sampling ratio, which is equal to the probability of taking an action under the target policy
  divided by the probability of taking the same action under the behavior policy.
<a name='var_oldrho'></a>
- old_rho (oldρ): is the importance sampling ratio at the previous time step.
<a name='var_pi'></a>
- pi (π): is the probability of taking an action under the target policy at the current time step.
<a name='var_oldpi'></a>
- old_pi (oldπ): is the probability of taking an action under the target policy in the previous time step. The variable
  π itself is the probability of taking action under the target policy at the current time step.
<a name='var_F'></a>
- F : is the follow-on trace used by [Emphatic-TD](#etd) algorithms.
<a name='var_m'></a>
- m : is the emphasis used by [Emphatic-TD](#etd) algorithms.
<a name='var_mu'></a>
- mu (μ): is the probability of taking action under the behavior policy at the current time step.
<a name='var_oldmu'></a>
- old_mu (oldμ): is the probability of taking an action under the target policy at the previous time step.
- gamma (γ): is the discount factor parameter.

## How to Run the Code
The code can be run in two different ways.
One way is through `learning.py` that can be used to run small experiments on a local computer.
The other way is through the files inside the Job directory. 
We explain each of these approaches below by means of an example.

### Running on Your Local Machine
Let's take the following example: applying LC-ETD1 to the Two-state task.
There are multiple ways for doing this.
The first way is to open a terminal and go into the root directory of the code and run `Learning.py` with proper parameters:
```
python3 Learning.py --algorithm LCETD1 --task TwoStateSimpleHighVariance --num_of_runs 100 --num_steps 100000 --environment BidirectionalChain
--alpha 0.0001 --beta 0.2 --lmbda 0.0
```
In case any of the parameters are not specified, a default value will be used.
The default value is set in the `Job` directory, inside the `JobBuilder.py` file.
This means, the code, can alternatively be run, by setting all the necessary values that an algorithm needs at the top of the `JobBuilder.py` file.
Note that not all parameters specified in the `default_params` dict are required for all algorithms. For example, the `beta` parameter is only
required to be set for the emphatic algorithms (excluding Emphatic-TD).
Once the variables inside the `default_params` dictionary, the code can be run:
```
python3 Learning.py
```
Or one can choose to specify some parameters in the `default_params` dictionary and specify the rest as command line argumets 
like the following:
```
python3 Learning.py --algorithm LCETD1 --task TwoStateSimpleHighVariance --alpha 0.01
```

### Running on Servers with Slurm Workload Managers
When parameter sweeps are necessary, the code can be run on supercomputers. 
The current code supports running on servers that use slurm workload managers such as compute canada.
For exampole, to apply the LC-ETD1 algorithm to the Two-state (TwoStateSimpleHighVariance) task, with various parameters,
first you need to create a json file that specifies all the parameters that you would like to run, for example:
```json
{
  "agent": "LCETD1",
  "environment": "BidirectionalChain",
  "task": "TwoStateSimpleHighVariance",
  "number_of_runs": 100,
  "number_of_steps": 100000,
  "sub_sample": 1,
  "meta_parameters": {
    "alpha": [
      0.000003814, 0.000007629, 0.000015258, 0.000030517, 0.000061035, 0.000122070, 0.000244140, 0.000488281,
      0.000976562, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0
    ],
    "beta": [
      0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    ],
    "lmbda": [
      0.0, 0.5, 0.9
    ]
  }
}
```
and then run `main.py` using python:
```
python3 main.py -f <path_to_the_json_file> -s <kind_of_submission>
```
where `kind_of_submission` refers to one of the two ways you can submit your code:
1) You can request an individual cpu for each of the algorithm instances, where an algorithm instance refers to an 
algorithm with specific parameters. To request an individual cpu, run the following command:
```
python3 main.py -f <path_to_the_json_file_or_dir> -s cpu
```
When running each algorithm instance on a single cpu, you need to specify the following parameters inside 
`Job/SubmitJobsTemplatesCedar.SL`:
```shell
#SBATCH --account=xxx
#SBATCH --time=00:15:58
#SBATCH --mem=3G
```
where `#SBATCH --account=xxx` requires the account you are using in place of `xxx`,
`#SBATCH --time=00:15:58` requires the time you want to request for each individual cpu,
and `#SBATCH --mem=xG` requires the amount of memory in place of x.

2) You can request a node, that we assume includes 40 cpus. If you request a node, the jobs you submit will run in 
parallel 40 at a time, and once one job is finished, the next one in line will start running.
This process continues until either all jobs are finished running, or you run out of the time you requested for that node.
```
python3 main.py -f <path_to_the_json_file_or_dir> -s node
```
When running the jobs on nodes, you need to specify the following parameters inside `Job/SubmitJobsTemplates.SL`:
```shell
#SBATCH --account=xxx
#SBATCH --time=11:58:59
#SBATCH --nodes=x
#SBATCH --ntasks-per-node=40
```
where `#SBATCH --account=xxx` requires the account you are using in place of `xxx`,
`#SBATCH --time=11:58:59` requires the time you want to request for each individual node, each of which includes 40 cpus in this case,
and `#SBATCH --nodes=x` requires the number of nodes you would like to request in place of x.
If you request more than one node, your jobs will be spread across nodes, 40 on each node, and once each job finishes, 
the next job in the queue will start running.
`#SBATCH --ntasks-per-node=xx` is the number of jobs you would like to run concurrently on a single node. In this case,
for example, we set it to 40.

If `path_to_the_json_file_or_dir` is a directory, then the code will walk into all the subdirectories, and submits jobs for
all the parameters in the json files that it finds inside those directories sequentially.
If `path_to_the_json_file_or_dir` is a file, then the code will submit jobs for all the parameters that it finds inside that 
single json file.
Note that you can create a new directory for each experiment that you would like to run, and create directories for each
of the algorithms you would like to run in that experiment.
For example, we created a directory called `TwoState` inside the `Experiments` directory and created one directory
per algorithm inside the `TwoState` directory for each of the algorithms and specified a json file in that directory.
It is worth noting that whatever parameter that is not specified in the json file will be read from the `default_params`
dictionary inside the `Job` directory inside the `JobBuilder.py` file.
