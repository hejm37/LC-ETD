import os
import numpy as np
import argparse

from data_presister import DataPersister, ParameterBuilder
from utils import save_result, Configuration, save_value_function, get_save_value_function_steps
from Registry.AlgRegistry import alg_dict
from Registry.EnvRegistry import environment_dict
from Registry.TaskRegistry import task_dict
from Job.JobBuilder import default_params
from Environments.rendering import ErrorRender


def learn(config: Configuration):
    params = ParameterBuilder().add_algorithm_params(config).build()

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)

    env = environment_dict[config.environment](**config)

    rmsve = np.zeros((task_dict[config.task].num_of_policies(), config.num_steps, config.num_of_runs))
    for run in range(config.num_of_runs):
        random_seed = (run + config.num_of_runs) if config.rerun else run
        random_seed += config.seed if 'seed' in config else 0
        np.random.seed(random_seed)
        task = task_dict[config.task](run_number=run, num_steps=config.num_steps, lmbda=params['lmbda'])
        agent = alg_dict[config.algorithm](task, **params)

        rmsve_of_run = np.zeros((task.num_policies, task.num_steps))
        agent.state = env.reset()
        error_render = ErrorRender(task.num_policies, task.num_steps)
        for step in range(task.num_steps):
            rmsve_of_run[:, step], error = agent.compute_rmsve()
            if config.render:
                error_render.add_error(error)
            agent.action = agent.choose_behavior_action()
            agent.next_state, r, is_terminal, info = env.step(agent.action)
            agent.learn(agent.state, agent.next_state, r, is_terminal)
            if config.render:
                env.render(mode='screen', render_cls=error_render)
            if config.save_value_function and (step in get_save_value_function_steps(task.num_steps)):
                save_value_function(agent.compute_value_function(), config.save_path, step, run)
            if is_terminal:
                agent.state = env.reset()
                agent.reset()
                continue
            agent.state = agent.next_state
        print(f'Run {run}:', np.mean(rmsve_of_run, axis=0))
        rmsve[:, :, run] = rmsve_of_run
    rmsve_of_runs = np.transpose(np.mean(rmsve, axis=0))  # Average over all policies.
    # print(rmsve_of_runs.shape)  # (num_runs, num_steps)

    # _IQM_mean_over_runs
    q75, q25 = np.percentile(rmsve_of_runs, [75, 25], axis=0)
    # print(q75.shape)  # (num_steps)
    mask75 = rmsve_of_runs > q75
    mask25 = rmsve_of_runs < q25
    rmsve_masked = np.ma.masked_where(mask75 | mask25, rmsve_of_runs)
    masked_mean = np.ma.getdata(rmsve_masked.mean(axis=0))
    masked_std = np.ma.getdata(rmsve_masked.std(axis=0, ddof=1) / np.sqrt(config.num_of_runs * 0.5))
    print('IQM mean:', masked_mean)
    print('IQM std', masked_std)

    DataPersister.save_result(masked_mean, '_RMSVE_mean_over_runs_IQM', config)
    DataPersister.save_result(masked_std, '_RMSVE_stderr_over_runs_IQM', config)
    final_errors_mean_over_steps = rmsve_masked[:, config.num_steps - int(0.01 * config.num_steps) - 1:].mean(axis=1)
    final_errors_mean_over_runs = np.ma.getdata(final_errors_mean_over_steps.mean())
    final_errors_std_over_runs = np.ma.getdata(final_errors_mean_over_steps.std(ddof=1) / np.sqrt(config.num_of_runs * 0.5))
    print('final_errors_mean_over_runs:', final_errors_mean_over_runs)
    DataPersister.save_result(np.array([final_errors_mean_over_runs, final_errors_std_over_runs]), '_mean_stderr_final_IQM', config)
    auc_mean_over_steps = rmsve_masked.mean(axis=1)
    auc_mean_over_runs = np.ma.getdata(auc_mean_over_steps.mean())
    auc_std_over_runs = np.ma.getdata(auc_mean_over_steps.std(ddof=1) / np.sqrt(config.num_of_runs * 0.5))
    print('auc_mean_over_runs:', auc_mean_over_runs)
    DataPersister.save_result(np.array([auc_mean_over_runs, auc_std_over_runs]), '_mean_stderr_auc_IQM', config)

    # _RMSVE_mean_over_runs
    mean = np.mean(rmsve_of_runs, axis=0)
    std = np.std(rmsve_of_runs, axis=0, ddof=1) / np.sqrt(config.num_of_runs)
    print('mean:', mean)
    print('std:', std)

    DataPersister.save_result(mean, '_RMSVE_mean_over_runs', config)
    DataPersister.save_result(std, '_RMSVE_stderr_over_runs', config)

    # _mean_stderr_final
    final_errors_mean_over_steps = np.mean(rmsve_of_runs[:, config.num_steps - int(0.01 * config.num_steps) - 1:],
                                           axis=1)
    final_errors_mean_over_runs = np.mean(final_errors_mean_over_steps)
    final_errors_std_over_runs = np.std(final_errors_mean_over_steps, ddof=1) / np.sqrt(config.num_of_runs)
    print('final_errors_mean_over_runs:', final_errors_mean_over_runs)
    DataPersister.save_result(np.array([final_errors_mean_over_runs, final_errors_std_over_runs]), '_mean_stderr_final', config)
    # save_result(config.save_path, '_mean_stderr_final',
    #             np.array([np.mean(final_errors_mean_over_steps), np.std(final_errors_mean_over_steps, ddof=1) /
    #                       np.sqrt(config.num_of_runs)]), params, config.rerun)

    # _mean_stderr_auc
    auc_mean_over_steps = np.mean(rmsve_of_runs, axis=1)
    auc_mean_over_runs = np.mean(auc_mean_over_steps)
    auc_std_over_runs = np.std(auc_mean_over_steps, ddof=1) / np.sqrt(config.num_of_runs)
    print('auc_mean_over_runs:', auc_mean_over_runs)
    DataPersister.save_result(np.array([auc_mean_over_runs, auc_std_over_runs]), '_mean_stderr_auc', config)
    # save_result(config.save_path, '_mean_stderr_auc',
    #             np.array([np.mean(auc_mean_over_steps),
    #                       np.std(auc_mean_over_steps, ddof=1) / np.sqrt(config.num_of_runs)]), params, config.rerun)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', '-a', type=float, default=default_params['meta_parameters']['alpha'])
    parser.add_argument('--lmbda', '-l', type=float, default=default_params['meta_parameters']['lmbda'])
    parser.add_argument('--eta', '-et', type=float, default=default_params['meta_parameters']['eta'])
    parser.add_argument('--beta', '-b', type=float, default=default_params['meta_parameters']['beta'])
    parser.add_argument('--zeta', '-z', type=float, default=default_params['meta_parameters']['zeta'])
    parser.add_argument('--tdrc_beta', '-tb', type=float, default=default_params['meta_parameters']['tdrc_beta'])
    parser.add_argument('--gem_alpha', '-ga', type=float, default=default_params['meta_parameters']['gem_alpha'])
    parser.add_argument('--gem_beta', '-gb', type=float, default=default_params['meta_parameters']['gem_beta'])
    parser.add_argument('--algorithm', '-alg', type=str, default=default_params['agent'])
    parser.add_argument('--task', '-t', type=str, default=default_params['task'])
    parser.add_argument('--num_of_runs', '-nr', type=int, default=default_params['num_of_runs'])
    parser.add_argument('--num_steps', '-ns', type=int, default=default_params['num_steps'])
    parser.add_argument('--sub_sample', '-ss', type=int, default=default_params['sub_sample'])
    parser.add_argument('--environment', '-e', type=str, default=default_params['environment'])
    parser.add_argument('--reward', '-rd', type=int, default=1)
    parser.add_argument('--save_path', '-sp', type=str, default='-')
    parser.add_argument('--rerun', '-rrn', type=bool, default=False)
    parser.add_argument('--render', '-rndr', type=bool, default=False)
    parser.add_argument('--save_value_function', '-svf', type=bool, default=default_params['save_value_function'])
    parser.add_argument('--seed', '-s', type=int, default=default_params['seed'])
    args = parser.parse_args()
    if args.save_path == '-':
        args.save_path = os.path.join(os.getcwd(), 'Results', default_params['exp'], args.algorithm)

    learn(config=Configuration(vars(args)))
