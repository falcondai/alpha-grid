import grid2op as gp
from lightsim2grid import LightSimBackend
from numpy import mean, median
from matplotlib import subplots, xlabel


def quick_start():
    from grid2op.PlotGrid import PlotMatplot
    env = gp.make('l2rpn_neurips_2020_track1_small', backend=LightSimBackend())
    plot_helper = PlotMatplot(env.observation_space, line_id=True)
    return env, plot_helper


def april_eval(agent, backend=LightSimBackend(), seed=253):
    # Performance is measured in rewards and durations
    acc_rews, durs = [], []

    env = gp.make('l2rpn_neurips_2020_track1_small', backend=backend)
    env.set_id(0)
    env.seed(seed)
    for i in range(48):
        obs = env.reset()
        print(i, env.chronics_handler.get_name(), end=' ')
        done, acc_rew, dur, rew = False, 0, 0, 0
        while not done:
            ac = agent.act(obs, rew)
            obs, rew, done, _ = env.step(ac)
            acc_rew += rew
            dur += 1
        print(acc_rew, dur)
        acc_rews.append(acc_rew)
        durs.append(dur)
    return acc_rews, durs


def summary_and_plot(acc_rewards, durations, draw_plots=True):
    print(
        'mean rewards:',
        mean(acc_rewards),
        'mean duration:',
        mean(durations),
        'median rewards:',
        median(acc_rewards),
        'median duration:',
        median(durations),
    )
    if draw_plots:
        _, (ax1, ax2) = subplots(2, 1, sharex='all')
        ax1.bar(range(48), sorted(acc_rewards, reverse=True))
        ax2.bar(range(48), sorted(durations, reverse=True))
        xlabel('rank')
    return mean(acc_rewards), mean(durations), median(acc_rewards), median(durations)
