import grid2op as gp
from lightsim2grid import LightSimBackend
from l2rpn_baselines.PandapowerOPFAgent import PandapowerOPFAgent

backend = LightSimBackend()

env = gp.make('l2rpn_neurips_2020_track1_small', backend=backend)
env.seed(1)
obs = env.reset()
opf_agent = PandapowerOPFAgent(env.action_space, env.init_grid_path, opf_type='powermodels')
print(env.chronics_handler.get_name())
done = False
acc_rew, duration = 0, 0
while not done:
#     ac = null_agent.act(obs, rew)
    ac = opf_agent.act(obs, 0)
    print(ac)
    obs, rew, done, info = env.step(ac)
    acc_rew += rew
    duration += 1
    if duration > 10:
        break
print(acc_rew, duration)
# Takes 9.86 s ± 4.76 ms per loop (7 runs) on pandapower backend
# Takes 2.08 s ± 83.6 ms per loop (7 runs) on lightsim2grid backend
