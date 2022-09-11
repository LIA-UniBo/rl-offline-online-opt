import numpy as np
from pyagents.agents import SAC


class SacMod(SAC):

    def init(self, env, env_config=None, min_memories=None, actions=None, *args, **kwargs):
        self.num_envs = getattr(env, "num_envs", 1)
        if self._wandb_run is not None and env_config is not None:
            self._wandb_run.config.update(env_config)
        if min_memories is None:
            min_memories = self._memory.get_config()['size']
        s_t = env.reset().reshape(1, -1)
        for _ in range(min_memories // self.num_envs):
            a_t = env.action_space.sample()
            s_tp1, r_t, done, info = env.step(a_t)
            s_tp1 = s_tp1.reshape(1, -1)
            feasible_action = env.rescale(info['action'], to_network_range=True).reshape(1, -1)
            self.remember(state=s_t,
                          action=feasible_action,
                          reward=np.asarray([r_t]),
                          next_state=s_tp1,
                          done=[done])
            s_t = env.reset().reshape(1, -1) if done else s_tp1
