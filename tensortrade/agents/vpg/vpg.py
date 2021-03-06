import numpy as np
from numpy.lib.npyio import save
import torch
from torch.optim import Adam
import gym
import time
import os

from . import core
from tensortrade.agents.utils.logx import EpochLogger
# from tensortrade.agents.utils.mpi_pytorch import (
#     setup_pytorch_for_mpi,
#     sync_params,
#     mpi_avg_grads,
# )
# from tensortrade.agents.utils.mpi_tools import (
#     mpi_fork,
#     mpi_avg,
#     proc_id,
#     mpi_statistics_scalar,
#     num_procs,
# )
from tensortrade.agents import Agent


DIV_LINE_WIDTH = 80
DEFAULT_DATA_DIR = "experiments"

def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.
    If no seed is given and datestamp is false, 
    ::
        output_dir = data_dir/exp_name
    If a seed is given and datestamp is false,
    ::
        output_dir = data_dir/exp_name/exp_name_s[seed]
    If datestamp is true, amend to
    ::
        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]
    You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in 
    ``spinup/user_config.py``. 
    Args:
        exp_name (string): Name for experiment.
        seed (int): Seed for random number generators used by experiment.
        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``.
        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.
    Returns:
        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])
    
    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = os.path.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=os.path.join(data_dir, relpath), 
                         exp_name=exp_name)
    return logger_kwargs



class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class VPG(Agent):
    """
    Vanilla Policy Gradient 

    (with GAE-Lambda for advantage estimation)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    def __init__(
        self,
        exp_name,
        env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(),
        seed=0,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=1000,
        save_freq=10,
    ):

        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.epochs = epochs

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        # setup_pytorch_for_mpi()

        # Set up logger and save configuration
        logger_kwargs = setup_logger_kwargs(exp_name, seed)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # Random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate environment
        self.env = env_fn()
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape

        # Create actor-critic module
        self.ac = actor_critic(obs_dim, act_dim, **ac_kwargs)

        # Sync params across processes
        # sync_params(self.ac)

        # Count variables
        var_counts = tuple(
            core.count_vars(module) for module in [self.ac.pi, self.ac.v]
        )
        self.logger.log("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

        # Set up experience buffer
        self.local_steps_per_epoch = int(steps_per_epoch)
        self.buf = VPGBuffer(obs_dim, act_dim, self.local_steps_per_epoch, gamma, lam)

        self.env.agent_id = self.id

    # Set up function for computing VPG policy loss
    def _compute_loss_pi(self, data):
        obs, act, adv, logp_old = data["obs"], data["act"], data["adv"], data["logp"]

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def _compute_loss_v(self, data):
        obs, ret = data["obs"], data["ret"]
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def _update(self):
        data = self.buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = self._compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self._compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self._compute_loss_pi(data)
        loss_pi.backward()
        # mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
        self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(self.ac.v)  # average grads across MPI processes
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent = pi_info["kl"], pi_info_old["ent"]
        self.logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            KL=kl,
            Entropy=ent,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV=(loss_v.item() - v_l_old),
        )

    def restore(self, path: str, **kwargs):
        self.ac.load_state_dict(torch.load(path))

    def save(self, path: str, **kwargs):
        episode: int = kwargs.get("episode", None)

        if episode:
            filename = (
                "policy_network__" + self.id + "__" + str(episode).zfill(3) + ".hdf5"
            )
        else:
            filename = "policy_network__" + self.id + ".hdf5"

        torch.save(self.ac.state_dict(), path + filename)

    def train(self, render_interval=100, save_path=None):
        # Set up optimizers for policy and value function
        pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

        # Prepare for interaction with environment
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            for t in range(self.local_steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))

                next_o, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1

                # save and log
                self.buf.store(o, a, r, v, logp)
                self.logger.store(VVals=v)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == self.max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.local_steps_per_epoch - 1

                if render_interval is not None and ep_len % render_interval == 0:
                    self.env.render(
                        episode=epoch,
                        max_episodes=self.epochs,
                        max_steps=self.max_ep_len,
                    )
                self.env.save()

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print(
                            "Warning: trajectory cut off by epoch at %d steps."
                            % ep_len,
                            flush=True,
                        )
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = self.env.reset(soft=True), 0, 0

            # Save model
            if save_path and (
                (epoch % self.save_freq == 0) or (epoch == self.epochs - 1)
            ):
                self.save(save_path)

            # Perform VPG update!
            self.update()

            # Log info about epoch
            self.logger.log_tabular("Epoch", epoch)
            self.logger.log_tabular("EpRet", with_min_and_max=True)
            self.logger.log_tabular("EpLen", average_only=True)
            self.logger.log_tabular("VVals", with_min_and_max=True)
            self.logger.log_tabular(
                "TotalEnvInteracts", (epoch + 1) * self.steps_per_epoch
            )
            self.logger.log_tabular("LossPi", average_only=True)
            self.logger.log_tabular("LossV", average_only=True)
            self.logger.log_tabular("DeltaLossPi", average_only=True)
            self.logger.log_tabular("DeltaLossV", average_only=True)
            self.logger.log_tabular("Entropy", average_only=True)
            self.logger.log_tabular("KL", average_only=True)
            self.logger.log_tabular("Time", time.time() - start_time)
            self.logger.dump_tabular()
