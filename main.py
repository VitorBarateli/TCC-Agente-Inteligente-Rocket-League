def build_rlgym_v2_env():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper
    from rewards import InAirReward, SpeedTowardBallReward, VelocityBallToGoalReward, FaceBallReward, BallCloserToGoalReward, AdvancedUsefulFlipReward, AdvancedTouchForceReward, AdvancedCollectBoostReward, SaveBoostReward, AerialTouchReward, AdvancedKickoffReward, WasteBoostAtMaxSpeedPunishment
    from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import RocketSimVisRenderer

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds)
    )

    reward_fn = CombinedReward(
        (InAirReward(), 0.5),
        (AerialTouchReward(), 15),
        (FaceBallReward(), 1),
        (AdvancedTouchForceReward(), 30),
        (AdvancedCollectBoostReward(), 25),
        (SaveBoostReward(), 10),
        (WasteBoostAtMaxSpeedPunishment(), 10),
        (SpeedTowardBallReward(), 7),
        (VelocityBallToGoalReward(), 20),
        (AdvancedUsefulFlipReward(), 4),
        (BallCloserToGoalReward(), 15),
        (AdvancedKickoffReward(), 40),
        (GoalReward(), 20)
    )

    obs_builder = DefaultObs(zero_padding=None,
                           pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 
                                              1 / common_values.BACK_NET_Y, 
                                              1 / common_values.CEILING_Z]),
                           ang_coef=1 / np.pi,
                           lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                           ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                           boost_coef=1 / 100.0)

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator()
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer()
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from rlgym_ppo import Learner
    import os

    n_proc = 24

    min_inference_size = max(1, int(round(n_proc * 0.9)))

    base_dir = "data/checkpoints/"

    if os.path.exists(base_dir):
        # Encontra o diretório com o maior número dentro de base_dir
        latest_folder = os.path.join(base_dir, "rlgym-ppo-run-" + str(max(
            (int(d.split('-')[-1]) for d in os.listdir(base_dir) if d.startswith('rlgym-ppo-run-')),
            default=-1
        )))

        latest_checkpoint_dir = os.path.join(latest_folder, str(max(
                (int(d) for d in os.listdir(latest_folder) if d.isdigit()),
                default=-1
            )))
    else:
        latest_checkpoint_dir = None

    learner = Learner(build_rlgym_v2_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=None,
                      ppo_batch_size=100_000,  # batch size
                      policy_layer_sizes=[1024, 1024, 512, 512],  # policy network
                      critic_layer_sizes=[1024, 1024, 512, 512],  # critic network
                      ts_per_iteration=100_000,  # timesteps per training iteration
                      exp_buffer_size=300_000,  # size of experience buffer
                      ppo_minibatch_size=50_000,  # minibatch size
                      ppo_ent_coef=0.01,  # entropy coefficient
                      policy_lr=1e-4,  # policy learning rate
                      critic_lr=1e-4,  # critic learning rate
                      ppo_epochs=2,   # number of PPO epochs
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=1_000_000,
                      timestep_limit=465_000_000,
                      log_to_wandb=False,
                      render=True,
                      render_delay=0.047,
                      checkpoint_load_folder=latest_checkpoint_dir
                      )
    
    learner.learn()