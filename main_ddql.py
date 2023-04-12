from Common import Play, get_params, Logger, make_handmade_mario, stack_states
import numpy as np
from Agent.DDQL_agent import MarioDDQL
from tqdm import tqdm


if __name__ == '__main__':
    config = get_params()
    config["algo"] = "DDQL"

    test_env = make_handmade_mario(config["max_frames_per_episode"])
    config.update({"n_actions": test_env.action_space.n})
    test_env.close()

    config.update({"batch_size": (config["rollout_length"] * config["n_workers"]) // config["n_mini_batch"]})
    config.update({"predictor_proportion": 32 / config["n_workers"]})

    agent = MarioDDQL(state_dim=(4, 84, 84), action_dim=config['n_actions'])

    if config["do_train"]:

        logger = Logger(agent, experiment=None, **config)

        if not config["train_from_scratch"]:
            init_iteration, episode = logger.load_weights()

        else:
            init_iteration = 0
            episode = 0

        envs = [make_handmade_mario(config["max_frames_per_episode"]) for i in range(config["n_workers"])]

        for env in envs:
            env.reset()
            env.stacked_states = np.zeros(config["state_shape"], dtype=np.uint8)
            env.t = 0
            env.max_pos = 0
            env.episod_reward = 0

        if config["train_from_scratch"]:
            print("---Pre_normalization started.---")
            total_pre_normalization_steps = int(config["rollout_length"] * config["pre_normalization_steps"] / config["n_workers"])
            actions = np.random.randint(0, config["n_actions"], (total_pre_normalization_steps, config["n_workers"]))
            for t in tqdm(range(total_pre_normalization_steps)):

                for worker_id, env in enumerate(envs):
                    state = env.stacked_states
                    next_state, r, d, info = env.step(actions[t, worker_id])
                    if config["render"] and worker_id==0:
                        env.render()
                    env.t += 1
                    if env.t % env.max_episode_steps == 0 or info["flag_get"]:
                        d = True
                        env.t = 0

                    env.max_pos = max(env.max_pos, info["x_pos"])
                    env.episod_reward += r
                    env.stacked_states = stack_states(env.stacked_states, next_state, False)
                    agent.cache(state, env.stacked_states, actions[t, worker_id], r, d)

                    if d:
                        state = env.reset()
                        env.stached_states = stack_states(env.stacked_states, state, True)
                        env.episode_reward = 0
                        env.max_pos = 0
            print("---Pre_normalization is done.---")

        rollout_base_shape = config["n_workers"], config["rollout_length"]

        init_states = np.zeros(rollout_base_shape + config["state_shape"], dtype=np.uint8)
        init_actions = np.zeros(rollout_base_shape, dtype=np.uint8)
        init_action_probs = np.zeros(rollout_base_shape + (config["n_actions"],))
        init_int_rewards = np.zeros(rollout_base_shape)
        init_ext_rewards = np.zeros(rollout_base_shape)
        init_dones = np.zeros(rollout_base_shape, dtype=np.bool)
        init_int_values = np.zeros(rollout_base_shape)
        init_ext_values = np.zeros(rollout_base_shape)
        init_log_probs = np.zeros(rollout_base_shape)
        init_next_states = np.zeros((rollout_base_shape[0],) + config["state_shape"], dtype=np.uint8)
        init_next_obs = np.zeros(rollout_base_shape + config["obs_shape"], dtype=np.uint8)

        logger.on()
        episode_ext_reward = 0
        concatenate = np.concatenate
        for iteration in tqdm(range(init_iteration + 1, config["total_rollouts_per_env"] + 1)):
            total_states = init_states
            total_actions = init_actions
            total_action_probs = init_action_probs
            total_int_rewards = init_int_rewards
            total_ext_rewards = init_ext_rewards
            total_dones = init_dones
            total_int_values = init_int_values
            total_ext_values = init_ext_values
            total_log_probs = init_log_probs
            next_states = init_next_states
            total_next_obs = init_next_obs

            for t in range(config["rollout_length"]):
                infos = []
                for worker_id, env in enumerate(envs):
                    state = env.stacked_states
                    action, proba = agent.step(state)
                    next_state, r, d, info = env.step(action)
                    if config["render"] and worker_id==0:
                        env.render()
                    env.t += 1
                    if env.t % env.max_episode_steps == 0 or info["flag_get"]:
                        d = True
                    env.episod_reward += r
                    env.stacked_states = stack_states(env.stacked_states, next_state, False)
                    agent.cache(state, env.stacked_states, action, r, d)
                    infos.append(info)
                    total_dones[worker_id, t] = d
                    total_action_probs[worker_id, t] = proba
                    if d:
                        state = env.reset()
                        env.stached_states = stack_states(env.stacked_states, state, True)
                        if worker_id == 0:
                            episode_ext_reward = env.episod_reward
                        env.episod_reward = 0

                if total_dones[0, t]:
                    episode += 1
                    x_pos = infos[0]["x_pos"]
                    stage = infos[0]["stage"]
                    logger.log_episode(episode, episode_ext_reward, x_pos, stage)

            training_logs = agent.learn()
            agent.schedule_exploration_rate()
            # agent.schedule_clip_range(iteration)

            logger.log_iteration_ddql(iteration,
                                 training_logs,
                                 total_action_probs[0].max(-1).mean(),
                                 agent.exploration_rate)

    else:
        logger = Logger(agent, experiment=None, **config)
        logger.load_weights_ddql()
        play = Play(None, agent)
        play.evaluate()
