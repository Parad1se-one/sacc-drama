from sacc.envs.drama import DramaRMEnv


def test_per_agent_rm_shapes_and_masks():
    env = DramaRMEnv(difficulty="easy", seed=7)
    obs_list = env.build_obs_list()
    state = env.build_state()
    masks = env.get_action_masks()

    assert len(obs_list) == env.num_agents
    assert all(obs.shape[0] == env.obs_dim for obs in obs_list)
    assert state.shape[0] == env.state_dim
    assert masks.shape == (env.num_agents, env.action_n)
    assert all(mask.any() for mask in masks)
