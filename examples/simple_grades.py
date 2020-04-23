if __name__ == '__main__':
    from ppo_pytorch.examples.init_vars import *
    from torch.optim.adamw import AdamW

    env_factory = partial(rl.common.SimpleVecEnv, 'CartPole-v1', parallel='dummy')

    alg_class = rl.algs.GradES
    alg_params = rl.algs.create_ppo_kwargs(
        2e6,

        num_actors=4,
        es_learning_rate=0.1,
        models_per_update=8,
        episodes_per_model=None,
        horizon=64,
        batch_size=64,
        entropy_loss_scale=0.0,
        cuda_eval=False,
        cuda_train=False,
        barron_alpha_c=(1.5, 1),
        lr_scheduler_factory=None,
        clip_decay_factory=None,
        entropy_decay_factory=None,
        replay_buf_size=16 * 1024,
        num_batches=16,
        min_replay_size=10000,
        grad_clip_norm=None,
        kl_pull=0.0,
        replay_end_sampling_factor=1.0,
        eval_model_blend=1.0,

        # model_factory=partial(rl.actors.create_ppo_fc_actor, hidden_sizes=(256, 256, 256),
        #                       activation=nn.ReLU),
        optimizer_factory=partial(RMSprop, lr=5e-4, eps=1e-5),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[]',
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=2e6)
