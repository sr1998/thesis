def fast_adapt(
    X_support,
    y_support,
    learner,
    loss,
    adapt_opt,
    n_adaptation_steps,
    initial_lr,
    inner_rl_reduction_factor,
):
    # Adapt the model
    for step in range(n_adaptation_steps):
        lr = initial_lr / inner_rl_reduction_factor
        for param_group in adapt_opt.param_groups:
            param_group["lr"] = lr
        adapt_opt.zero_grad()
        error = loss(learner(X_support).squeeze(), y_support)
        error.backward()
        adapt_opt.step()

    return learner
