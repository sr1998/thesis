def fast_adapt(
    X_support,
    y_support,
    learner,
    loss,
    adapt_opt,
    adaptation_steps,
    initial_lr: int = 0.5,
    inner_rl_reduction_factor: int = 1.5,
):
    # Adapt the model
    for step in range(adaptation_steps):
        lr = initial_lr / inner_rl_reduction_factor
        for param_group in adapt_opt.param_groups:
            param_group["lr"] = lr
        adapt_opt.zero_grad()
        error = loss(learner(X_support).squeeze(), y_support)
        error.backward()
        adapt_opt.step()

    return learner
