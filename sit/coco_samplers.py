"""Deterministic ODE sampler for the COCO MMDiT experiments."""

import torch


def euler_sampler(
    model,
    latents,
    context,
    null_context,
    num_steps=50,
    cfg_scale=2.0,
):
    """Euler ODE sampler with full-interval classifier-free guidance."""
    if latents.shape[0] != context.shape[0]:
        raise ValueError("Latents and context must have the same batch size")

    dtype = latents.dtype
    device = latents.device
    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float64)
    x_next = latents.to(torch.float64)

    with torch.no_grad():
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            x_cur = x_next
            if cfg_scale == 1.0:
                model_input = x_cur
                context_input = context
            else:
                model_input = torch.cat([x_cur, x_cur], dim=0)
                context_input = torch.cat([context, null_context], dim=0)
            time_input = torch.full(
                (model_input.shape[0],),
                float(t_cur),
                device=device,
                dtype=dtype,
            )
            velocity = model(model_input.to(dtype), time_input, context_input)[0].to(torch.float64)
            if cfg_scale == 1.0:
                guided_velocity = velocity
            else:
                cond_velocity, uncond_velocity = velocity.chunk(2, dim=0)
                guided_velocity = uncond_velocity + cfg_scale * (cond_velocity - uncond_velocity)
            x_next = x_cur + (t_next - t_cur) * guided_velocity

    return x_next.to(dtype)
