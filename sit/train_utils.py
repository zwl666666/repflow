import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from transport import path
from transport.transport import ModelType
from transport.utils import mean_flat


def none_or_str(value):
    if value == 'None':
        return None
    return value


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for param in model.parameters():
        param.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def transport_training_losses_with_features(
    transport,
    model,
    x1,
    model_kwargs=None,
    feature_layers=None,
    feature_layer=-1,
    return_teacher_features=True,
    align_feature_layer=None,
    repa_feature_layer=None,
):
    """
    Compute SiT transport loss and optionally teacher token features from one forward pass.
    When align_feature_layer is set, teacher_align_feat holds LN tokens at that block (same forward).
    When repa_feature_layer is set, teacher_repa_feat holds tokens for REPA alignment.
    """
    if model_kwargs is None:
        model_kwargs = {}

    t, x0, x1 = transport.sample(x1)
    t, xt, ut = transport.path_sampler.plan(t, x0, x1)
    need_features = return_teacher_features or align_feature_layer is not None or repa_feature_layer is not None
    if need_features:
        if return_teacher_features:
            layers = feature_layers if feature_layers is not None else [feature_layer]
        else:
            layers = []
        model_output, teacher_feat, teacher_align_feat, teacher_repa_feat = model(
            xt,
            t,
            return_features=True,
            feature_layers=layers,
            align_feature_layer=align_feature_layer,
            repa_feature_layer=repa_feature_layer,
            use_dropout=False,
            **model_kwargs,
        )
        if not return_teacher_features:
            teacher_feat = None
    else:
        model_output = model(
            xt,
            t,
            return_features=False,
            use_dropout=False,
            **model_kwargs,
        )
        teacher_feat = None
        teacher_align_feat = None
        teacher_repa_feat = None

    batch_size, *_, channels = xt.shape
    assert model_output.size() == (batch_size, *xt.size()[1:-1], channels)

    terms = {"pred": model_output, "x0": x0, "x1": x1, "xt": xt}
    if transport.model_type == ModelType.VELOCITY:
        terms["loss"] = mean_flat((model_output - ut) ** 2)
    else:
        _, drift_var = transport.path_sampler.compute_drift(xt, t)
        sigma_t, _ = transport.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
        if transport.loss_type.name == "VELOCITY":
            weight = (drift_var / sigma_t) ** 2
        elif transport.loss_type.name == "LIKELIHOOD":
            weight = drift_var / (sigma_t ** 2)
        elif transport.loss_type.name == "NONE":
            weight = 1
        else:
            raise NotImplementedError()

        if transport.model_type == ModelType.NOISE:
            terms["loss"] = mean_flat(weight * ((model_output - x0) ** 2))
        else:
            terms["loss"] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

    return terms, teacher_feat, t, teacher_align_feat, teacher_repa_feat


def token_level_cosine_loss(teacher_tokens, student_tokens, eps=1e-8):
    """
    Token-wise cosine distance: mean over tokens of (1 - cos_sim), cos_sim on the last dimension.
    teacher_tokens, student_tokens: (N, T, D) with the same D.
    """
    t = F.normalize(teacher_tokens, dim=-1, eps=eps)
    s = F.normalize(student_tokens, dim=-1, eps=eps)
    return (1.0 - (t * s).sum(dim=-1)).mean()


def sra_layer_index(block_out_1based: int) -> int:
    """SRA --block-out-* is 1-based (after block i when i+1 == ad). Convert to 0-based."""
    idx = int(block_out_1based) - 1
    if idx < 0:
        raise ValueError(f"SRA block-out must be >= 1, got {block_out_1based}")
    return idx


def sra_alignment_loss(online_feat, teacher_feat, loss_type="cos"):
    """SRA-main Simpleloss on token features (N, T, D). Cosine uses last dim."""
    if loss_type == "cos":
        return (1.0 - F.cosine_similarity(online_feat, teacher_feat, dim=-1)).mean()
    if loss_type == "sml1":
        return F.smooth_l1_loss(online_feat, teacher_feat, beta=0.05)
    if loss_type == "l2":
        return F.mse_loss(online_feat, teacher_feat)
    if loss_type == "l1":
        return F.l1_loss(online_feat, teacher_feat)
    raise NotImplementedError(f"Unknown SRA loss type: {loss_type}")


def masked_smooth_l1_loss(pred, target, mask=None, eps=1e-6):
    """
    Smooth L1 loss on token features, optionally restricted to masked positions.
    """
    loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    if mask is None:
        return loss.mean()
    mask = mask.float()
    return (loss * mask).sum() / mask.sum().clamp_min(eps)

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)

def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")