import argparse
import copy
from copy import deepcopy
import logging
import os
import yaml
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from pathlib import Path
from collections import OrderedDict
import json

import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers.optimization import get_scheduler

from models.sit import SiT_models
from loss import SRALoss, linear_cls_probability

from dataset import HFImgLatentDataset
from dino_cls import load_dino_cls_extractor
from diffusers.models import AutoencoderKL

import math
from torchvision.utils import make_grid
from PIL import Image

logger = get_logger(__name__)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device

    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}

    dino_cls_extractor = load_dino_cls_extractor(
        args.dino_model_name,
        args.resolution,
        device,
        accelerator,
    )
    if dino_cls_extractor.embed_dim != args.cls_dim:
        raise ValueError(
            f"DINO CLS dimension {dino_cls_extractor.embed_dim} does not match "
            f"--cls-dim={args.cls_dim}"
        )

    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        class_dropout_prob=args.cfg_prob,
        cls_condition_dim=args.cls_dim,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    requires_grad(ema, False)

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
    ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
    ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SRALoss(
        prediction=args.prediction,
        path_type=args.path_type,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        block_out_s=args.block_out_s,
        block_out_t=args.block_out_t,
        t_max=args.t_max,
        loss_type=args.loss_type,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    # Setup dataset:
    train_dataset = HFImgLatentDataset(
        "sdvae-ft-mse-f8d4",
        args.data_dir,
        split="train",
    )


    num_images = len(train_dataset)
    local_batch_size = int(args.batch_size)


    # Create data loaders:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if accelerator.is_main_process:
        logger.info(f"Dataset contains {num_images:,} images ({args.data_dir})")
        logger.info(
            f"Total batch size: {local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume:
    global_step = 0
    epoch_start = -1
    if args.resume_ckpt is not None:
        ckpt = torch.load(
            args.resume_ckpt,
            map_location='cpu',
            weights_only=False,
        )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        epoch_start = ckpt['epoch'] - 1
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    if args.resume_ckpt is None:
        # DDP synchronizes the online model during prepare; initialize every local
        # teacher from that synchronized model rather than its process-local seed.
        update_ema(ema, model, decay=0)

    if accelerator.is_main_process:
        logger.info(f"Starting training experiment: {args.exp_name}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    _, gt_xs, gt_labels = next(iter(train_dataloader))
    gt_xs = gt_xs[:sample_batch_size]
    gt_labels = gt_labels[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
    )
    ys = gt_labels.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)


    last_epoch = epoch_start
    for epoch in range(epoch_start+1, args.epochs):
        last_epoch = epoch

        model.train()
        for raw_image, images_l, y in train_dataloader:
            # save checkpoint (feel free to adjust the frequency)
            if (global_step % args.checkpoint_steps == 0) and global_step > 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/step-{global_step}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")


            # sample and save images (feel free to adjust the frequency)
            if (global_step % args.sample_steps == 0) and global_step > 0:
                from samplers import euler_sampler
                with torch.no_grad():
                    model.eval()
                    samples = euler_sampler(
                        model,
                        xT,
                        ys,
                        num_steps=50,
                        cfg_scale= 4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)

                    samples = vae.decode((samples - latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.

                # Save images locally
                accelerator.wait_for_everyone()
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))

                # Save as grid images
                out_samples = Image.fromarray(array2grid(out_samples))
                gt_samples = Image.fromarray(array2grid(gt_samples))

                if accelerator.is_main_process:
                    base_dir = os.path.join(args.output_dir, args.exp_name)
                    sample_dir = os.path.join(base_dir, "samples")
                    os.makedirs(sample_dir, exist_ok=True)
                    out_samples.save(f"{sample_dir}/samples_step_{global_step}.png")
                    gt_samples.save(f"{sample_dir}/gt_samples_step_{global_step}.png")
                    logger.info(f"Saved samples at step {global_step}")
                model.train()

            x = images_l.to(device)
            y = y.to(device)
            raw_image = raw_image.to(device)
            labels = y

            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                with accelerator.autocast():
                    cls_condition = dino_cls_extractor(raw_image)

            with accelerator.accumulate(model):
                cls_probability = linear_cls_probability(
                    global_step,
                    args.cls_prob_start,
                    args.cls_prob_end,
                    args.cls_prob_decay_steps,
                )
                gen_loss, align_loss, student_cls_present = loss_fn(
                    model,
                    x,
                    ema,
                    labels,
                    cls_condition,
                    cls_probability,
                )
                gen_loss_mean = gen_loss.mean()
                align_loss_raw = align_loss.mean()
                align_loss_mean = args.align_weight * align_loss_raw

                # total loss
                loss = gen_loss_mean + align_loss_mean

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model, decay=args.ema_decay)

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "gen_loss": accelerator.gather(gen_loss_mean).mean().detach().item(),
                "align_loss": accelerator.gather(align_loss_mean).mean().detach().item(),
                "align_loss_raw": accelerator.gather(align_loss_raw).mean().detach().item(),
                "cls_probability": cls_probability,
                "cls_present_rate": accelerator.gather(
                    student_cls_present.float().mean().unsqueeze(0)
                ).mean().detach().item(),
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        # save checkpoint (feel free to adjust the frequency)
        if (epoch+1) % args.checkpoint_epochs == 0:
            if accelerator.is_main_process:
                checkpoint = {
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "ema": ema.state_dict(),
                    "opt": optimizer.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "steps": global_step,
                }
                checkpoint_path = f"{checkpoint_dir}/epoch-{epoch}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")







        if global_step >= args.max_train_steps:
            break

    # The periodic step checkpoint runs before each update, so explicitly save
    # the state that reached max_train_steps (or the last completed epoch).
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "ema": ema.state_dict(),
            "opt": optimizer.state_dict(),
            "args": vars(args),
            "epoch": last_epoch,
            "steps": global_step,
        }
        checkpoint_path = f"{checkpoint_dir}/step-{global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved final checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML file containing training arguments; explicit CLI arguments override it.",
    )

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--sample-steps", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=801)
    parser.add_argument("--checkpoint-steps", type=int, default=50000)
    parser.add_argument("--checkpoint-epochs", type=int, default=200)
    parser.add_argument("--max-train-steps", type=int, default=4100000)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument(
        "--data-dir",
        "--data-dir-train",
        dest="data_dir",
        type=str,
        default="/dev/shm/data",
    )
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)

    # precision
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--ema-decay", default=0.9999, type=float,
                        help="Fixed EMA decay used to update the privileged teacher.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=8)

    # loss
    parser.add_argument("--loss-type", type=str, default="cos", choices=["sml1", "l2", "l1","cos"]) # legacy sml1 (only used for reproducing paper results, but suggest use cosine sim loss)
    parser.add_argument("--cfg-prob", type=float, default=0.1, help="use class-free guidance if > 0")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cos"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"])  # currently we only support v-prediction
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--block-out-s", type=int, default=4)
    parser.add_argument("--block-out-t", type=int, default=8)
    parser.add_argument("--t-max", type=float, default=0.2
                        , help="The max time-distance for teacher-student matching.")
    parser.add_argument("--align-weight", type=float, default=0.5,
                        help="Single weight applied to SRA alignment for every sample.")
    parser.add_argument("--cls-dim", type=int, default=768,
                        help="Dimension of the clean DINO CLS feature.")
    parser.add_argument("--dino-model-name", type=str, default="dinov2_vitb14",
                        help="torch.hub DINOv2 model used only during training.")
    parser.add_argument("--cls-prob-start", type=float, default=1.0,
                        help="Initial probability that a student sample receives clean CLS.")
    parser.add_argument("--cls-prob-end", type=float, default=0.1,
                        help="Student CLS probability after schedule decay.")
    parser.add_argument("--cls-prob-decay-steps", type=int, default=1_000_000,
                        help="Number of optimizer steps for linear CLS probability decay.")
    raw_args = input_args
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(raw_args)

    if config_args.config is not None:
        try:
            with open(config_args.config, encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file) or {}
        except (OSError, yaml.YAMLError) as exc:
            parser.error(f"Could not load config {config_args.config!r}: {exc}")
        if not isinstance(config, dict):
            parser.error("The training config must contain a top-level YAML mapping")

        actions_by_destination = {
            action.dest: action
            for action in parser._actions
            if action.dest != "help"
        }
        config_defaults = {}
        for key, value in config.items():
            if not isinstance(key, str):
                parser.error(f"Config keys must be strings, got {key!r}")
            destination = key.replace("-", "_")
            if destination not in actions_by_destination or destination == "config":
                parser.error(f"Unknown training config key: {key}")
            action = actions_by_destination[destination]
            if isinstance(action, argparse.BooleanOptionalAction):
                if not isinstance(value, bool):
                    parser.error(
                        f"Config value for {key} must be true or false, got {value!r}"
                    )
                converted_value = value
            elif value is None:
                converted_value = None
            elif action.type is not None:
                try:
                    converted_value = action.type(value)
                except (TypeError, ValueError) as exc:
                    parser.error(f"Invalid config value for {key}: {exc}")
            else:
                converted_value = value

            if action.choices is not None and converted_value not in action.choices:
                parser.error(
                    f"Invalid config value for {key}: {converted_value!r}; "
                    f"choose from {list(action.choices)}"
                )
            config_defaults[destination] = converted_value
        parser.set_defaults(**config_defaults)

    args = parser.parse_args(raw_args)

    if not args.exp_name:
        parser.error("--exp-name must be provided on the CLI or in --config")
    if args.model not in SiT_models:
        parser.error(
            "--model must be one of: " + ", ".join(sorted(SiT_models))
        )
    if args.cls_dim <= 0:
        parser.error("--cls-dim must be positive for privileged CLS training")
    if not 0 <= args.cls_prob_end <= args.cls_prob_start <= 1:
        parser.error("CLS probabilities must satisfy 0 <= end <= start <= 1")
    if args.cls_prob_decay_steps <= 0:
        parser.error("--cls-prob-decay-steps must be positive")
    if args.align_weight < 0:
        parser.error("--align-weight must be non-negative")
    if not 0 <= args.cfg_prob <= 1:
        parser.error("--cfg-prob must be in [0, 1]")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must satisfy 0 <= decay < 1")

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
