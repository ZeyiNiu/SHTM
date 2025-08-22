# train_diffusion.py (DDP-safe, single/multi-GPU)
# =========================================================================
#  DDPM-lite learns hi-res residuals in the SAME normalized domain as stage-1.
#  Target: resid_n (5ch); Condition: concat([mu_hat_n(5), era_up_n(13)]).
#  Launch:
#    python3 -m torch.distributed.run --standalone --nproc_per_node=1 train_diffusion.py
#    CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run --standalone --nproc_per_node=2 train_diffusion.py
# =========================================================================

import os, time, traceback
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import DownscaleDataset
from model   import ERA2HiResUNet
from diffusion.diffusion import Unet, GaussianDiffusion

# Global backends
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True

class Args:
    npz_dir      = "/public/home/niuzeyi/down2/traindata"
    reg_ckpt     = "era2hires_unet_best.pth"
    save_dir     = "ckpts_ddpm_lite"

    epochs       = 150
    batch_size   = 1
    lr           = 2e-4
    lr_min       = 1e-5
    weight_decay = 0.0
    grad_clip    = 1
    amp          = True
    num_workers  = 4

    T            = 1000
    sampling_T   = 50
    base_ch      = 128
    image_size   = (521, 721)

def setup_ddp():
    """Safe DDP setup: auto single-process fallback when WORLD_SIZE==1."""
    rank  = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))

    use_cuda = torch.cuda.is_available()

    if world > 1:
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://",
                                timeout=timedelta(hours=5))
        if use_cuda:
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
        else:
            device = torch.device("cpu")
        use_ddp = True
    else:
        # single process fallback: do NOT init DDP
        if use_cuda:
            torch.cuda.set_device(0)
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
        use_ddp = False

    return rank, world, local, device, use_ddp

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()

def is_main(rank): return rank == 0

def main():
    args = Args()
    rank, world, local, device, use_ddp = setup_ddp()

    if is_main(rank):
        print("== DDPM-lite Training (normalized domain) ==")
    print(f"[rank{rank}] world={world} local={local} device={device} cuda={torch.cuda.is_available()}", flush=True)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Dataset / Loader
    ds = DownscaleDataset(args.npz_dir)
    if use_ddp and world > 1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    else:
        sampler = None

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Stage-1 regression (frozen)
    reg_net = ERA2HiResUNet().to(device)
    ckpt = torch.load(args.reg_ckpt, map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt
    reg_net.load_state_dict(state, strict=False)
    reg_net.eval()
    for p in reg_net.parameters():
        p.requires_grad_(False)

    # Denoiser Unet: resid_n is x; cond is passed separately
    # NOTE: If your Unet concatenates x|cond internally, keep channels=23.
    deno = Unet(
        channels=23,
        out_dim=5,
        dim=args.base_ch,
        dim_mults=(1, 2, 4, 8),
        attn_heads=8,
        attn_dim_head=64,
    ).to(device)

    diffusion = GaussianDiffusion(
        model=deno,
        image_size=args.image_size,
        timesteps=args.T,
        sampling_timesteps=args.sampling_T,
        beta_schedule="cosine",
        objective="pred_v",         # or 'pred_noise'
        min_snr_loss_weight=True,
        min_snr_gamma=5,
    ).to(device)

    # Wrap DDP ONLY if world>1; for CPU (gloo) do NOT pass device_ids
    if use_ddp and world > 1:
        if device.type == "cuda":
            diffusion = DDP(diffusion, device_ids=[local], output_device=local,
                            find_unused_parameters=False, broadcast_buffers=False)
        else:
            diffusion = DDP(diffusion, find_unused_parameters=False, broadcast_buffers=False)

    params = diffusion.module.model.parameters() if isinstance(diffusion, DDP) else diffusion.model.parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and (device.type == "cuda"))
    scheduler = CosineAnnealingLR(opt , args.epochs, eta_min=args.lr_min)

    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
                scheduler.step()

            # train mode
            (diffusion.module if isinstance(diffusion, DDP) else diffusion).model.train()

            t0 = time.time()
            loss_sum, nb = 0.0, 0

            for step, (era, hirez) in enumerate(loader, 1):
                era   = era.to(device, non_blocking=True)    # (B,13,209,289) normalized
                hirez = hirez.to(device, non_blocking=True)  # (B,5, 521,721) normalized

                # stage-1 forward (fp32)
                with torch.no_grad():
                    mu_hat_n = reg_net(era)                  # (B,5,521,721)

                # residual (normalized domain)
                resid_n = hirez - mu_hat_n                   # (B,5,521,721)

                # upsample ERA to hi-res
                era_up_n = F.interpolate(era, size=hirez.shape[2:], mode='bilinear', align_corners=False)

                # condition tensor
                cond_in = torch.cat([mu_hat_n, era_up_n], dim=1).contiguous(memory_format=torch.channels_last)  # (B,18,521,721)

                opt.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    with torch.cuda.amp.autocast():
                        loss = (diffusion.module if isinstance(diffusion, DDP) else diffusion)(resid_n, cond=cond_in)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        (diffusion.module.model if isinstance(diffusion, DDP) else diffusion.model).parameters(),
                        args.grad_clip
                    )
                    scaler.step(opt); scaler.update()
                else:
                    loss = (diffusion.module if isinstance(diffusion, DDP) else diffusion)(resid_n, cond=cond_in)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        (diffusion.module.model if isinstance(diffusion, DDP) else diffusion.model).parameters(),
                        args.grad_clip
                    )
                    opt.step()

                # logging (avg across ranks if DDP)
                with torch.no_grad():
                    loss_det = loss.detach()
                    if use_ddp and world > 1:
                        dist.all_reduce(loss_det, op=dist.ReduceOp.AVG)
                    loss_sum += loss_det.item()
                    nb += 1
                    global_step += 1

                if is_main(rank) and (step % 20 == 0):
                    print(f"[Epoch {epoch} | Step {step}/{len(loader)}] loss {loss_det.item():.6f} | gs {global_step}")

            if is_main(rank):
                avg = loss_sum / max(1, nb)
                print(f"Epoch {epoch:03d} done | avg_loss {avg:.6f} | time {time.time()-t0:.1f}s")
                to_save = diffusion.module if isinstance(diffusion, DDP) else diffusion
                save_path = Path(args.save_dir) / f"ddpm_norm_e{epoch:04d}.pth"
                torch.save({
                    "epoch": epoch,
                    "model": to_save.model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": {k: getattr(args, k) for k in dir(args) if not k.startswith('_') and not callable(getattr(args, k))},
                }, save_path)
                print(f"✓ saved {save_path}")

    except Exception:
        if is_main(rank):
            print("Training crashed with exception:")
            traceback.print_exc()
        raise
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main()
