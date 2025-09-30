#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import ViTModel
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch_dct import dct_2d, idct_2d
from scipy.optimize import differential_evolution


# -------------------------
# Dataset (no disk overwrite)
# -------------------------
class ImageFolderPILDataset(Dataset):
    def __init__(self, root_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp"), resize=(224, 224)):
        self.root_dir = Path(root_dir)
        self.extensions = extensions
        self.resize = resize
        self.image_paths = [
            p for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]
        if not self.image_paths:
            raise ValueError(f"No images found in {root_dir} with extensions {extensions}")

        for path in self.image_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize((224,224), Image.BILINEAR)
            img.save(path)  # overwrite on disk

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.resize is not None:
            img = img.resize(self.resize, Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0  # [H,W,C], float32 0..1
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C,H,W]
        return tensor, str(path)


def get_image_dataloader(root_dir, batch_size=8, num_workers=4, shuffle=False):
    dataset = ImageFolderPILDataset(root_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=lambda batch: (torch.stack([b[0] for b in batch]), [b[1] for b in batch])
    )
    return loader

# -------------------------
# ViT Embedder
# -------------------------
class ViTEmbedder:
    def __init__(self, model_name="google/vit-base-patch16-224", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_embeddings(self, images: torch.Tensor):
        """
        images: Tensor [B, C, H, W] float32 in 0..1 and already resized to 224
        Note: the original notebook expected "no preprocessing"; if you want the exact processor
        from transformers, swap to ViTImageProcessor and use it before calling this function.
        """
        if images.device != torch.device(self.device):
            images = images.to(self.device)
        outputs = self.model(pixel_values=images)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        return embeddings

# -------------------------
# Helpers for saving
# -------------------------

def make_out_dir(root: str, folder_name: str):
    out_dir = Path(root) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _epsilon_to_foldername(epsilon: float) -> str:
    # e.g., 4/255 -> "4_255"
    denom = 255
    num = round(float(epsilon) * denom)
    return f"{num}_{denom}"


def save_adv_batch(batch: torch.Tensor, pth_list, out_dir: Path):
    """
    Save a batch of adv images to out_dir using basename of original paths.
    batch: [B, C, H, W] in 0..1 float or 0..255 uint8
    pth_list: list of original file paths (strings)
    out_dir: pathlib.Path
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = batch.detach().cpu()
    if batch.dtype != torch.uint8:
        img = (batch * 255.0).clamp(0, 255).byte()
    else:
        img = batch

    for i in range(img.shape[0]):
        im = img[i].permute(1, 2, 0).numpy()  # H,W,C
        pil = Image.fromarray(im)
        basename = Path(pth_list[i]).name
        save_path = out_dir / basename
        pil.save(save_path)

# -------------------------
# Configurable loss functions
# -------------------------

def get_loss_fn(name: str, temp: float = 0.5):
    """
    Return a function loss_fn(emb_adv, emb_clean) -> scalar torch.Tensor
    The returned value is a *measure we want to maximize* (higher = better adversarial objective).

    Supported names: 'mse', 'kl', 'cosine', 'l2' (euclidean distance on embeddings)
    """
    name = name.lower()

    if name == "mse":
        def fn(emb_adv, emb_clean):
            return F.mse_loss(emb_adv, emb_clean, reduction='mean')
        return fn

    if name == "l2":
        def fn(emb_adv, emb_clean):
            # mean euclidean distance across batch
            return torch.norm(emb_adv - emb_clean, dim=-1).mean()
        return fn

    if name == "kl":
        def fn(emb_adv, emb_clean):
            # KL divergence between clean distribution and adv distribution (maximize)
            p = F.log_softmax(emb_clean / temp, dim=-1)
            q = F.softmax(emb_adv / temp, dim=-1)
            # use batchmean for stability
            return F.kl_div(p, q, reduction='batchmean')
        return fn

    if name == "cosine":
        def fn(emb_adv, emb_clean):
            # cosine similarity (we want to *minimize* similarity, so we maximize its negative later)
            sim = F.cosine_similarity(emb_adv, emb_clean, dim=-1)
            return sim.mean()
        return fn

    raise ValueError(f"Unknown loss name: {name}. Choose from mse, kl, cosine, l2.")

# -------------------------
# Attacks (FGSM, PGD, Fourier, DCT, One-pixel)
# Each attack uses the provided loss_fn to compute the adversarial objective.
# The attacks follow the previous convention: compute measure = loss_fn(emb_adv, emb_clean)
# then negate it (loss = -measure) and call backward() so gradients maximize the measure.
# -------------------------

def fgsm_kl_attack(model, x_with_path, epsilon, out_folder_root, attack_name="fgsm", loss_fn=None, loss_name="mse", temp=0.5):
    x, pth = x_with_path
    out_folder = make_out_dir(out_folder_root, f"adv_{_epsilon_to_foldername(epsilon)}_{attack_name}_{loss_name}")
    x_adv = x.clone().detach().requires_grad_(True)

    with torch.no_grad():
        emb_clean = model.get_embeddings(x)

    emb_adv = model.get_embeddings(x_adv)

    measure = loss_fn(emb_adv, emb_clean)
    loss = -measure
    loss.backward()

    x_adv = (x_adv + epsilon * x_adv.grad.sign()).detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    save_adv_batch(x_adv, pth, out_folder)
    return x_adv


def pgd_kl_attack(model, x_with_path, epsilon, out_folder_root, attack_name="pgd", alpha=0.01, steps=40, loss_fn=None, loss_name="mse", temp=0.5):
    x, pth = x_with_path
    out_folder = make_out_dir(out_folder_root, f"adv_{_epsilon_to_foldername(epsilon)}_{attack_name}_{loss_name}")
    x_adv = x.clone().detach()

    with torch.no_grad():
        emb_clean = model.get_embeddings(x)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        emb_adv = model.get_embeddings(x_adv)

        measure = loss_fn(emb_adv, emb_clean)
        loss = -measure
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + perturbation, 0.0, 1.0)
            x_adv.grad = None

    x_adv = x_adv.detach()
    save_adv_batch(x_adv, pth, out_folder)
    return x_adv


def fgsm_kl_attack_dct(model, x_with_path, epsilon, out_folder_root=None, attack_name="fgsm_dct", loss_fn=None, loss_name="mse", temp=0.5):
    """
    FGSM-style attack in the DCT domain.
    """
    x, pth = x_with_path

    out_folder = None
    if out_folder_root is not None:
        out_folder = make_out_dir(out_folder_root, f"adv_{_epsilon_to_foldername(epsilon)}_{attack_name}_{loss_name}")

    # Work in DCT domain
    x_adv = dct_2d(x).detach().requires_grad_(True)
    x_adv_pixel = idct_2d(x_adv)

    with torch.no_grad():
        emb_clean = model.get_embeddings(x)

    emb_adv = model.get_embeddings(x_adv_pixel)

    measure = loss_fn(emb_adv, emb_clean)
    loss = -measure
    loss.backward()

    # gradients are in DCT domain; simple FGSM step
    with torch.no_grad():
        grad = x_adv.grad
        if grad is None:
            raise RuntimeError("No gradient computed for DCT FGSM attack")
        x_adv = x_adv + epsilon * torch.sign(grad)

        # project back to pixel domain and clip to valid range
        x_adv_pixel = torch.clamp(idct_2d(x_adv), 0.0, 1.0).detach()

    if out_folder is not None:
        save_adv_batch(batch=x_adv_pixel, pth=pth, out_dir=out_folder)
    else:
        save_adv_batch(batch=x_adv_pixel, pth=pth)

    return x_adv_pixel


def fgsm_kl_attack_fourier(model, x_with_path, epsilon, out_folder_root=None, attack_name="fgsm_fourier", loss_fn=None, loss_name="mse", temp=0.5):
    """
    FGSM-style attack in the Fourier domain.
    """
    x, pth = x_with_path

    out_folder = None
    if out_folder_root is not None:
        out_folder = make_out_dir(out_folder_root, f"adv_{_epsilon_to_foldername(epsilon)}_{attack_name}_{loss_name}")

    # create frequency-domain variable
    x_adv_freq = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1)).detach().requires_grad_(True)

    # pixel-domain view
    x_adv_pixel = torch.fft.ifft2(torch.fft.ifftshift(x_adv_freq, dim=(-2, -1)), dim=(-2, -1)).real

    with torch.no_grad():
        emb_clean = model.get_embeddings(x)

    emb_adv = model.get_embeddings(x_adv_pixel)

    measure = loss_fn(emb_adv, emb_clean)
    loss = -measure
    loss.backward()

    # grad is on the complex tensor x_adv_freq; operate on real/imag parts
    with torch.no_grad():
        grad = x_adv_freq.grad
        if grad is None:
            raise RuntimeError("No gradient computed for Fourier FGSM attack")
        real = x_adv_freq.real + epsilon * torch.sign(grad.real)
        imag = x_adv_freq.imag + epsilon * torch.sign(grad.imag)
        x_adv_freq = torch.complex(real, imag)

        x_adv_pixel = torch.fft.ifft2(torch.fft.ifftshift(x_adv_freq, dim=(-2, -1)), dim=(-2, -1)).real
        x_adv_pixel = torch.clamp(x_adv_pixel, 0.0, 1.0).detach()

    if out_folder is not None:
        save_adv_batch(batch=x_adv_pixel, pth=pth, out_dir=out_folder)
    else:
        save_adv_batch(batch=x_adv_pixel, pth=pth)

    return x_adv_pixel


def pgd_kl_attack_fourier(model, x_with_path, epsilon, out_folder_root, attack_name="pgd_fourier", alpha=0.01, steps=40, loss_fn=None, loss_name="mse", temp=0.5):
    x, pth = x_with_path
    out_folder = make_out_dir(out_folder_root, f"adv_{_epsilon_to_foldername(epsilon)}_{attack_name}_{loss_name}")

    # Work in frequency domain: operate per-channel on 2D FFT
    x_adv_freq = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))

    with torch.no_grad():
        emb_clean = model.get_embeddings(x)

    for _ in range(steps):
        x_adv_freq.requires_grad_(True)
        x_adv_pixel = torch.fft.ifft2(torch.fft.ifftshift(x_adv_freq, dim=(-2, -1)), dim=(-2, -1)).real

        emb_adv = model.get_embeddings(x_adv_pixel)
        measure = loss_fn(emb_adv, emb_clean)
        loss = -measure
        loss.backward()

        with torch.no_grad():
            grad = x_adv_freq.grad
            if grad is None:
                break
            real = x_adv_freq.real + alpha * torch.sign(grad.real)
            imag = x_adv_freq.imag + alpha * torch.sign(grad.imag)
            x_adv_freq = torch.complex(real, imag)

            x_adv_pixel_new = torch.fft.ifft2(torch.fft.ifftshift(x_adv_freq, dim=(-2, -1)), dim=(-2, -1)).real
            perturbation = torch.clamp(x_adv_pixel_new - x, min=-epsilon, max=epsilon)
            x_adv_pixel_clipped = torch.clamp(x + perturbation, 0.0, 1.0)
            x_adv_freq = torch.fft.fftshift(torch.fft.fft2(x_adv_pixel_clipped, dim=(-2, -1)), dim=(-2, -1))

        if x_adv_freq.grad is not None:
            x_adv_freq.grad = None

    x_adv_final = torch.fft.ifft2(torch.fft.ifftshift(x_adv_freq, dim=(-2, -1)), dim=(-2, -1)).real.detach()
    save_adv_batch(x_adv_final, pth, out_folder)
    return x_adv_final


def pgd_kl_attack_dct(model, x_with_path, epsilon, out_folder_root, attack_name="pgd_dct", alpha=0.01, steps=40, loss_fn=None, loss_name="mse", temp=0.5):
    if dct_2d is None or idct_2d is None:
        raise RuntimeError("torch_dct is required for DCT attacks. Install torch_dct or adjust script.")
    x, pth = x_with_path
    out_folder = make_out_dir(out_folder_root, f"adv_{_epsilon_to_foldername(epsilon)}_{attack_name}_{loss_name}")

    x_adv = dct_2d(x)

    with torch.no_grad():
        emb_clean = model.get_embeddings(x)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        x_adv_pixel = idct_2d(x_adv)
        emb_adv = model.get_embeddings(x_adv_pixel)

        measure = loss_fn(emb_adv, emb_clean)
        loss = -measure
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad
            if grad is None:
                break
            x_adv = x_adv + alpha * torch.sign(grad)
            x_adv_pixel_new = idct_2d(x_adv)
            perturbation = torch.clamp(x_adv_pixel_new - x, min=-epsilon, max=epsilon)
            x_adv_pixel_clipped = torch.clamp(x + perturbation, min=0.0, max=1.0)
            x_adv = dct_2d(x_adv_pixel_clipped)

        if x_adv.grad is not None:
            x_adv.grad = None

    x_adv_final = idct_2d(x_adv).detach()
    save_adv_batch(x_adv_final, pth, out_folder)
    return x_adv_final

# One-pixel attack using differential_evolution (optional)
def one_pixel_kl_attack(model, x_with_path, out_folder_root, attack_name="one_pixel", temp=0.5, maxiter=100, popsize=30, loss_fn=None, loss_name="mse"):

    x, pth = x_with_path
    out_folder = make_out_dir(out_folder_root, f"adv_one_pixel_{attack_name}_{loss_name}")

    with torch.no_grad():
        emb_clean = model.get_embeddings(x)

    B, C, H, W = x.shape

    def objective_function(candidate):
        x_adv = x.clone()
        for i in range(B):
            start = i * 5
            xi, yi = int(round(candidate[start])), int(round(candidate[start + 1]))
            r, g, b = candidate[start + 2:start + 5]
            r = float(np.clip(r, 0.0, 1.0))
            g = float(np.clip(g, 0.0, 1.0))
            b = float(np.clip(b, 0.0, 1.0))
            xi = max(0, min(H - 1, xi))
            yi = max(0, min(W - 1, yi))
            x_adv[i, 0, xi, yi] = r
            x_adv[i, 1, xi, yi] = g
            x_adv[i, 2, xi, yi] = b

        emb_adv = model.get_embeddings(x_adv)
        # we want to maximize the measure, but differential_evolution minimizes the objective,
        # so return negative measure here (minimize -measure == maximize measure)
        measure = loss_fn(emb_adv, emb_clean)
        return -measure.item()

    bounds = []
    for i in range(B):
        bounds += [(0, H - 1), (0, W - 1), (0, 1), (0, 1), (0, 1)]

    result = differential_evolution(
        objective_function,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-5,
        workers=1
    )

    # apply best candidate
    x_adv = x.clone()
    for i in range(B):
        cand = result.x[i*5:(i+1)*5]
        xi, yi = int(round(cand[0])), int(round(cand[1]))
        color = torch.tensor(cand[2:5], dtype=x.dtype, device=x.device).view(3, 1, 1)
        xi = max(0, min(H - 1, xi))
        yi = max(0, min(W - 1, yi))
        x_adv[i, :, xi, yi] = color.squeeze()

    save_adv_batch(x_adv, pth, out_folder)
    return x_adv, result

# -------------------------
# Main flow
# -------------------------

def make_attack_sequence():
    # return a list of (callable, kwargs, short_name)
    return [
        # FGSM pixel-domain attacks
        (fgsm_kl_attack, {"epsilon": 4.0/255, "attack_name": "fgsm"}, "fgsm_4_255"),
        (fgsm_kl_attack, {"epsilon": 1.0/255, "attack_name": "fgsm"}, "fgsm_1_255"),

        # FGSM DCT-domain attacks
        (fgsm_kl_attack_dct, {"epsilon": 4.0/255, "attack_name": "fgsm_dct"}, "fgsm_dct_4_255"),
        (fgsm_kl_attack_dct, {"epsilon": 1.0/255, "attack_name": "fgsm_dct"}, "fgsm_dct_1_255"),

        # FGSM Fourier-domain attacks
        (fgsm_kl_attack_fourier, {"epsilon": 4.0/255, "attack_name": "fgsm_fourier"}, "fgsm_fourier_4_255"),
        (fgsm_kl_attack_fourier, {"epsilon": 1.0/255, "attack_name": "fgsm_fourier"}, "fgsm_fourier_1_255"),

        # PGD pixel-domain attacks
        (pgd_kl_attack, {"epsilon": 4.0/255, "alpha": 1.0/255, "steps": 200, "attack_name": "pgd"}, "pgd_4_255"),
        (pgd_kl_attack, {"epsilon": 1.0/255, "alpha": 1.0/255, "steps": 200, "attack_name": "pgd"}, "pgd_1_255"),

        # PGD DCT-domain attacks
        (pgd_kl_attack_dct, {"epsilon": 4.0/255, "alpha": 1.0/255, "steps": 200, "attack_name": "pgd_dct"}, "pgd_dct_4_255"),
        (pgd_kl_attack_dct, {"epsilon": 1.0/255, "alpha": 1.0/255, "steps": 200, "attack_name": "pgd_dct"}, "pgd_dct_1_255"),

        # PGD Fourier-domain attacks
        (pgd_kl_attack_fourier, {"epsilon": 4.0/255, "alpha": 1.0/255, "steps": 200, "attack_name": "pgd_fourier"}, "pgd_fourier_4_255"),
        (pgd_kl_attack_fourier, {"epsilon": 1.0/255, "alpha": 1.0/255, "steps": 200, "attack_name": "pgd_fourier"}, "pgd_fourier_1_255"),

        # One-pixel attack
        (one_pixel_kl_attack, {"attack_name": "one_pixel", "maxiter": 100, "popsize": 30}, "one_pixel"),
    ]


def main(args):
    loader = get_image_dataloader(args.data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    vit = ViTEmbedder(device=args.device, model_name=args.model_name)

    # create loss function according to configuration
    loss_fn = get_loss_fn(args.loss, temp=args.temp)

    attacks = make_attack_sequence()

    for batch in tqdm(loader, desc="Batches"):
        # batch is (tensor [B,C,H,W], [paths])
        for attack_func, kw, short_name in attacks:
            try:
                # inject loss_fn, loss_name and temp into kwargs so each attack can call it
                kw = dict(kw)
                kw.update({"loss_fn": loss_fn, "loss_name": args.loss, "temp": args.temp})
                attack_func(model=vit, x_with_path=batch, out_folder_root=args.out, **kw)
            except Exception as e:
                print(f"Attack {short_name} failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate adversarial examples and save into per-attack folders.")
    parser.add_argument("--data", required=True, help="Path to clean images root")
    parser.add_argument("--out", required=True, help="Output root folder where adv_* folders will be created")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if not provided)")
    parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "kl", "cosine", "l2"], help="Which loss/measure to maximize when crafting adversarial examples")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature used in KL or softmax-based losses")
    args = parser.parse_args()
    main(args)