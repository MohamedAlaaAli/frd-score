import os
import csv
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor


# ============================================================
# 🔹 Reproducibility
# ============================================================
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 🔹 ViT Encoder Wrapper (with proper normalization)
# ============================================================
class ViTEmbedder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTModel.from_pretrained(model_name).to(self.device).eval()
        proc = ViTImageProcessor.from_pretrained(model_name)
        mean = torch.tensor(proc.image_mean).view(1, -1, 1, 1).to(self.device)
        std = torch.tensor(proc.image_std).view(1, -1, 1, 1).to(self.device)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, x):
        # x: (B,C,H,W) in [0,1]
        if x.device != self.device:
            x = x.to(self.device)
        x = (x - self._mean) / (self._std + 1e-12)
        outputs = self.model(x)
        return outputs.last_hidden_state[:, 0, :]


# ============================================================
# 🔹 KL Divergence (fitness function)
# ============================================================
def kl_pq_from_embeddings(emb_clean, emb_adv, temp=1.0, eps=1e-12):
    p = F.softmax(emb_clean / temp, dim=-1).clamp(min=eps)
    q = F.softmax(emb_adv / temp, dim=-1).clamp(min=eps)
    kl = (p * (torch.log(p) - torch.log(q))).sum(dim=1)
    return kl


# ============================================================
# 🔹 GenAttack (KL objective, eps-ball + query counting)
# ============================================================
class GenAttackKL:
    def __init__(
        self,
        embedder,
        clean_img,
        img_path,
        pop_size=8,
        mutation_rate=0.08,
        mutation_decay=0.995,
        max_iters=2000,
        eps=4 / 255,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=1.0,
        image_channels=3,
        max_queries=20000,
        patience_gens=300,
        min_improve=1e-4,
        selection_temp=0.3,
    ):
        self.device = device
        self.embedder = embedder.to(self.device).eval()
        self.clean_img = clean_img.unsqueeze(0).to(self.device)  # (1,C,H,W)
        self.img_path = img_path
        self.pop_size = int(pop_size)
        self.mutation_rate = float(mutation_rate)
        self.mutation_decay = float(mutation_decay)
        self.max_iters = int(max_iters)
        self.temperature = float(temperature)
        self.image_channels = int(image_channels)
        self.eps = float(eps)

        _, C, H, W = self.clean_img.shape
        self.C, self.H, self.W = C, H, W

        self.query_count = 0
        self.max_queries = int(max_queries)
        self.patience_gens = int(patience_gens)
        self.min_improve = float(min_improve)
        self.selection_temp = float(selection_temp)

    def _clip_eps_ball(self, candidate):
        clean = self.clean_img.repeat(candidate.shape[0], 1, 1, 1)
        minv = (clean - self.eps).clamp(0.0, 1.0)
        maxv = (clean + self.eps).clamp(0.0, 1.0)
        return torch.max(torch.min(candidate, maxv), minv)

    # def attack(self, save_dir="datasets/adv", log_csv="attack_log.csv"):
    #     # ====================================================
    #     # Encode clean image
    #     # ====================================================
    #     with torch.no_grad():
    #         clean_for_embed = self.clean_img
    #         if self.C == 1:
    #             clean_for_embed = clean_for_embed.repeat(1, 3, 1, 1)
    #         emb_clean = self.embedder(clean_for_embed).detach()

    #     # ====================================================
    #     # Init population
    #     # ====================================================
    #     base = self.clean_img.repeat(self.pop_size, 1, 1, 1)
    #     init_noise = (torch.rand_like(base) * 2.0 - 1.0) * self.eps
    #     population = (base + init_noise).clamp(0.0, 1.0)
    #     population = self._clip_eps_ball(population)
    #     population[0] = self.clean_img.clone()  # elite = clean

    #     best_adv = self.clean_img.squeeze(0).detach().clone()
    #     best_score = -float("inf")

    #     no_improve = 0
    #     gen = 0

    #     os.makedirs(save_dir, exist_ok=True)
    #     log_path = os.path.join(save_dir, log_csv)
    #     write_header = not os.path.exists(log_path)
    #     csvfile = open(log_path, "a", newline="")
    #     writer = csv.writer(csvfile)
    #     if write_header:
    #         writer.writerow(["image", "gen", "score", "queries"])

    #     while (
    #         gen < self.max_iters
    #         and self.query_count < self.max_queries
    #         and no_improve < self.patience_gens
    #     ):
    #         gen += 1

    #         pop_eval = population
    #         if self.C == 1:
    #             pop_eval = pop_eval.repeat(1, 3, 1, 1)

    #         with torch.no_grad():
    #             emb_pop = self.embedder(pop_eval.contiguous().float())
    #             self.query_count += pop_eval.shape[0]

    #         fitness = kl_pq_from_embeddings(
    #             emb_clean.repeat(self.pop_size, 1), emb_pop, temp=self.temperature
    #         )for

    #         best_idx = torch.argmax(fitness).item()
    #         best_val = fitness[best_idx].item()
    #         if best_val > best_score + self.min_improve:
    #             best_score = best_val
    #             best_adv = population[best_idx].detach().clone()
    #             no_improve = 0
    #         else:
    #             no_improve += 1

    #         # Save logs + intermediate images
    #         if gen % 200 == 0 or gen == 1:
    #             filename = os.path.basename(self.img_path)
    #             writer.writerow([filename, gen, best_score, self.query_count])
    #             csvfile.flush()
    #             save_path = os.path.join(save_dir, f"{filename}_gen{gen}.png")
    #             self._save_tensor_as_img(best_adv, save_path)

    #         # Selection (prob ∝ fitness with temperature)
    #         sel_logits = (fitness - fitness.max()) / (self.selection_temp + 1e-12)
    #         sel_probs = torch.softmax(sel_logits, dim=0)
    #         if not torch.isfinite(sel_probs).all():
    #             sel_probs = torch.ones_like(sel_probs) / float(self.pop_size)
    #         parent_idx = torch.multinomial(sel_probs, self.pop_size, replacement=True)
    #         parents = population[parent_idx]

    #         # Crossover
    #         crossover_mask = torch.rand_like(parents) > 0.5
    #         crossover = torch.where(crossover_mask, parents, parents.flip(0))

    #         # Mutation (decaying noise)
    #         noise_std = 0.5 * self.eps * (0.5 + 0.5 * (1 - gen / self.max_iters))
    #         mutation_mask = (torch.rand_like(crossover) < self.mutation_rate).float()
    #         noise = torch.randn_like(crossover) * noise_std
    #         mutants = torch.clamp(crossover + mutation_mask * noise, 0, 1)
    #         mutants = self._clip_eps_ball(mutants)

    #         mutants[0] = best_adv
    #         population = mutants
    #         self.mutation_rate = max(1e-6, self.mutation_rate * self.mutation_decay)

    #     csvfile.close()

    #     final = best_adv.detach().cpu().numpy()
    #     if final.ndim == 3 and final.shape[0] in (1, 3):
    #         final = np.transpose(final, (1, 2, 0))
    #     return final, best_score, self.query_count
    def attack(self, save_dir="datasets/adv", log_csv="attack_log.csv"):
        """
        Run the genetic attack. Shows tqdm bar over generations.
        Saves only the final best adversarial example.
        """
        # ====================================================
        # Encode clean image
        # ====================================================
        with torch.no_grad():
            clean_for_embed = self.clean_img
            if self.C == 1:
                clean_for_embed = clean_for_embed.repeat(1, 3, 1, 1)
            emb_clean = self.embedder(clean_for_embed).detach()

        # ====================================================
        # Init population
        # ====================================================
        base = self.clean_img.repeat(self.pop_size, 1, 1, 1)
        init_noise = (torch.rand_like(base) * 2.0 - 1.0) * self.eps
        population = (base + init_noise).clamp(0.0, 1.0)
        population = self._clip_eps_ball(population)
        population[0] = self.clean_img.clone()

        best_adv = self.clean_img.squeeze(0).detach().clone()
        best_score = -float("inf")
        no_improve = 0

        # prepare logging
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, log_csv)
        write_header = not os.path.exists(log_path)
        csvfile = open(log_path, "a", newline="")
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["image", "gen", "score", "queries"])

        # ====================================================
        # Main optimization loop
        # ====================================================
        basename = os.path.basename(self.img_path) if self.img_path is not None else "image"
        pbar = tqdm(range(1, self.max_iters + 1),
            desc=f"GenAttack:{basename}",
            ncols=100,
            leave=True)  # keeps one clean bar after finishing
        for gen in pbar:
            if self.query_count >= self.max_queries:
                break
            if no_improve >= self.patience_gens:
                break

            pop_eval = population
            if self.C == 1:
                pop_eval = pop_eval.repeat(1, 3, 1, 1)

            with torch.no_grad():
                emb_pop = self.embedder(pop_eval.contiguous().float())
                self.query_count += int(pop_eval.shape[0])

            fitness = kl_pq_from_embeddings(
                emb_clean.repeat(self.pop_size, 1), emb_pop, temp=self.temperature
            )

            best_idx = int(torch.argmax(fitness).item())
            best_val = float(fitness[best_idx].item())
            if best_val > best_score + self.min_improve:
                best_score = best_val
                best_adv = population[best_idx].detach().clone()
                no_improve = 0
            else:
                no_improve += 1

            # Selection
            sel_logits = (fitness - fitness.max()) / (self.selection_temp + 1e-12)
            sel_probs = torch.softmax(sel_logits, dim=0)
            if not torch.isfinite(sel_probs).all():
                sel_probs = torch.ones_like(sel_probs) / float(self.pop_size)
            parent_idx = torch.multinomial(sel_probs, self.pop_size, replacement=True)
            parents = population[parent_idx]

            # Crossover
            crossover_mask = torch.rand_like(parents) > 0.5
            crossover = torch.where(crossover_mask, parents, parents.flip(0))

            # Mutation
            noise_std = 0.5 * self.eps * (0.5 + 0.5 * (1 - gen / float(self.max_iters)))
            mutation_mask = (torch.rand_like(crossover) < self.mutation_rate).float()
            noise = torch.randn_like(crossover) * noise_std
            mutants = torch.clamp(crossover + mutation_mask * noise, 0, 1)
            mutants = self._clip_eps_ball(mutants)

            mutants[0] = best_adv
            population = mutants
            self.mutation_rate = max(1e-6, self.mutation_rate * self.mutation_decay)

            # live metrics
            pbar.set_postfix({"KL": f"{best_score:.4f}", "Q": f"{self.query_count}", "gen": gen})

        pbar.close()
        csvfile.close()

        # ====================================================
        # Save only final best
        # ====================================================
        filename = os.path.basename(self.img_path)
        save_path = os.path.join(save_dir, filename)
        self._save_tensor_as_img(best_adv, save_path)

        # log final row
        with open(log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, gen, best_score, self.query_count])

        final = best_adv.detach().cpu().numpy()
        if final.ndim == 3 and final.shape[0] in (1, 3):
            final = np.transpose(final, (1, 2, 0))
        return final, best_score, self.query_count


    def _save_tensor_as_img(self, tensor, path):
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)


# ============================================================
# 🔹 Dataset Loader
# ============================================================
class ImageFolderPILDataset(torch.utils.data.Dataset):
    def __init__(self, folder, image_size=(224, 224)):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transforms.Compose(
            [transforms.Resize(image_size), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        return img_t, path


# ============================================================
# 🔹 Save Final Adversarial Image
# ============================================================
def save_adv_batch(batch, save_dir="datasets/adv", pth=None):
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(batch, np.ndarray):
        batch = [batch]
    if isinstance(pth, str):
        pth = [pth]

    for img, p in zip(batch, pth):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(2)
        filename = os.path.basename(p)
        file_path = os.path.join(save_dir, filename)
        Image.fromarray(img).save(file_path)


# ============================================================
# 🔹 Main
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ImageFolderPILDataset("datasets/clean/busi", image_size=(224, 224))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    encoder = ViTEmbedder()

    for clean_img, path in loader:
        clean_img = clean_img.squeeze(0)
        attacker = GenAttackKL(
            encoder,
            clean_img,
            path[0],
            # recommended defaults
            pop_size=8,
            max_iters=2000,
            mutation_rate=0.08,
            mutation_decay=0.995,
            eps=4 / 255,
            temperature=1.0,
            image_channels=3,
            max_queries=20000,
            patience_gens=300,
            selection_temp=0.5,
        )

        adv_img, best_score, queries = attacker.attack()
        save_adv_batch(batch=adv_img, pth=path)

        print(
            f"Saved adversarial example for {path}, "
            f"KL={best_score:.4f}, Queries={queries}"
        )


if __name__ == "__main__":
    main()
