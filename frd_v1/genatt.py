# import os
# import csv
# import numpy as np
# from tqdm import tqdm
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from transformers import ViTModel, ViTImageProcessor
# from compute_frd import get_distance_fn

# # ============================================================
# # 🔹 Reproducibility
# # ============================================================
# seed = 1234
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)


# # ============================================================
# # 🔹 ViT Encoder Wrapper (with proper normalization)
# # ============================================================
# class ViTEmbedder(nn.Module):
#     def __init__(self, model_name="google/vit-base-patch16-224", device=None):
#         super().__init__()
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = ViTModel.from_pretrained(model_name).to(self.device).eval()
#         proc = ViTImageProcessor.from_pretrained(model_name)
#         mean = torch.tensor(proc.image_mean).view(1, -1, 1, 1).to(self.device)
#         std = torch.tensor(proc.image_std).view(1, -1, 1, 1).to(self.device)
#         self.register_buffer("_mean", mean)
#         self.register_buffer("_std", std)

#     def forward(self, x):
#         # x: (B,C,H,W) in [0,1]
#         if x.device != self.device:
#             x = x.to(self.device)
#         x = (x - self._mean) / (self._std + 1e-12)
#         outputs = self.model(x)
#         return outputs.last_hidden_state[:, 0, :]


# # ============================================================
# # 🔹 KL Divergence (fitness function)
# # ============================================================
# def kl_pq_from_embeddings(emb_clean, emb_adv, temp=1.0, eps=1e-12):
#     p = F.softmax(emb_clean / temp, dim=-1).clamp(min=eps)
#     q = F.softmax(emb_adv / temp, dim=-1).clamp(min=eps)
#     kl = (p * (torch.log(p) - torch.log(q))).sum(dim=1)
#     return kl

# def l2(emb_clean, emb_adv):
#     # L2 distance per sample across the last dimension
#     diff = emb_clean - emb_adv
#     return torch.norm(diff, p=2, dim=1)

# # ============================================================
# # 🔹 GenAttack (KL objective, eps-ball + query counting)
# # ============================================================
# class GenAttackKL:
#     def __init__(
#         self,
#         embedder,
#         clean_img,
#         img_path,
#         pop_size=8,
#         mutation_rate=0.08,
#         mutation_decay=0.995,
#         max_iters=2000,
#         eps=4 / 255,
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         temperature=1.0,
#         image_channels=3,
#         max_queries=20000,
#         patience_gens=300,
#         min_improve=1e-4,
#         selection_temp=0.3,
#     ):
#         self.device = device
#         self.embedder = embedder.to(self.device).eval()
#         self.clean_img = clean_img.unsqueeze(0).to(self.device)  # (1,C,H,W)
#         self.img_path = img_path
#         self.pop_size = int(pop_size)
#         self.mutation_rate = float(mutation_rate)
#         self.mutation_decay = float(mutation_decay)
#         self.max_iters = int(max_iters)
#         self.temperature = float(temperature)
#         self.image_channels = int(image_channels)
#         self.eps = float(eps)

#         _, C, H, W = self.clean_img.shape
#         self.C, self.H, self.W = C, H, W

#         self.query_count = 0
#         self.max_queries = int(max_queries)
#         self.patience_gens = int(patience_gens)
#         self.min_improve = float(min_improve)
#         self.selection_temp = float(selection_temp)

#     def _clip_eps_ball(self, candidate):
#         clean = self.clean_img.repeat(candidate.shape[0], 1, 1, 1)
#         minv = (clean - self.eps).clamp(0.0, 1.0)
#         maxv = (clean + self.eps).clamp(0.0, 1.0)
#         return torch.max(torch.min(candidate, maxv), minv)

#     # def attack(self, save_dir="datasets/adv", log_csv="attack_log.csv"):
#     #     # ====================================================
#     #     # Encode clean image
#     #     # ====================================================
#     #     with torch.no_grad():
#     #         clean_for_embed = self.clean_img
#     #         if self.C == 1:
#     #             clean_for_embed = clean_for_embed.repeat(1, 3, 1, 1)
#     #         emb_clean = self.embedder(clean_for_embed).detach()

#     #     # ====================================================
#     #     # Init population
#     #     # ====================================================
#     #     base = self.clean_img.repeat(self.pop_size, 1, 1, 1)
#     #     init_noise = (torch.rand_like(base) * 2.0 - 1.0) * self.eps
#     #     population = (base + init_noise).clamp(0.0, 1.0)
#     #     population = self._clip_eps_ball(population)
#     #     population[0] = self.clean_img.clone()  # elite = clean

#     #     best_adv = self.clean_img.squeeze(0).detach().clone()
#     #     best_score = -float("inf")

#     #     no_improve = 0
#     #     gen = 0

#     #     os.makedirs(save_dir, exist_ok=True)
#     #     log_path = os.path.join(save_dir, log_csv)
#     #     write_header = not os.path.exists(log_path)
#     #     csvfile = open(log_path, "a", newline="")
#     #     writer = csv.writer(csvfile)
#     #     if write_header:
#     #         writer.writerow(["image", "gen", "score", "queries"])

#     #     while (
#     #         gen < self.max_iters
#     #         and self.query_count < self.max_queries
#     #         and no_improve < self.patience_gens
#     #     ):
#     #         gen += 1

#     #         pop_eval = population
#     #         if self.C == 1:
#     #             pop_eval = pop_eval.repeat(1, 3, 1, 1)

#     #         with torch.no_grad():
#     #             emb_pop = self.embedder(pop_eval.contiguous().float())
#     #             self.query_count += pop_eval.shape[0]

#     #         fitness = kl_pq_from_embeddings(
#     #             emb_clean.repeat(self.pop_size, 1), emb_pop, temp=self.temperature
#     #         )for

#     #         best_idx = torch.argmax(fitness).item()
#     #         best_val = fitness[best_idx].item()
#     #         if best_val > best_score + self.min_improve:
#     #             best_score = best_val
#     #             best_adv = population[best_idx].detach().clone()
#     #             no_improve = 0
#     #         else:
#     #             no_improve += 1

#     #         # Save logs + intermediate images
#     #         if gen % 200 == 0 or gen == 1:
#     #             filename = os.path.basename(self.img_path)
#     #             writer.writerow([filename, gen, best_score, self.query_count])
#     #             csvfile.flush()
#     #             save_path = os.path.join(save_dir, f"{filename}_gen{gen}.png")
#     #             self._save_tensor_as_img(best_adv, save_path)

#     #         # Selection (prob ∝ fitness with temperature)
#     #         sel_logits = (fitness - fitness.max()) / (self.selection_temp + 1e-12)
#     #         sel_probs = torch.softmax(sel_logits, dim=0)
#     #         if not torch.isfinite(sel_probs).all():
#     #             sel_probs = torch.ones_like(sel_probs) / float(self.pop_size)
#     #         parent_idx = torch.multinomial(sel_probs, self.pop_size, replacement=True)
#     #         parents = population[parent_idx]

#     #         # Crossover
#     #         crossover_mask = torch.rand_like(parents) > 0.5
#     #         crossover = torch.where(crossover_mask, parents, parents.flip(0))

#     #         # Mutation (decaying noise)
#     #         noise_std = 0.5 * self.eps * (0.5 + 0.5 * (1 - gen / self.max_iters))
#     #         mutation_mask = (torch.rand_like(crossover) < self.mutation_rate).float()
#     #         noise = torch.randn_like(crossover) * noise_std
#     #         mutants = torch.clamp(crossover + mutation_mask * noise, 0, 1)
#     #         mutants = self._clip_eps_ball(mutants)

#     #         mutants[0] = best_adv
#     #         population = mutants
#     #         self.mutation_rate = max(1e-6, self.mutation_rate * self.mutation_decay)

#     #     csvfile.close()

#     #     final = best_adv.detach().cpu().numpy()
#     #     if final.ndim == 3 and final.shape[0] in (1, 3):
#     #         final = np.transpose(final, (1, 2, 0))
#     #     return final, best_score, self.query_count
#     def attack(self, save_dir="advw", log_csv="attack_log.csv"):
#         """
#         Run the genetic attack. Shows tqdm bar over generations.
#         Saves only the final best adversarial example.
#         """
#         # ====================================================
#         # Encode clean image
#         # ====================================================
#         with torch.no_grad():
#             clean_for_embed = self.clean_img
#             if self.C == 1:
#                 clean_for_embed = clean_for_embed.repeat(1, 3, 1, 1)
#             emb_clean = self.embedder(clean_for_embed).detach()

#         # ====================================================
#         # Init population
#         # ====================================================
#         base = self.clean_img.repeat(self.pop_size, 1, 1, 1)
#         init_noise = (torch.rand_like(base) * 2.0 - 1.0) * self.eps
#         population = (base + init_noise).clamp(0.0, 1.0)
#         population = self._clip_eps_ball(population)
#         population[0] = self.clean_img.clone()

#         best_adv = self.clean_img.squeeze(0).detach().clone()
#         best_score = -float("inf")
#         no_improve = 0

#         # prepare logging
#         os.makedirs(save_dir, exist_ok=True)
#         log_path = os.path.join(save_dir, log_csv)
#         write_header = not os.path.exists(log_path)
#         csvfile = open(log_path, "a", newline="")
#         writer = csv.writer(csvfile)
#         if write_header:
#             writer.writerow(["image", "gen", "score", "queries"])

#         # ====================================================
#         # Main optimization loop
#         # ====================================================
#         basename = os.path.basename(self.img_path) if self.img_path is not None else "image"
#         pbar = tqdm(range(1, self.max_iters + 1),
#             desc=f"GenAttack:{basename}",
#             ncols=100,
#             leave=True)  # keeps one clean bar after finishing
#         for gen in pbar:
#             if self.query_count >= self.max_queries:
#                 break
#             if no_improve >= self.patience_gens:
#                 break

#             pop_eval = population
#             if self.C == 1:
#                 pop_eval = pop_eval.repeat(1, 3, 1, 1)

#             with torch.no_grad():
#                 emb_pop = self.embedder(pop_eval.contiguous().float())
#                 self.query_count += int(pop_eval.shape[0])

#             fitness = kl_pq_from_embeddings(
#                 emb_clean.repeat(self.pop_size, 1), emb_pop, temp=self.temperature
#             )
#             #fitness = l2(emb_clean.repeat(self.pop_size, 1), emb_pop)

#             best_idx = int(torch.argmax(fitness).item())
#             best_val = float(fitness[best_idx].item())
#             if best_val > best_score + self.min_improve:
#                 best_score = best_val
#                 best_adv = population[best_idx].detach().clone()
#                 no_improve = 0
#             else:
#                 no_improve += 1

#             # Selection
#             sel_logits = (fitness - fitness.max()) / (self.selection_temp + 1e-12)
#             sel_probs = torch.softmax(sel_logits, dim=0)
#             if not torch.isfinite(sel_probs).all():
#                 sel_probs = torch.ones_like(sel_probs) / float(self.pop_size)
#             parent_idx = torch.multinomial(sel_probs, self.pop_size, replacement=True)
#             parents = population[parent_idx]

#             # Crossover
#             crossover_mask = torch.rand_like(parents) > 0.5
#             crossover = torch.where(crossover_mask, parents, parents.flip(0))

#             # Mutation
#             noise_std = 0.5 * self.eps * (0.5 + 0.5 * (1 - gen / float(self.max_iters)))
#             mutation_mask = (torch.rand_like(crossover) < self.mutation_rate).float()
#             noise = torch.randn_like(crossover) * noise_std
#             mutants = torch.clamp(crossover + mutation_mask * noise, 0, 1)
#             mutants = self._clip_eps_ball(mutants)

#             mutants[0] = best_adv
#             population = mutants
#             self.mutation_rate = max(1e-6, self.mutation_rate * self.mutation_decay)

#             # live metrics
#             pbar.set_postfix({"KL": f"{best_score:.4f}", "Q": f"{self.query_count}", "gen": gen})

#         pbar.close()
#         csvfile.close()

#         # ====================================================
#         # Save only final best
#         # ====================================================
#         filename = os.path.basename(self.img_path)
#         save_path = os.path.join(save_dir, filename)
#         self._save_tensor_as_img(best_adv, save_path)

#         # log final row
#         with open(log_path, "a", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow([filename, gen, best_score, self.query_count])

#         final = best_adv.detach().cpu().numpy()
#         if final.ndim == 3 and final.shape[0] in (1, 3):
#             final = np.transpose(final, (1, 2, 0))
#         return final, best_score, self.query_count


#     def _save_tensor_as_img(self, tensor, path):
#         arr = tensor.detach().cpu().numpy()
#         if arr.ndim == 3 and arr.shape[0] in (1, 3):
#             arr = np.transpose(arr, (1, 2, 0))
#         arr = np.clip(arr, 0, 1)
#         arr = (arr * 255).astype(np.uint8)
#         Image.fromarray(arr).save(path)


# # ============================================================
# # 🔹 Dataset Loader
# # ============================================================
# class ImageFolderPILDataset(torch.utils.data.Dataset):
#     def __init__(self, folder, image_size=(224, 224)):
#         self.paths = [
#             os.path.join(folder, f)
#             for f in os.listdir(folder)
#             if f.lower().endswith((".png", ".jpg", ".jpeg"))
#         ]
#         self.transform = transforms.Compose(
#             [transforms.Resize(image_size), transforms.ToTensor()]
#         )

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         path = self.paths[idx]
#         img = Image.open(path).convert("RGB")
#         img_t = self.transform(img)
#         return img_t, path


# # ============================================================
# # 🔹 Save Final Adversarial Image
# # ============================================================
# def save_adv_batch(batch, save_dir="contamin", pth=None):
#     os.makedirs(save_dir, exist_ok=True)
#     if isinstance(batch, np.ndarray):
#         batch = [batch]
#     if isinstance(pth, str):
#         pth = [pth]

#     for img, p in zip(batch, pth):
#         if isinstance(img, torch.Tensor):
#             img = img.detach().cpu().numpy()
#         if img.ndim == 3 and img.shape[0] in (1, 3):
#             img = np.transpose(img, (1, 2, 0))
#         img = np.clip(img, 0, 1)
#         img = (img * 255).astype(np.uint8)
#         if img.ndim == 3 and img.shape[2] == 1:
#             img = img.squeeze(2)
#         filename = os.path.basename(p)
#         file_path = os.path.join(save_dir, filename)
#         Image.fromarray(img).save(file_path)


# # ============================================================
# # 🔹 Main
# # ============================================================
# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     dataset = ImageFolderPILDataset("clean", image_size=(224, 224))
#     loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

#     encoder = ViTEmbedder()
#     from compute_frd import get_distance_fn
#     get_distance_fn()
#     for clean_img, path in loader:
#         clean_img = clean_img.squeeze(0)
#         attacker = GenAttackKL(
#             encoder,
#             clean_img,
#             path[0],
#             # recommended defaults
#             pop_size=8,
#             max_iters=2000,
#             mutation_rate=0.08,
#             mutation_decay=0.995,
#             eps=4 / 255,
#             temperature=1.0,
#             image_channels=3,
#             max_queries=20000,
#             patience_gens=300,
#             selection_temp=0.5,
#         )

#         adv_img, best_score, queries = attacker.attack()
#         save_adv_batch(batch=adv_img, pth=path)

#         print(
#             f"Saved adversarial example for {path}, "
#             f"KL={best_score:.4f}, Queries={queries}"
#         )


# if __name__ == "__main__":
#     main()



# import os
# import csv
# import numpy as np
# from tqdm import tqdm
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from transformers import ViTModel, ViTImageProcessor
# from compute_frd import get_distance_fn

# # ============================================================
# # 🔹 Reproducibility
# # ============================================================
# seed = 1234
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)


# # penalty: only apply to positive constraint violations
# def default_penalty(h, rho, mu):
#     # h can be scalar or numpy float; ensure float then clamp
#     h_pos = max(0.0, float(h))
#     return mu * h_pos + 0.5 * rho * (h_pos ** 2)


# # ============================================================
# # 🔹 ViT Encoder Wrapper
# # ============================================================
# class ViTEmbedder(nn.Module):
#     def __init__(self, model_name="google/vit-base-patch16-224", device=None):
#         super().__init__()
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = ViTModel.from_pretrained(model_name).to(self.device).eval()
#         proc = ViTImageProcessor.from_pretrained(model_name)
#         mean = torch.tensor(proc.image_mean).view(1, -1, 1, 1).to(self.device)
#         std = torch.tensor(proc.image_std).view(1, -1, 1, 1).to(self.device)
#         self.register_buffer("_mean", mean)
#         self.register_buffer("_std", std)

#     def forward(self, x):
#         if x.device != self.device:
#             x = x.to(self.device)
#         x = (x - self._mean) / (self._std + 1e-12)
#         outputs = self.model(x)
#         return outputs.last_hidden_state[:, 0, :]

# # ============================================================
# # 🔹 KL Divergence
# # ============================================================
# def kl_pq_from_embeddings(emb_clean, emb_adv, temp=1.0, eps=1e-12):
#     p = F.softmax(emb_clean / temp, dim=-1).clamp(min=eps)
#     q = F.softmax(emb_adv / temp, dim=-1).clamp(min=eps)
#     kl = (p * (torch.log(p) - torch.log(q))).sum(dim=1)
#     return kl

# # ============================================================
# # 🔹 L2 Divergence
# # ============================================================
# def l2_from_embeddings(emb_clean, emb_adv, eps=1e-12, normalize=True):
#     if normalize:
#         emb_clean = F.normalize(emb_clean, p=2, dim=1)
#         emb_adv = F.normalize(emb_adv, p=2, dim=1)
#     diff = emb_clean - emb_adv
#     l2 = torch.sqrt((diff ** 2).sum(dim=1) + eps)
#     return l2


# # ============================================================
# # 🔹 BlackBoxALMA Class
# # ============================================================
# class BlackBoxALMA:
#     def __init__(
#         self,
#         model: callable,
#         clean_img: torch.Tensor,
#         clean_batch: torch.Tensor,
#         get_distance_fn: callable,
#         tau: float,
#         divergence: str = 'kl',
#         penalty: callable = default_penalty,
#         μ_init: float = 1.0,
#         ρ_init: float = 0.1,
#         check_steps: int = 10,
#         τ: float = 0.95,
#         γ: float = 1.2,
#         α: float = 0.9,
#         tolerance: float = 1e-4,
#     ):
#         self.model = model
#         self.clean_img = clean_img
#         self.frd = get_distance_fn(clean_batch)
#         self.tau = tau
#         self.divergence = divergence
#         self.penalty = penalty
#         self.μ = μ_init
#         self.ρ = ρ_init
#         self.check_steps = check_steps
#         self.τ = τ
#         self.γ = γ
#         self.α = α
#         self.tolerance = tolerance
#         self.step_count = 0
#         self.prev_mean_h = None
#         self.embed_clean = self.model(clean_img)

#     def _compute_divergence(self, embed_adv: torch.Tensor) -> float:
#         if self.divergence == 'l2':
#             l2 = l2_from_embeddings(self.embed_clean, embed_adv)
#             return l2.item()
#         else:
#             raise ValueError(f"Unknown divergence type: {self.divergence}")


#     def _compute_constraint(self, adv_img: torch.Tensor) -> float:
#         return self.frd(adv_img) - self.tau

#     def attack(self, δ: torch.Tensor, embed_adv: torch.Tensor) -> tuple[torch.Tensor, float]:
#         adv_img = (self.clean_img + δ).clamp(0, 1)
#         divergence = self._compute_divergence(embed_adv)
#         h = self._compute_constraint(adv_img)
#         penalty = self.penalty(h, self.ρ, self.μ)
#         fitness = divergence - penalty   # higher = better
#         return adv_img, fitness


#     def update(self, hs: list[float]) -> None:
#         self.step_count += 1
#         mean_h = np.mean(hs)
#         new_μ = self.μ + self.ρ * mean_h
#         self.μ = self.α * self.μ + (1 - self.α) * new_μ
#         self.μ = max(1e-6, min(self.μ, 1e12))
#         if (self.step_count % self.check_steps == 0 and self.prev_mean_h is not None and
#             abs(mean_h) >= self.τ * abs(self.prev_mean_h)):
#             self.ρ *= self.γ
#         self.prev_mean_h = mean_h

#     def is_adversarial(self, adv_img: torch.Tensor) -> bool:
#         embed_adv = self.model(adv_img)
#         kl = self._compute_divergence(embed_adv)
#         return kl > 0.1

# # ============================================================
# # 🔹 GenAttackKL (Modified with ALMA)
# # ============================================================
# class GenAttackKL:
#     def __init__(
#         self,
#         embedder,
#         clean_img,
#         img_path,
#         dataset,  # Added: Full dataset for clean_batch sampling
#         tau=0.1,  # Target FRD
#         pop_size=8,
#         mutation_rate=0.08,
#         mutation_decay=0.995,
#         max_iters=2000,
#         eps=4 / 255,
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         temperature=1.0,
#         image_channels=3,
#         max_queries=20000,
#         patience_gens=300,
#         min_improve=1e-5,
#         selection_temp=0.3,
#     ):
#         self.device = device
#         self.embedder = embedder.to(self.device).eval()
#         self.clean_img = clean_img.unsqueeze(0).to(self.device)
#         self.img_path = img_path
#         self.dataset = dataset  # For sampling clean_batch
#         self.tau = tau
#         self.pop_size = int(pop_size)
#         self.mutation_rate = float(mutation_rate)
#         self.mutation_decay = float(mutation_decay)
#         self.max_iters = int(max_iters)
#         self.temperature = float(temperature)
#         self.image_channels = int(image_channels)
#         self.eps = float(eps)
#         _, self.C, self.H, self.W = self.clean_img.shape
#         self.query_count = 0
#         self.max_queries = int(max_queries)
#         self.patience_gens = int(patience_gens)
#         self.min_improve = float(min_improve)
#         self.selection_temp = float(selection_temp)

#     def _clip_eps_ball(self, candidate):
#         clean = self.clean_img.repeat(candidate.shape[0], 1, 1, 1)
#         minv = (clean - self.eps).clamp(0.0, 1.0)
#         maxv = (clean + self.eps).clamp(0.0, 1.0)
#         return torch.max(torch.min(candidate, maxv), minv)

#     def _sample_clean_batch(self, batch_size=12):
#         """Return all images in the dataset as a batch, up to batch_size."""
#         indices = list(range(len(self.dataset)))
#         # Take all available images, limited to batch_size
#         selected_indices = indices[:min(len(indices), batch_size)]
#         batch_imgs = [self.dataset[i][0] for i in selected_indices]
#         return torch.stack(batch_imgs).to(self.device)

#     def attack(self, save_dir="advw", log_csv="attack_log.csv"):
#         # Sample clean_batch
#         clean_batch = self._sample_clean_batch(12)
        
#         # Init ALMA
#         alma = BlackBoxALMA(
#             model=self.embedder,
#             clean_img=self.clean_img,
#             clean_batch=clean_batch,
#             get_distance_fn=get_distance_fn,
#             tau=self.tau,
#             divergence='l2',
#         )

#         # Encode clean (for KL)
#         with torch.no_grad():
#             clean_for_embed = self.clean_img
#             if self.C == 1:
#                 clean_for_embed = clean_for_embed.repeat(1, 3, 1, 1)
#             emb_clean = self.embedder(clean_for_embed).detach()

#         # Init population
#         base = self.clean_img.repeat(self.pop_size, 1, 1, 1)
#         init_noise = (torch.rand_like(base) * 2.0 - 1.0) * self.eps
#         population = (base + init_noise).clamp(0.0, 1.0)
#         population = self._clip_eps_ball(population)
#         population[0] = self.clean_img.clone()

#         best_adv = self.clean_img.squeeze(0).detach().clone()
#         best_score = -float("inf")
#         no_improve = 0

#         # Logging
#         os.makedirs(save_dir, exist_ok=True)
#         log_path = os.path.join(save_dir, log_csv)
#         write_header = not os.path.exists(log_path)
#         csvfile = open(log_path, "a", newline="")
#         writer = csv.writer(csvfile)
#         if write_header:
#             writer.writerow(["image", "gen", "score", "queries", "frd_violation"])

#         basename = os.path.basename(self.img_path) if self.img_path is not None else "image"
#         pbar = tqdm(range(1, self.max_iters + 1), desc=f"GenAttack:{basename}")

#         for gen in pbar:
#             if self.query_count >= self.max_queries:
#                 break
#             if no_improve >= self.patience_gens:
#                 continue

#             # Evaluate population with ALMA
#             hs = []
#             fitnesses = []
#             pop_eval = population
#             if self.C == 1:
#                 pop_eval = pop_eval.repeat(1, 3, 1, 1)

#             with torch.no_grad():
#                 emb_pop = self.embedder(pop_eval.contiguous().float())
#                 self.query_count += int(pop_eval.shape[0])

#             for i in range(self.pop_size):
#                 δ = pop_eval[i] - self.clean_img  # Perturbation
#                 adv_img, fitness_val = alma.attack(δ, emb_pop[i:i+1])
#                 hs.append(alma._compute_constraint(adv_img))
#                 fitnesses.append(fitness_val)

#             fitness = torch.tensor(fitnesses).to(self.device)

#             # Update ALMA
#             alma.update(hs)

#             # Best update
#             best_idx = int(torch.argmax(fitness).item())
#             best_val = float(fitness[best_idx].item())
#             if best_val > best_score + self.min_improve:
#                 best_score = best_val
#                 best_adv = population[best_idx].detach().clone()
#                 no_improve = 0
#             else:
#                 no_improve += 1

#             # Selection 
#             sel_logits = (fitness - fitness.max()) / (self.selection_temp + 1e-12)
#             sel_probs = torch.softmax(sel_logits, dim=0)
#             if not torch.isfinite(sel_probs).all():
#                 sel_probs = torch.ones_like(sel_probs) / float(self.pop_size)
#             parent_idx = torch.multinomial(sel_probs, self.pop_size, replacement=True)
#             parents = population[parent_idx]

#             # Crossover & Mutation (unchanged)
#             crossover_mask = torch.rand_like(parents) > 0.5
#             crossover = torch.where(crossover_mask, parents, parents.flip(0))
#             noise_std = 0.5 * self.eps * (0.5 + 0.5 * (1 - gen / float(self.max_iters)))
#             mutation_mask = (torch.rand_like(crossover) < self.mutation_rate).float()
#             noise = torch.randn_like(crossover) * noise_std
#             mutants = torch.clamp(crossover + mutation_mask * noise, 0, 1)
#             mutants = self._clip_eps_ball(mutants)
#             mutants[0] = best_adv
#             population = mutants
#             self.mutation_rate = max(1e-6, self.mutation_rate * self.mutation_decay)

#             # Progress
#             mean_h = np.mean(hs)
#             pbar.set_postfix({
#                 "Fittness": f"{best_score}",
#                 "FRD_h": f"{mean_h}",
#                 "Q": f"{self.query_count}",
#             })

#         pbar.close()
#         csvfile.close()

#         # Save final
#         filename = os.path.basename(self.img_path)
#         save_path = os.path.join(save_dir, filename)
#         self._save_tensor_as_img(best_adv, save_path)

#         with open(log_path, "a", newline="") as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow([filename, gen, best_score, self.query_count, mean_h])

#         final = best_adv.detach().cpu().numpy()
#         if final.ndim == 3 and final.shape[0] in (1, 3):
#             final = np.transpose(final, (1, 2, 0))
#         return final, best_score, self.query_count

#     def _save_tensor_as_img(self, tensor, path):
#         arr = tensor.detach().cpu().numpy()
#         if arr.ndim == 3 and arr.shape[0] in (1, 3):
#             arr = np.transpose(arr, (1, 2, 0))
#         arr = np.clip(arr, 0, 1)
#         arr = (arr * 255).astype(np.uint8)
#         Image.fromarray(arr).save(path)

# # ============================================================
# # 🔹 Dataset Loader
# # ============================================================
# class ImageFolderPILDataset(torch.utils.data.Dataset):
#     def __init__(self, folder, image_size=(224, 224)):
#         self.paths = [
#             os.path.join(folder, f)
#             for f in os.listdir(folder)
#             if f.lower().endswith((".png", ".jpg", ".jpeg"))
#         ]
#         self.transform = transforms.Compose(
#             [transforms.Resize(image_size), transforms.ToTensor()]
#         )

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         path = self.paths[idx]
#         img = Image.open(path).convert("RGB")
#         img_t = self.transform(img)
#         return img_t, path

# # ============================================================
# # 🔹 Main
# # ============================================================
# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     dataset = ImageFolderPILDataset("clean", image_size=(224, 224))
#     loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

#     encoder = ViTEmbedder()
#     for clean_img, path in loader:
#         clean_img = clean_img.squeeze(0)
#         attacker = GenAttackKL(
#             encoder,
#             clean_img,
#             path[0],
#             dataset=dataset,  # Pass full dataset for sampling
#             tau=0.4,  # Target FRD
#             # Your defaults
#             pop_size=8,
#             max_iters=40000,
#             mutation_rate=0.15,
#             mutation_decay=0.999,
#             eps=16 / 255,
#             temperature=1.0,
#             image_channels=3,
#             max_queries=40000,
#             patience_gens=40000,
#             selection_temp=0.5,
#         )

#         adv_img, best_score, queries = attacker.attack()
#         print(f"Saved adversarial for {path}, KL={best_score:.4f}, Queries={queries}")
#         break
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
RL-based black-box adversarial attack acting in the block-DCT domain.

Fixes & improvements over the original:
 - correct log_prob / entropy computation (tensors)
 - conservative policy init + coeff_scale to avoid huge IDCT values
 - stable REINFORCE update (adv as tensor)
 - clamped IDCT perturbations
 - consistent device placement
 - improved logging
"""
import os
import csv
import math
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor
from compute_frd import get_distance_fn

# ---------------------------
# reproducibility
# ---------------------------
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------------------------
# utils: DCT / IDCT (8x8) blockwise
# ---------------------------
def dct_1d_matrix(N, dtype=torch.float32, device='cpu'):
    """
    Create NxN orthonormal DCT-II matrix (type-II, orthonormal).
    """
    n = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)  # (N,1)
    k = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)  # (1,N)
    M = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
    M = M * math.sqrt(2.0 / N)
    M[0, :] = M[0, :] * (1.0 / math.sqrt(2.0))
    return M  # (N,N)

def block_dct2(img, block_size=8):
    """
    img: Tensor (C,H,W) values in [0,1]
    returns DCT coeffs with same (C,H,W) layout (block-wise)
    """
    C, H, W = img.shape
    assert H % block_size == 0 and W % block_size == 0
    B = block_size
    device = img.device
    M = dct_1d_matrix(B, device=device)  # (B,B)
    # reshape to blocks: (C, H/B, B, W/B, B) -> (C, H/B, W/B, B, B)
    blocks = img.view(C, H // B, B, W // B, B).permute(0, 1, 3, 2, 4)  # (C,n_h,n_w,B,B)
    # apply DCT: coeff = M @ block @ M.T
    coeffs = torch.einsum('ij,cnhjk->cnhik', M, blocks)
    coeffs = torch.einsum('ij,cnhik->cnhjk', M, coeffs)
    coeffs = coeffs.permute(0, 1, 3, 2, 4).contiguous().view(C, H, W)
    return coeffs

def block_idct2(coeffs, block_size=8):
    """
    coeffs: Tensor (C,H,W) blockwise DCT coefficients
    returns reconstructed image (C,H,W)
    """
    C, H, W = coeffs.shape
    B = block_size
    device = coeffs.device
    M = dct_1d_matrix(B, device=device)
    Mt = M.t()
    blocks = coeffs.view(C, H // B, B, W // B, B).permute(0, 1, 3, 2, 4)  # (C,n_h,n_w,B,B)
    block = torch.einsum('ij,cnhjk->cnhik', Mt, blocks)
    block = torch.einsum('ij,cnhik->cnhjk', Mt, block)
    img = block.permute(0, 1, 3, 2, 4).contiguous().view(C, H, W)
    return img

# ---------------------------
# ViT embedder wrapper
# ---------------------------
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
        return outputs.last_hidden_state[:, 0, :]  # (B, D)

def l2_from_embeddings(emb_clean, emb_adv, eps=1e-12, normalize=True):
    if normalize:
        emb_clean = F.normalize(emb_clean, p=2, dim=1)
        emb_adv = F.normalize(emb_adv, p=2, dim=1)
    diff = emb_clean - emb_adv
    l2 = torch.sqrt((diff ** 2).sum(dim=1) + eps)
    return l2  # (B,)

# ---------------------------
# RL policy acting on low-frequency DCT coeffs
# ---------------------------
class DCTPolicy(nn.Module):
    def __init__(self, img_shape, block_size=8, keep_coeffs=16, coeff_scale=1.0, device='cpu'):
        """
        img_shape: (C,H,W)
        keep_coeffs: number of low-frequency coefficients per block (zig-zag)
        coeff_scale: global scale applied to sampled coefficients (helps avoid huge IDCT)
        """
        super().__init__()
        C, H, W = img_shape
        assert H % block_size == 0 and W % block_size == 0
        self.device = device
        self.block_size = block_size
        self.C = C
        self.H = H
        self.W = W
        self.n_h_blocks = H // block_size
        self.n_w_blocks = W // block_size
        self.blocks = self.n_h_blocks * self.n_w_blocks
        self.keep_coeffs = keep_coeffs
        self.coeff_scale = coeff_scale

        self.zigzag_idx = self._zigzag_indices(block_size)
        sel = self.zigzag_idx[:keep_coeffs]
        self.total_params = C * self.blocks * keep_coeffs

        # policy parameters (mean, log_std)
        # init mean=0 (no bias), log_std small negative (std ~ 0.05)
        self.mean = nn.Parameter(torch.zeros(self.total_params, device=device, dtype=torch.float32))
        self.log_std = nn.Parameter(torch.full((self.total_params,), -4.0, device=device, dtype=torch.float32))

    def _zigzag_indices(self, B):
        indices = []
        for s in range(2 * B - 1):
            if s % 2 == 0:
                for i in range(s + 1):
                    j = s - i
                    if i < B and j < B:
                        indices.append((i, j))
            else:
                for j in range(s + 1):
                    i = s - j
                    if i < B and j < B:
                        indices.append((i, j))
        return indices

    def sample(self):
        """
        Sample perturbation in DCT domain and return:
         - coeffs tensor (C,H,W)
         - log_prob (tensor scalar)
         - entropy (tensor scalar)
        """
        std = torch.exp(self.log_std)  # (P,)
        eps = torch.randn_like(self.mean)  # (P,)
        sample = self.mean + std * eps  # (P,)
        # scale sample to control amplitude
        sample = sample * self.coeff_scale

        device = self.mean.device
        coeffs = torch.zeros((self.C, self.H, self.W), device=device, dtype=torch.float32)

        ptr = 0
        B = self.block_size
        sel = self.zigzag_idx[: self.keep_coeffs]
        for c in range(self.C):
            for bh in range(self.n_h_blocks):
                for bw in range(self.n_w_blocks):
                    r = bh * B
                    s = bw * B
                    for (u, v) in sel:
                        coeffs[c, r + u, s + v] = sample[ptr]
                        ptr += 1

        # log_prob for multivariate independent Gaussians:
        # log_prob = sum( -0.5 * ( ((x - mean)/std)^2 + 2*log_std + log(2π) ) )
        # we already used eps = (sample - mean)/std
        log_prob = -0.5 * (((eps ** 2) + 2.0 * self.log_std + math.log(2.0 * math.pi)).sum())
        # entropy per dim = 0.5*(1 + ln(2π)) + log_std
        ent_per_dim = 0.5 * (1.0 + math.log(2.0 * math.pi)) + self.log_std
        entropy = ent_per_dim.sum()

        return coeffs, log_prob, entropy

    def to_device(self, device):
        self.to(device)
        self.device = device

# ---------------------------
# RL attack orchestrator
# ---------------------------
class RL_DCT_Attack:
    def __init__(
        self,
        embedder: ViTEmbedder,
        clean_img: torch.Tensor,
        dataset,
        img_path: str,
        tau: float = 0.1,
        block_size: int = 8,
        keep_coeffs: int = 16,
        coeff_scale: float = 0.02,
        eps_pixel: float = 4 / 255,
        penalty_coef: float = 1.0,
        lr: float = 0.01,
        entropy_coeff: float = 0.001,
        max_iters: int = 2000,
        max_queries: int = 20000,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = embedder.to(self.device).eval()
        self.clean_img = clean_img.to(self.device)  # (C,H,W)
        self.dataset = dataset
        self.img_path = img_path
        self.tau = tau
        self.block_size = block_size
        self.keep_coeffs = keep_coeffs
        self.coeff_scale = coeff_scale
        self.eps_pixel = eps_pixel
        self.penalty_coef = penalty_coef
        self.entropy_coeff = entropy_coeff
        self.max_iters = max_iters
        self.max_queries = max_queries
        self.query_count = 0

        C, H, W = self.clean_img.shape
        self.policy = DCTPolicy((C, H, W), block_size=block_size, keep_coeffs=keep_coeffs, coeff_scale=coeff_scale, device=self.device)
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # FRD
        self.clean_batch = self.sample_clean_batch(12)
        self.frd_fn = get_distance_fn(self.clean_batch)

        # clean embedding
        with torch.no_grad():
            self.emb_clean = self.embedder(self.clean_img.unsqueeze(0))  # (1,D)

        # baseline
        self.baseline = 0.0
        self.baseline_alpha = 0.05

    def sample_clean_batch(self, batch_size=12):
        indices = list(range(len(self.dataset)))
        selected = indices[:min(len(indices), batch_size)]
        imgs = [self.dataset[i][0] for i in selected]
        return torch.stack(imgs).to(self.device)

    def frd_violation(self, adv_img_tensor):
        adv = adv_img_tensor.unsqueeze(0) if adv_img_tensor.dim() == 3 else adv_img_tensor
        return max(0.0, float(self.frd_fn(adv) - self.tau))

    def embedding_divergence(self, adv_img_tensor):
        adv = adv_img_tensor.unsqueeze(0) if adv_img_tensor.dim() == 3 else adv_img_tensor
        with torch.no_grad():
            emb_adv = self.embedder(adv)  # (1,D)
            l2 = l2_from_embeddings(self.emb_clean.repeat(emb_adv.shape[0], 1), emb_adv, normalize=True)
            return float(l2[0].item())

    def attack(self, save_dir="adv_rl", log_csv="rl_attack_log.csv"):
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, log_csv)
        write_header = not os.path.exists(log_path)
        csvfile = open(log_path, "a", newline="")
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["image", "gen", "reward", "frd_violation", "policy_loss", "entropy", "baseline", "divergence"])

        best_adv = self.clean_img.clone()
        best_reward = -float("inf")

        pbar = tqdm(range(1, self.max_iters + 1), desc=f"RL-DCT:{os.path.basename(self.img_path)}")
        for gen in pbar:
            if self.query_count >= self.max_queries:
                break

            # 1) sample policy
            coeffs, log_prob, entropy = self.policy.sample()  # coeffs (C,H,W)
            # ensure tensors are on device
            coeffs = coeffs.to(self.device)
            log_prob = log_prob.to(self.device)
            entropy = entropy.to(self.device)

            # 2) idct to pixels
            perturb_pixels = block_idct2(coeffs, block_size=self.block_size)  # (C,H,W)
            # clamp perturbation to eps ball
            perturb_pixels = torch.clamp(perturb_pixels, -self.eps_pixel, self.eps_pixel).to(self.device)
            adv_img = torch.clamp(self.clean_img + perturb_pixels, 0.0, 1.0)

            # 3) evaluate (embedding + FRD)
            with torch.no_grad():
                emb_adv = self.embedder(adv_img.unsqueeze(0))  # (1,D)
            self.query_count += 1
            div = float(l2_from_embeddings(self.emb_clean.repeat(1, 1), emb_adv, normalize=True)[0].item())
            frd_v = max(0.0, float(self.frd_fn(adv_img.unsqueeze(0)) - self.tau))

            # 4) compute reward and advantage
            reward = div - self.penalty_coef * frd_v
            advantage = reward - self.baseline
            # update baseline (running average)
            self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * reward

            # 5) policy gradient (REINFORCE)
            # we want to maximize expected reward -> minimize (-log_prob * advantage)
            # advantage may be positive or negative; convert to tensor
            adv_tensor = torch.tensor(advantage, dtype=torch.float32, device=self.device)
            policy_loss = - log_prob * adv_tensor - self.entropy_coeff * entropy
            # policy_loss is a tensor scalar
            loss = policy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record best
            if reward > best_reward:
                best_reward = reward
                best_adv = adv_img.detach().clone()

            writer.writerow([os.path.basename(self.img_path), gen, float(reward), float(frd_v), float(policy_loss.item()), float(entropy.item()), float(self.baseline), float(div)])
            csvfile.flush()

            pbar.set_postfix({
                "BestReward": f"{best_reward:.5f}",
                "Div": f"{div:.5f}",
                "FRD_v": f"{frd_v:.5f}",
                "Loss": f"{float(policy_loss.item()):.6f}",
                "Ent": f"{float(entropy.item()):.4f}",
                "Q": f"{self.query_count}"
            })

        csvfile.close()

        # save best
        out_path = os.path.join(save_dir, os.path.basename(self.img_path))
        self._save_tensor_as_img(best_adv, out_path)

        return best_adv.detach().cpu().numpy(), best_reward, self.query_count

    def _save_tensor_as_img(self, tensor, path):
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)

# ---------------------------
# dataset loader
# ---------------------------
class ImageFolderPILDataset(torch.utils.data.Dataset):
    def __init__(self, folder, image_size=(224, 224)):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img_t = self.transform(img)
        return img_t, path

# ---------------------------
# main
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ImageFolderPILDataset("clean", image_size=(224, 224))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    encoder = ViTEmbedder().to(device)
    for clean_img, path in loader:
        clean_img = clean_img.squeeze(0).to(device)
        attacker = RL_DCT_Attack(
            embedder=encoder,
            clean_img=clean_img,
            dataset=dataset,
            img_path=path[0],
            tau=0.01,
            block_size=8,
            keep_coeffs=16,
            coeff_scale=0.02,   # try 0.01..0.05 if things explode/are too weak
            eps_pixel=16/255,
            penalty_coef=1.0,
            lr=0.01,
            entropy_coeff=0.001,
            max_iters=2000,
            max_queries=5000,
            device=device
        )

        adv_img, best_reward, queries = attacker.attack(save_dir="adv_rl", log_csv="rl_log.csv")
        print(f"Saved adversarial for {path}, Reward={best_reward:.4f}, Queries={queries}")
        break

if __name__ == "__main__":
    main()
