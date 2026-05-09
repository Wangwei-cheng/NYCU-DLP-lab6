import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from torchvision.utils import save_image, make_grid
from diffusers import DDPMScheduler
import sys
import argparse
import random

# Add 'file' directory to path to import evaluator
sys.path.append('./file')
from evaluator import evaluation_model
from dataset import ICLEVRDataset, get_test_conditions
from model import ConditionalUnet

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # 1. Dataset & Dataloader
        full_dataset = ICLEVRDataset(args.img_dir, args.train_path, args.objects_path)
        val_size = int(len(full_dataset) * 0.01) # 用 1% 驗證即可
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.val_conditions = torch.stack([self.val_dataset[i][1] for i in range(min(len(self.val_dataset), 64))]).to(self.device)
        
        # 2. Model, Noise Scheduler, Optimizer & Scheduler
        self.model = ConditionalUnet(num_classes=24).to(self.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=args.timesteps, beta_schedule='squaredcos_cap_v2')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        
        # 加入 Cosine Annealing 學習率調整
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        
        self.criterion = nn.MSELoss()
        
        # 3. Evaluator
        if not os.path.exists('checkpoint.pth'):
            if os.path.exists('file/checkpoint.pth'):
                import shutil
                shutil.copy('file/checkpoint.pth', 'checkpoint.pth')
                print("Copied file/checkpoint.pth to root for evaluator.")
        
        self.evaluator = evaluation_model()
        
        # 4. Test conditions
        self.test_cond = get_test_conditions(args.test_path, args.objects_path).to(self.device)
        self.new_test_cond = get_test_conditions(args.new_test_path, args.objects_path).to(self.device)
        
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    def train(self):
        best_acc = 0.0
        
        for epoch in range(self.args.epochs):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            total_loss = 0
            
            for i, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # CFG 訓練: 10% 機率將標籤設為全零
                if random.random() < 0.1:
                    labels = torch.zeros_like(labels)
                
                noise = torch.randn_like(images)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=self.device).long()
                
                noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
                noise_pred = self.model(noisy_images, timesteps, labels)
                
                loss = self.criterion(noise_pred, noise)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])
            
            # 更新學習率
            self.lr_scheduler.step()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.6f}")
            
            # Validation
            if (epoch + 1) % self.args.eval_interval == 0:
                val_acc = self.evaluate(self.val_conditions, f"val_epoch_{epoch}")
                print(f"Validation Accuracy (Split Val Set): {val_acc:.4f}")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save_checkpoint(f"best_model.pth")
                    print(f"New best model saved with validation accuracy {val_acc:.4f}")
            
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch}.pth")

    @torch.no_grad()
    def evaluate(self, conditions, prefix):
        self.model.eval()
        # 使用 CFG 進行採樣，預設 guidance_scale 為 3.0
        images = self.sample(conditions, guidance_scale=self.args.guidance_scale)
        
        acc = self.evaluator.eval(images, conditions)
        
        grid = make_grid(images, nrow=8, normalize=True, value_range=(-1, 1))
        save_path = os.path.join(self.args.save_dir, f"{prefix}.png")
        save_image(grid, save_path)
        
        return acc

    @torch.no_grad()
    def sample(self, conditions, guidance_scale=3.0):
        self.model.eval()
        batch_size = conditions.shape[0]
        shape = (batch_size, 3, 64, 64)
        images = torch.randn(shape, device=self.device)
        
        # CFG 需要無條件的標籤
        uncond_conditions = torch.zeros_like(conditions).to(self.device)
        
        for t in tqdm(self.noise_scheduler.timesteps, desc="Sampling", leave=False):
            # 為了效率，合併 cond 和 uncond 的 Batch
            batched_images = torch.cat([images] * 2)
            batched_timesteps = torch.full((batched_images.shape[0],), t, device=self.device, dtype=torch.long)
            batched_conditions = torch.cat([conditions, uncond_conditions])
            
            # 預測雜訊
            noise_pred_all = self.model(batched_images, batched_timesteps, batched_conditions)
            noise_pred_cond, noise_pred_uncond = noise_pred_all.chunk(2)
            
            # 進行 Classifier-Free Guidance 加強
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 更新圖片
            images = self.noise_scheduler.step(noise_pred, t, images).prev_sample
            
        return images

    def save_checkpoint(self, name):
        ckpt_path = os.path.join(self.args.ckpt_dir, name)
        torch.save(self.model.state_dict(), ckpt_path)

    def generate_final_results(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path))
        print(f"Loaded checkpoint from {ckpt_path}")
        
        print(f"Generating results with guidance_scale={self.args.guidance_scale}...")
        test_acc = self.evaluate(self.test_cond, "final_test")
        print(f"Final Test Accuracy: {test_acc:.4f}")
        
        new_test_acc = self.evaluate(self.new_test_cond, "final_new_test")
        print(f"Final New Test Accuracy: {new_test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path arguments
    parser.add_argument('--img_dir',        type=str, default="./iclevr")
    parser.add_argument('--train_path',     type=str, default="./file/train.json")
    parser.add_argument('--test_path',      type=str, default="./file/test.json")
    parser.add_argument('--new_test_path',  type=str, default="./file/new_test.json")
    parser.add_argument('--objects_path',   type=str, default="./file/objects.json")
    parser.add_argument('--ckpt_dir',       type=str, default="./checkpoints")
    parser.add_argument('--save_dir',       type=str, default="./results")
    
    # Training arguments
    parser.add_argument('--device',         type=str, default="cuda")
    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--epochs',         type=int, default=100)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--timesteps',      type=int, default=1000)
    parser.add_argument('--num_workers',    type=int, default=4)
    parser.add_argument('--eval_interval',  type=int, default=5)
    parser.add_argument('--save_interval',  type=int, default=20)
    parser.add_argument('--guidance_scale', type=float, default=3.0, help='CFG guidance scale')
    
    # Mode
    parser.add_argument('--test_only',      action='store_true')
    parser.add_argument('--load_ckpt',      type=str, default=None)
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    
    if args.test_only:
        if args.load_ckpt:
            trainer.generate_final_results(args.load_ckpt)
        else:
            print("Error: Please provide --load_ckpt for test_only mode.")
    else:
        trainer.train()
