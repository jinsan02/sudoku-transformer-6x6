# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data.dataset import SudokuDataset
from src.model.transformer import SudokuTransformer
from src.utils import calculate_accuracy, seed_everything

def main():
    seed_everything(42)
    print(f"🔧 학습 장치: {Config.DEVICE}")
    
    # 모델 저장 폴더 생성
    if not os.path.exists(Config.MODEL_SAVE_DIR):
        os.makedirs(Config.MODEL_SAVE_DIR)
    
    # 데이터 로드
    train_loader = DataLoader(SudokuDataset(f"{Config.DATA_DIR}/train.pt"), 
                            batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SudokuDataset(f"{Config.DATA_DIR}/val.pt"), 
                          batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = SudokuTransformer(Config).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # --- [복구됨] 체크포인트 로드 로직 ---
    start_epoch = 0
    best_acc = 0.0
    checkpoint_path = f"{Config.MODEL_SAVE_DIR}/last_checkpoint.pth"

    if os.path.exists(checkpoint_path):
        print(f"🔄 체크포인트 발견! 학습을 재개합니다: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            print(f"   ▶ Epoch {start_epoch+1}부터 시작합니다. (현재 최고 기록: {best_acc*100:.2f}%)")
        except Exception as e:
            print(f"⚠️ 체크포인트 로드 실패 (파일 깨짐 등): {e}")
            print("   ✨ 처음부터 다시 시작합니다.")
    else:
        print("✨ 처음부터 학습을 시작합니다.")

    # 학습 루프
    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for p, s in loop:
            p, s = p.to(Config.DEVICE), s.to(Config.DEVICE)
            
            optimizer.zero_grad()
            out = model(p)
            loss = criterion(out.view(-1, Config.NUM_CLASSES), s.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_accuracy(out, s)
            
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        
        # 검증
        model.eval()
        val_acc = sum([calculate_accuracy(model(p.to(Config.DEVICE)), s.to(Config.DEVICE)) for p, s in val_loader]) / len(val_loader)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"   Done! Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.6f}")
        
        # --- 체크포인트 저장 (매 에폭마다) ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)

        # 최고 기록 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"   🏆 최고 기록 경신! 모델 저장됨: {Config.MODEL_PATH}")

if __name__ == "__main__":
    main()