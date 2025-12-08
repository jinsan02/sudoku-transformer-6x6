# train.py (스케줄러 추가된 최종 버전)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import SudokuDataset
from src.model.transformer import SudokuTransformer
import time

# === 설정 ===
BATCH_SIZE = 256        # 3060 성능 활용 (128 -> 256)
LEARNING_RATE = 0.001   # 스케줄러를 쓰므로 높게 시작
EPOCHS = 30             # 30 에폭이면 충분
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "saved_models/checkpoints"
BEST_MODEL_PATH = "saved_models/best_model.pth"

def calculate_accuracy(outputs, targets):
    predictions = torch.argmax(outputs, dim=-1)
    if targets.dim() == 3: targets = targets.view(targets.size(0), -1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total

def main():
    print(f"🔧 학습 장치: {DEVICE} (6x6 최종 버전)")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("📂 대용량 데이터를 로드하는 중... (50만 개)")
    train_dataset = SudokuDataset("data/processed/train.pt")
    val_dataset = SudokuDataset("data/processed/val.pt")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SudokuTransformer().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # [핵심] 학습률 스케줄러 추가
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 이어하기 로직
    start_epoch = 0
    last_ckpt_path = os.path.join(SAVE_DIR, "last_checkpoint.pth")
    
    if os.path.exists(last_ckpt_path):
        try:
            checkpoint = torch.load(last_ckpt_path, weights_only=True)
            # 모델 구조가 바뀌었으므로 불러오기 실패할 수 있음 (무시하고 새로 시작)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"🔄 {start_epoch} 에폭부터 학습을 재개합니다.")
        except:
            print("✨ (모델 구조 변경됨) 처음부터 학습을 시작합니다.")
    else:
        print("✨ 처음부터 학습을 시작합니다.")

    best_val_acc = 0.0
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        train_acc = 0
        
        for batch_idx, (problems, solutions) in enumerate(train_loader):
            problems, solutions = problems.to(DEVICE), solutions.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(problems)
            loss = criterion(outputs.view(-1, 7), solutions.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_acc += calculate_accuracy(outputs, solutions)
            
            if batch_idx % 500 == 0:
                print(f"   Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # 에폭 끝날 때마다 학습률 조정
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        val_acc = 0
        with torch.no_grad():
            for p, s in val_loader:
                p, s = p.to(DEVICE), s.to(DEVICE)
                val_acc += calculate_accuracy(model(p), s)
        avg_val_acc = val_acc / len(val_loader)
        
        print(f"📊 Epoch {epoch+1}/{EPOCHS} | Val Acc: {avg_val_acc*100:.2f}% | LR: {current_lr:.6f}")
        
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, last_ckpt_path)
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   🏆 최고 기록 경신! ({BEST_MODEL_PATH})")

    print("\n🏁 학습 종료!")

if __name__ == "__main__":
    main()