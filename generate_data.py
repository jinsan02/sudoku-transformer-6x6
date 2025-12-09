# generate_data.py
import torch
import numpy as np
import os
import time
from src.data.generator import SudokuGenerator
from src.config import Config

def save_dataset(problems, solutions, filename):
    data = {
        "problems": torch.tensor(problems, dtype=torch.long),
        "solutions": torch.tensor(solutions, dtype=torch.long)
    }
    torch.save(data, filename)
    print(f"   💾 저장 완료: {filename} (크기: {len(problems)}개)")

def main():
    # 학습/검증 데이터 개수
    TRAIN_SIZE = 500000 
    VAL_SIZE = 20000
    
    # Config에서 경로 가져오기
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    # 생성기 초기화
    gen = SudokuGenerator()
    
    print(f"🚀 [Config: {Config.GRID_SIZE}x{Config.GRID_SIZE}] 데이터 생성을 시작합니다...")
    # [수정] 난이도를 Config에서 가져와서 출력
    print(f"   - 난이도(빈칸): {Config.MIN_HOLES} ~ {Config.MAX_HOLES}개")
    
    start_time = time.time()

    print(f"\n[1/2] 학습 데이터 ({TRAIN_SIZE}개)")
    train_probs, train_sols = gen.generate_dataset(
        TRAIN_SIZE, 
        min_holes=Config.MIN_HOLES,  # [수정] 하드코딩 제거
        max_holes=Config.MAX_HOLES
    )
    save_dataset(train_probs, train_sols, f"{Config.DATA_DIR}/train.pt")

    print(f"\n[2/2] 검증 데이터 ({VAL_SIZE}개)")
    val_probs, val_sols = gen.generate_dataset(
        VAL_SIZE, 
        min_holes=Config.MIN_HOLES,  # [수정] 하드코딩 제거
        max_holes=Config.MAX_HOLES
    )
    save_dataset(val_probs, val_sols, f"{Config.DATA_DIR}/val.pt")

    print(f"\n✅ 완료! ({time.time() - start_time:.2f}초)")

if __name__ == "__main__":
    main()