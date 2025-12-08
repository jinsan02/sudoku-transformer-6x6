# generate_data.py (검증 적용 버전)
import torch
import numpy as np
import os
from src.data.generator import Sudoku6x6Generator
import time

def save_dataset(problems, solutions, filename):
    data = {
        "problems": torch.tensor(problems, dtype=torch.long),
        "solutions": torch.tensor(solutions, dtype=torch.long)
    }
    torch.save(data, filename)
    print(f"   💾 저장 완료: {filename} (크기: {len(problems)}개)")

def main():
    # 데이터 50만 개
    TRAIN_SIZE = 500000 
    VAL_SIZE = 20000
    OUTPUT_DIR = "data/processed"
    
    gen = Sudoku6x6Generator()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🚀 '검증된' 스도쿠 데이터 생성을 시작합니다...")
    start_time = time.time()

    # [난이도 설정]
    # 이제 검증 로직이 있으므로 빈칸을 조금 더 뚫어도 됩니다.
    # 불량품은 알아서 걸러지므로, 만들어진 데이터는 무조건 '정답이 1개'입니다.
    # 추천: 12 ~ 20개 (6x6에서 꽤 어려운 수준까지 커버)
    
    print(f"\n[1/2] 학습 데이터 생성 중 ({TRAIN_SIZE}개)...")
    # 검증 때문에 생성 속도가 조금 느려질 수 있습니다. (3060 기준 2~3배 시간 소요 예상)
    train_probs, train_sols = gen.generate_dataset(TRAIN_SIZE, min_holes=12, max_holes=20)
    save_dataset(train_probs, train_sols, os.path.join(OUTPUT_DIR, "train.pt"))

    print(f"\n[2/2] 검증 데이터 생성 중 ({VAL_SIZE}개)...")
    val_probs, val_sols = gen.generate_dataset(VAL_SIZE, min_holes=12, max_holes=20)
    save_dataset(val_probs, val_sols, os.path.join(OUTPUT_DIR, "val.pt"))

    end_time = time.time()
    print(f"\n✅ 모든 작업 완료! (소요 시간: {end_time - start_time:.2f}초)")

if __name__ == "__main__":
    main()