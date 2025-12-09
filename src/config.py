# src/config.py
import torch

class Config:
    # 1. 스도쿠 규격 (나중에 여기만 9, 3, 3으로 바꾸면 9x9가 됩니다!)
    GRID_SIZE = 6
    BOX_H = 2
    BOX_W = 3
    
    # 자동 계산되는 값들
    SEQ_LEN = GRID_SIZE * GRID_SIZE
    NUM_CLASSES = GRID_SIZE + 1
    
    # [NEW] 2. 난이도 설정 (비율로 자동 계산)
    # 6x6 -> 12~19개 구멍
    # 9x9 -> 28~44개 구멍
    MIN_HOLES = int(SEQ_LEN * 0.35)  # 전체 칸의 35%
    MAX_HOLES = int(SEQ_LEN * 0.55)  # 전체 칸의 55%
    
    # 3. 학습 하이퍼파라미터
    BATCH_SIZE = 256
    LR = 0.001
    EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 4. 경로
    DATA_DIR = "data/processed"
    MODEL_SAVE_DIR = "saved_models"
    MODEL_PATH = f"{MODEL_SAVE_DIR}/best_model.pth"