# inference.py
import torch
import numpy as np
import os
from src.config import Config
from src.model.transformer import SudokuTransformer
from src.data.generator import SudokuGenerator

def load_model():
    if not os.path.exists(Config.MODEL_PATH):
        print(f"❌ 모델 파일 없음: {Config.MODEL_PATH}")
        return None

    model = SudokuTransformer(Config).to(Config.DEVICE)
    # weights_only=True는 보안상 권장됨
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    print(f"✅ 모델 로드 완료 ({Config.DEVICE})")
    return model

def solve_sudoku(model, problem_grid):
    # 입력 처리: (N, N) -> (1, N*N)
    inp = torch.tensor(problem_grid, dtype=torch.long).unsqueeze(0).to(Config.DEVICE)
    if inp.dim() == 3: inp = inp.view(1, -1)

    with torch.no_grad():
        output = model(inp)
        predictions = torch.argmax(output, dim=-1)
    
    # 결과 복원
    inp_flat = inp.view(-1).cpu().numpy()
    pred_flat = predictions.view(-1).cpu().numpy()
    
    final_grid = inp_flat.copy()
    mask = (inp_flat == 0)
    final_grid[mask] = pred_flat[mask]
    
    # Config에 따라 Reshape
    return final_grid.reshape(Config.GRID_SIZE, Config.GRID_SIZE)

def print_comparison(problem, ai_answer):
    print("\n" + "="*20)
    print("🧩 [AI 풀이 결과]")
    print(ai_answer)
    print("="*20)

def main():
    model = load_model()
    if model is None: return

    gen = SudokuGenerator()
    
    while True:
        print(f"\n[메뉴 ({Config.GRID_SIZE}x{Config.GRID_SIZE})] 1: 랜덤 문제  2: 종료")
        choice = input("선택: ")
        
        if choice == '1':
            # [수정] 하드코딩 제거! Config에서 난이도 자동 적용
            prob, sol = gen.generate_dataset(
                1, 
                min_holes=Config.MIN_HOLES, 
                max_holes=Config.MAX_HOLES
            )
            
            print("\n[문제]")
            print(prob[0])
            
            ai_answer = solve_sudoku(model, prob[0])
            print_comparison(prob[0], ai_answer)
            
            if np.array_equal(ai_answer, sol[0]):
                print("🎉 정답입니다!")
            else:
                print("😅 틀렸습니다.")
                
        elif choice == '2':
            break

if __name__ == "__main__":
    main()