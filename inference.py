# inference.py
import torch
import numpy as np
import os
from src.model.transformer import SudokuTransformer
from src.data.generator import Sudoku6x6Generator

# === 설정 ===
MODEL_PATH = "saved_models/best_model.pth" # 학습된 모델 경로
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """학습된 모델을 메모리에 로드"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 오류: 학습된 모델 파일이 없습니다. ({MODEL_PATH})")
        print("   먼저 'python train.py'를 실행해서 모델을 학습시켜주세요.")
        return None

    model = SudokuTransformer().to(DEVICE)
    # 저장된 가중치 불러오기
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval() # 평가 모드로 전환 (Dropout 끄기 등)
    print(f"✅ 모델 로드 완료! ({DEVICE})")
    return model

def solve_sudoku(model, problem_grid):
    """모델을 사용하여 스도쿠 풀기"""
    # 1. 입력 전처리 (Numpy -> Tensor -> Flatten)
    # (6, 6) -> (1, 6, 6) -> (1, 36)
    inp = torch.tensor(problem_grid, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # 2. 모델 추론
    with torch.no_grad():
        output = model(inp) # 결과: (1, 36, 7)
        # 가장 확률 높은 숫자 선택
        predictions = torch.argmax(output, dim=-1) # 결과: (1, 36)
    
    # 3. 결과 후처리 (Tensor -> Numpy)
    # 원래 문제에서 숫자(0이 아닌 것)가 있던 자리는 건드리지 않고,
    # 0(빈칸)이었던 자리만 모델의 예측값으로 채웁니다.
    inp_flat = inp.view(-1).cpu().numpy()     # 원래 문제 (1차원)
    pred_flat = predictions.view(-1).cpu().numpy() # 모델 답안 (1차원)
    
    final_grid = inp_flat.copy()
    
    # 빈칸(0)인 곳만 모델의 답으로 덮어쓰기
    mask = (inp_flat == 0)
    final_grid[mask] = pred_flat[mask]
    
    return final_grid.reshape(6, 6)

def print_comparison(problem, ai_answer, correct_answer=None):
    """문제, AI 답안, 실제 정답을 보기 좋게 출력"""
    print("\n" + "="*40)
    print("🧩 [문제] (0은 빈칸)")
    print(problem)
    
    print("\n🤖 [AI 모델의 풀이]")
    print(ai_answer)
    
    if correct_answer is not None:
        print("\n📝 [실제 정답]")
        print(correct_answer)
        
        # 정답 여부 확인
        if np.array_equal(ai_answer, correct_answer):
            print("\n🎉 결과: 정답입니다! 완벽해요.")
        else:
            diff = np.sum(ai_answer != correct_answer)
            print(f"\n😅 결과: {diff}개 틀렸습니다.")
    print("="*40 + "\n")

def main():
    # 1. 모델 준비
    model = load_model()
    if model is None: return

    # 2. 데이터 생성기 준비
    gen = Sudoku6x6Generator()
    
    while True:
        print("\n[메뉴] 1: 랜덤 문제 풀기  2: 직접 입력해서 풀기  q: 종료")
        choice = input("선택하세요: ")
        
        if choice == '1':
            # 랜덤 문제 생성
            print("\n🎲 랜덤 문제를 생성합니다...")
            prob, sol = gen.generate_dataset(1, min_holes=10, max_holes=15)
            # generate_dataset은 (N, 6, 6)을 반환하므로 [0]을 가져옴
            problem_grid = prob[0]
            solution_grid = sol[0]
            
            # 풀이
            ai_answer = solve_sudoku(model, problem_grid)
            print_comparison(problem_grid, ai_answer, solution_grid)
            
        elif choice == '2':
            # 사용자 입력 (테스트용 하드코딩 예시)
            print("\n✏️ 직접 입력 모드 (코드 내 예시 문제를 풉니다)")
            # 예시: 인터넷에서 본 6x6 스도쿠를 여기에 넣으세요
            custom_problem = np.array([
                [0, 3, 0, 4, 0, 0],
                [4, 0, 2, 0, 6, 0],
                [0, 5, 0, 0, 2, 0],
                [0, 2, 0, 0, 1, 0],
                [0, 6, 0, 5, 0, 2],
                [0, 0, 1, 0, 4, 0]
            ])
            
            ai_answer = solve_sudoku(model, custom_problem)
            print_comparison(custom_problem, ai_answer) # 정답지는 없으니 생략
            
        elif choice.lower() == 'q':
            break
        else:
            print("잘못된 입력입니다.")

if __name__ == "__main__":
    main()