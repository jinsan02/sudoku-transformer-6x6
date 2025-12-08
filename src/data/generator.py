# src/data/generator.py (검증 로직 추가 버전)
import numpy as np
import random

class Sudoku6x6Generator:
    def __init__(self):
        self.rows = 6
        self.cols = 6
        self.box_h = 2
        self.box_w = 3

    def get_empty_grid(self):
        return np.zeros((self.rows, self.cols), dtype=int)

    def is_valid(self, grid, row, col, num):
        # 가로, 세로 확인
        if num in grid[row, :]: return False
        if num in grid[:, col]: return False
        
        # 박스 확인
        start_row = (row // self.box_h) * self.box_h
        start_col = (col // self.box_w) * self.box_w
        if num in grid[start_row:start_row + self.box_h, start_col:start_col + self.box_w]:
            return False
        return True

    def fill_grid(self, grid):
        """빈 그리드를 채워 정답(Solution)을 만듦"""
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r, c] == 0:
                    nums = list(range(1, 7))
                    random.shuffle(nums)
                    for num in nums:
                        if self.is_valid(grid, r, c, num):
                            grid[r, c] = num
                            if self.fill_grid(grid): return True
                            grid[r, c] = 0
                    return False
        return True

    def count_solutions(self, grid, limit=2):
        """
        해답의 개수를 세는 함수 (검증용)
        limit=2로 설정하여 해가 2개 이상 발견되면 즉시 중단 (속도 최적화)
        """
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r, c] == 0:
                    count = 0
                    for num in range(1, 7):
                        if self.is_valid(grid, r, c, num):
                            grid[r, c] = num
                            count += self.count_solutions(grid, limit - count)
                            grid[r, c] = 0 # 백트래킹
                            if count >= limit: # 해가 2개 이상이면 더 볼 필요 없음
                                return count
                    return count
        return 1 # 빈칸이 없으면 해답 1개 찾음

    def remove_numbers(self, grid, holes):
        """구멍을 뚫고 문제를 만듦"""
        quiz = grid.copy()
        count = 0
        while count < holes:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            if quiz[r, c] != 0:
                quiz[r, c] = 0
                count += 1
        return quiz

    def generate_dataset(self, num_samples, min_holes=10, max_holes=20):
        problems = []
        solutions = []
        
        print(f"🧩 검증된 {num_samples}개의 데이터를 생성합니다 (불량품 자동 폐기 중...)")
        
        count = 0
        while count < num_samples:
            # 1. 정답 생성
            solution = self.get_empty_grid()
            self.fill_grid(solution)
            
            # 2. 구멍 뚫기
            holes = random.randint(min_holes, max_holes)
            problem = self.remove_numbers(solution, holes=holes)
            
            # [핵심] 3. 검증 (유일한 해답인가?)
            # 해답이 정확히 1개인 경우에만 통과
            if self.count_solutions(problem.copy()) == 1:
                problems.append(problem)
                solutions.append(solution)
                count += 1
                
                if count % 1000 == 0:
                    print(f"   ... {count}개 생성 완료")
            else:
                # 불량품(해답이 2개 이상)은 아무것도 안 하고 그냥 넘어감 (자동 폐기)
                # while 루프가 다시 돌면서 새로운 문제를 만듦
                continue
            
        return np.array(problems), np.array(solutions)

if __name__ == "__main__":
    gen = Sudoku6x6Generator()
    print("검증 로직 테스트 중...")
    p, s = gen.generate_dataset(1, min_holes=15, max_holes=20)
    print("문제:\n", p[0])
    print("✅ 검증된 생성기 정상 작동")