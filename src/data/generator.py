# src/data/generator.py
import numpy as np
import random
from src.config import Config # Config 연동

class SudokuGenerator: # 이름 변경 (6x6 제거)
    def __init__(self):
        # Config에서 크기 정보를 가져옴 (유지보수성 강화)
        self.rows = Config.GRID_SIZE
        self.cols = Config.GRID_SIZE
        self.box_h = Config.BOX_H
        self.box_w = Config.BOX_W
        self.num_classes = Config.NUM_CLASSES # 0~N

    def get_empty_grid(self):
        return np.zeros((self.rows, self.cols), dtype=int)

    def is_valid(self, grid, row, col, num):
        if num in grid[row, :]: return False
        if num in grid[:, col]: return False
        
        start_row = (row // self.box_h) * self.box_h
        start_col = (col // self.box_w) * self.box_w
        if num in grid[start_row:start_row + self.box_h, start_col:start_col + self.box_w]:
            return False
        return True

    def fill_grid(self, grid):
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r, c] == 0:
                    # 1부터 N까지 숫자 사용
                    nums = list(range(1, self.num_classes)) 
                    random.shuffle(nums)
                    for num in nums:
                        if self.is_valid(grid, r, c, num):
                            grid[r, c] = num
                            if self.fill_grid(grid): return True
                            grid[r, c] = 0
                    return False
        return True

    def count_solutions(self, grid, limit=2):
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r, c] == 0:
                    count = 0
                    for num in range(1, self.num_classes):
                        if self.is_valid(grid, r, c, num):
                            grid[r, c] = num
                            count += self.count_solutions(grid, limit - count)
                            grid[r, c] = 0
                            if count >= limit: return count
                    return count
        return 1

    def remove_numbers(self, grid, holes):
        quiz = grid.copy()
        count = 0
        while count < holes:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            if quiz[r, c] != 0:
                quiz[r, c] = 0
                count += 1
        return quiz

    def generate_dataset(self, num_samples, min_holes, max_holes):
        problems = []
        solutions = []
        
        print(f"🧩 스도쿠 데이터 생성 중... (크기: {self.rows}x{self.cols})")
        
        count = 0
        while count < num_samples:
            solution = self.get_empty_grid()
            self.fill_grid(solution)
            
            holes = random.randint(min_holes, max_holes)
            problem = self.remove_numbers(solution, holes=holes)
            
            if self.count_solutions(problem.copy()) == 1:
                problems.append(problem)
                solutions.append(solution)
                count += 1
                if count % 1000 == 0:
                    print(f"   ... {count}개 완료")
            else:
                continue
            
        return np.array(problems), np.array(solutions)