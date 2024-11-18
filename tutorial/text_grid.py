import numpy as np

class TextGrid:
    def __init__(self, num_cols, num_rows):
        self.grid = np.zeros((num_rows, num_cols), dtype='U1')

    def clear_all(self):
        self.grid.fill(0)

    def print(self):
        for row in self.grid:
            print(''.join(row))

    def __getitem__(self, coo):
        col, row = coo
        return self.grid[row][col]

    def __setitem__(self, coo, value):
        col, row = coo
        self.grid[row][col] = value

    def done(self):
        pass