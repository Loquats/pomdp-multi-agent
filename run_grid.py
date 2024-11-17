from gridworld.grid import Grid


grid = Grid(3, 3, 90, 90, title='Tic Tac Toe', margin=1)
grid[0, 0] = 'O'
grid[1, 1] = 'X'
grid[2, 1] = 'O'
grid[2, 2] = 'X'
grid.run()
