cat = 3
mouse = 3

if cat==3:
    cat=6

grid = [" " for i in range(6)]
grid[cat-1] = "C"
grid[mouse-1] = "M"


print(f"\
        _________________________\n\
        |   1   |       |   4   |\n\
        |   {grid[0]}   |   3   |   {grid[3]}   |\n\
        ---------   {grid[2]}   ---------\n\
        |   2   |   {grid[5]}   |   5   |\n\
        |   {grid[1]}   |       |   {grid[4]}   |")