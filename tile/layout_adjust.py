import sys
import numpy as np
from copy import deepcopy
sys.path.append('.')
from tile.rect import Rect, merge_rect
from tile.layout_generator import Layout
from tile.cost_function import CostFunc

class UnionSet:
    def __init__(self, N: int, M: int) -> None:
        self.n = N
        self.m = M
        self._data = [[(i, j) for j in range(M)] for i in range(N)]

    def __getitem__(self, idx):
        i, j = idx
        return self._data[i][j]

    def __setitem__(self, key, value):
        i, j = key
        self._data[i][j] = value

    def combine_rows(self, start, end):
        for j in range(self.m):
            parents = set()
            for i in range(start, end+1):
                for p in self._data[i][j]:
                    parents.add(p)
            self._data[start][j] = list(parents) # multi-parents
        for i in range(start+1, end+1):
            self._data.pop(i) # remove the i-th row
        self.n = len(self._data)

    def combine_cols(self, start, end):
        for i in range(self.n):
            parents = set()
            for j in range(start, end+1):
                for p in self._data[i][j]:
                    parents.add(p)
            self._data[i][start] = list(parents) # multi-parents
            for j in range(start+1, end+1):
                self._data[i].pop(j) # remove the j-th column
        self.m = len(self._data[0])

    def __repr__(self) -> str:
        return str(self._data)

def merging(layout: Layout, parent: UnionSet, rows: int, cols: int):
    N, M = len(layout), len(layout[0]) # rows and cols
    #merged_layout = deepcopy(layout)
    merged_layout = [[None for _ in range(M)] for _ in range(N)]
    for i in range(N):
        for j in range(M):
            tile, has_roi = layout[i][j]
            merged_layout[i][j] = (Rect(tile.x1, tile.y1, tile.x2, tile.y2), has_roi)
    if parent:
        merged_parent = deepcopy(parent)
    idx = 0
    while idx < len(rows):
        i = rows[idx]
        for j in range(M):
            tile1, has_roi1 = merged_layout[i-1][j]
            tile2, has_roi2 = merged_layout[i][j]
            tile = merge_rect(tile1, tile2)
            has_roi = has_roi1 or has_roi2
            merged_layout[i-1][j] = tile, has_roi
        merged_layout.pop(i)
        if parent:
            merged_parent.combine_rows(i-1, i)
        rows = [row - 1 for row in rows]
        idx += 1
    N, M = len(merged_layout), len(merged_layout[0]) # rows and cols
    idx = 0
    while idx < len(cols):
        j = cols[idx]
        for i in range(N):
            tile1, has_roi1 = merged_layout[i][j-1]
            tile2, has_roi2 = merged_layout[i][j]
            tile = merge_rect(tile1, tile2)
            has_roi = has_roi1 or has_roi2
            merged_layout[i][j-1] = tile, has_roi
            merged_layout[i].pop(j)
        if parent:
            merged_parent.combine_cols(j-1, j)
        cols = [col - 1 for col in cols]
        idx += 1
    if parent:
        return merged_layout, merged_parent
    else:
        return merged_layout

def cost_function(cost_func: CostFunc, layout: Layout, parent: UnionSet) -> float:
    N, M = len(layout), len(layout[0]) # rows and cols
    cost = 0
    for i in range(N):
        for j in range(M):
            tile, has_roi = layout[i][j] 
            if has_roi:
                n = len(parent[(i, j)])
                #cost += get_roi_rand_dec_time(codec, tile.area(), gop, gop) * n
                cost += cost_func(tile.area()) * n
    return cost

def heavy_cost_function(
        cost_func: CostFunc,
        layout: Layout, 
        parent: UnionSet, 
        merge_rows: list, 
        merge_cols: list,               # rows/cols here means the splitting lines
        visible_row_range: list=None, 
        visible_col_range: list=None    # rows/cols here means the layout lines
    ) -> float:
    N, M = len(layout), len(layout[0]) # rows and cols
    # find adjacent
    def find_adj(arr, max_val):
        adj, tmp = {}, []
        for i, x in enumerate(arr):
            if i == 0 or x != arr[i-1] + 1:
                if tmp:
                    tmp.append(tmp[0] - 1) # the left one
                    for y in tmp:
                        adj[y] = tmp
                tmp = [x]
            else:
                tmp.append(x)
        if tmp:
            tmp.append(tmp[0] - 1) # the left one
            for y in tmp:
                adj[y] = tmp
        for x in range(max_val):
            if x not in adj:
                adj[x] = [x]
        return adj
    
    adj_rows = find_adj(merge_rows, N)
    adj_cols = find_adj(merge_cols, M)
    
    cost = 0
    visited = set()
    row_range = visible_row_range if visible_row_range else range(N)
    col_range = visible_col_range if visible_col_range else range(M)
    for i in row_range:
        for j in col_range:
            if (i, j) in visited:
                continue
            # search in area
            clusters = set()
            area = 0
            for ii in adj_rows[i]:
                for jj in adj_cols[j]:
                    for p in parent[(ii, jj)]:
                        clusters.add(p)
                    tile, _ = layout[ii][jj]
                    area += tile.area()
                    visited.add((ii, jj))
            n = len(clusters)
            cost += cost_func(area) * n
            #cost += get_roi_rand_dec_time(codec, area, gop, gop) * n
            visited.add((i, j))
    return cost

def greedy_tiling(
        cost_func: CostFunc,
        layout: Layout, 
        parent: UnionSet, 
        curr_cost: float=np.inf
    ):
    N, M = len(layout), len(layout[0]) # rows and cols
    if curr_cost == np.inf:
        min_cost = cost_function(cost_func, layout, parent)
    else:
        min_cost = curr_cost
    best_layout = layout
    best_parent = parent
    for i1 in range(N):
        for j1 in range(M):
            _, has_roi1 = layout[i1][j1] 
            if not has_roi1: # find a roi tile
                continue
            for i2 in range(i1, N):
                for j2 in range(j1+1 if i1==i2 else 0, M):
                    _, has_roi2 = layout[i2][j2]
                    if not has_roi2: # find a roi tile
                        continue
                    i1, i2 = min(i1, i2), max(i1, i2)   # left-top
                    j1, j2 = min(j1, j2), max(j1, j2)   # right-bottom
                    rows = [i for i in range(i1+1, i2+1) if 0 < i < N]
                    cols = [j for j in range(j1+1, j2+1) if 0 < j < M]
                    merged_layout, merged_parent = merging(layout, parent, rows, cols)
                    cost = cost_function(cost_func, merged_layout, merged_parent)
                    if cost < min_cost:
                        min_cost = cost
                        best_layout = merged_layout
                        best_parent = merged_parent
    if best_layout == layout:
        return best_layout
    return greedy_tiling(cost_func, best_layout, best_parent, min_cost)

def fast_greedy_tiling(
        cost_func: CostFunc,
        layout: Layout, 
        parent: UnionSet, 
        curr_plan: tuple=([], []), 
        curr_cost: float=np.inf
    ):
    N, M = len(layout), len(layout[0]) # rows and cols
    if curr_cost == np.inf:
        min_cost = cost_function(cost_func, layout, parent)
    else:
        min_cost = curr_cost
    base_rows, base_cols = curr_plan
    best_plan = None
    for i1 in range(N):
        for j1 in range(M):
            _, has_roi1 = layout[i1][j1] 
            if not has_roi1: # find a roi tile
                continue
            for i2 in range(i1, N):
                for j2 in range(j1+1 if i1==i2 else 0, M):
                    _, has_roi2 = layout[i2][j2]
                    if not has_roi2: # find a roi tile
                        continue
                    rows = [i for i in range(i1+1, i2+1) if 0 < i < N]
                    cols = [j for j in range(j1+1, j2+1) if 0 < j < M]
                    rows = list(set(rows + base_rows))
                    cols = list(set(cols + base_cols))
                    if (len(rows) > len(base_rows) and len(cols) >= len(base_cols)) or \
                        (len(rows) >= len(base_rows) and len(cols) > len(base_cols)) :
                        rows, cols = sorted(rows), sorted(cols)
                        cost = heavy_cost_function(cost_func, layout, parent, rows, cols)
                        if cost < min_cost:
                            min_cost = cost
                            best_plan = (rows, cols)
    if not best_plan:
        rows, cols = curr_plan
        merged_layout = merging(layout, None, rows, cols)
        return merged_layout
    return fast_greedy_tiling(cost_func, layout, parent, best_plan, min_cost)


def solution_is_valid(arr: list, non_roi_arr: list, max_val: int):
    for i in non_roi_arr:
        if i == 0 and i + 1 in arr:
            return False # the first row/col is non-roi
        if i == max_val - 1 and i in arr:
            return False # the last row/col is non-roi
        if (i in arr) ^ (i+1 in arr):
            return False # a non-roi between 2 roi
    return True

def get_solutions(layout: Layout):
    N, M = len(layout), len(layout[0]) # rows and cols
    valid_rows = set() # this contains all horizontal splitting lines
    valid_cols = set() # this contains all vertical splitting lines
    non_roi_rows = [i for i in range(N)]
    non_roi_cols = [j for j in range(M)]
    for i in range(N):
        for j in range(M):
            _, has_roi = layout[i][j]
            if has_roi:
                if i in non_roi_rows:
                    non_roi_rows.remove(i)
                if j in non_roi_cols:
                    non_roi_cols.remove(j)
                if i > 0:
                    valid_rows.add(i)
                if i < N - 1:
                    valid_rows.add(i + 1)
                if j > 0:
                    valid_cols.add(j)
                if j < M - 1:
                    valid_cols.add(j + 1)
    def dfs(arr, start, path):
        res = [path]
        for i in range(start, len(arr)):
            res += dfs(arr, i + 1, path + [arr[i]])
        return res
    merge_rows = dfs(list(valid_rows), 0, [])
    merge_cols = dfs(list(valid_cols), 0, [])
    merge_rows = [rows for rows in merge_rows if solution_is_valid(rows, non_roi_rows, N)]
    merge_cols = [cols for cols in merge_cols if solution_is_valid(cols, non_roi_cols, M)]
    return merge_rows, merge_cols, non_roi_rows, non_roi_cols
 
def optimal_tiling(cost_func: CostFunc, layout: Layout, parent: UnionSet):
    # enumerate all possibilities
    merge_rows, merge_cols, _, _ = get_solutions(layout)
    min_cost = np.inf
    best_layout = layout
    for rows in merge_rows:
        for cols in merge_cols:
            merged_layout, merged_parent = merging(layout, parent, rows, cols)
            cost = cost_function(cost_func, merged_layout, merged_parent)
            if cost < min_cost:
                min_cost = cost
                best_layout = merged_layout
    return best_layout

def fast_optimal_tiling(cost_func: CostFunc, layout: Layout, parent: UnionSet):
    '''
    time complexity: O(2^(N+M))
    '''
    # enumerate all possibilities
    merge_rows, merge_cols, _, _ = get_solutions(layout)
    min_cost = np.inf
    best_plan = None
    for rows in merge_rows:
        for cols in merge_cols:
            cost = heavy_cost_function(cost_func, layout, parent, rows, cols)
            if cost < min_cost:
                min_cost = cost
                best_plan = (rows, cols)
            
    #print('optimal', min_cost)
    rows, cols = best_plan
    best_layout = merging(layout, None, rows, cols)
    return best_layout

def very_fast_optimal_tiling(cost_func: CostFunc, layout: Layout, parent: UnionSet):
    '''
    time complexity: O(2^min(N,M)*max(N,M)^3)
    '''
    N, M = len(layout), len(layout[0]) # rows and cols
    # enumerate all possibilities
    merge_rows, merge_cols, non_roi_rows, non_roi_cols = get_solutions(layout)

    def dfs(path, i, j):
        k = path[i][j]
        if k == -1:
            return []
        elif k == -2:
            return [x+1 for x in range(i, j)]
        return dfs(path, i, k) + dfs(path, k+1, j)
    
    def dp_func(enum_arr, non_roi_arr, max_val, dp_col=True):
        min_cost = np.inf
        best_plan = None
        for x in enum_arr:
            dp = np.full((max_val, max_val), np.inf)
            path = np.full((max_val, max_val), -1)
            for i in range(max_val):
                if i not in non_roi_arr:
                    if dp_col:
                        dp[i][i] = heavy_cost_function(cost_func, layout, parent, x, [], None, range(i, i+1))
                    else:
                        dp[i][i] = heavy_cost_function(cost_func, layout, parent, [], x, range(i, i+1), None)
                else:
                    dp[i][i] = 0
            for gap in range(1, max_val):
                for i in range(max_val - gap):
                    j = i + gap
                    y = [k for k in range(i+1, j+1)]
                    if solution_is_valid(y, non_roi_arr, max_val):
                        if dp_col:
                            dp[i][j] = heavy_cost_function(cost_func, layout, parent, x, y, None, range(i, j+1))
                        else:
                            dp[i][j] = heavy_cost_function(cost_func, layout, parent, y, x, range(i, j+1), None)
                        path[i][j] = -2
                    #start = i if path[i][j-1] < 0 else path[i][j-1]
                    #end = j if path[i+1][j] < 0 else path[i+1][j] + 1
                    for k in range(i, j):
                        cost = dp[i][k] + dp[k+1][j]
                        if cost < dp[i][j]:
                            dp[i][j] = cost
                            path[i][j] = k
            if dp[0][-1] < min_cost:
                min_cost = dp[0][-1]
                if dp_col:
                    best_plan = (x, path.copy())
                else:
                    best_plan = (path.copy(), x)
        if dp_col:
            x, best_path = best_plan
        else:
            best_path, x = best_plan
        y = sorted(dfs(best_path, 0, max_val-1))
        if dp_col:
            best_plan = (x, y)
        else:
            best_plan = (y, x)
        return best_plan
    
    if len(merge_rows) <= len(merge_cols):
        rows, cols = dp_func(merge_rows, non_roi_cols, M, True)
    else:
        rows, cols = dp_func(merge_cols, non_roi_rows, N, False)
    best_layout = merging(layout, None, rows, cols)
    return best_layout

def ultra_fast_optimal_tiling(cost_func: CostFunc, layout: Layout, parent: UnionSet):
    '''
    time complexity: O(2^min(N,M)*max(N,M)^2)
    '''
    N, M = len(layout), len(layout[0]) # rows and cols
    # enumerate all possibilities
    merge_rows, merge_cols, non_roi_rows, non_roi_cols = get_solutions(layout)

    def dfs(path, i):
        j = path[i]
        if j == -1:
            return [k for k in range(1, i+1)]
        else:
            return dfs(path, j) + [k for k in range(j+2, i+1)]
    
    def dp_func(enum_arr, non_roi_arr, max_val, dp_col=True):
        min_cost = np.inf
        best_plan = None
        for x in enum_arr:
            dp = np.zeros((max_val))
            path = np.full((max_val), -1)
            for i in range(max_val):
                if i > 0 and i in non_roi_arr:
                    dp[i] = dp[i-1]
                    path[i] = i - 1
                    continue
                if dp_col:
                    dp[i] = heavy_cost_function(cost_func, layout, parent, x, range(1,i+1), None, range(i+1))
                else:
                    dp[i] = heavy_cost_function(cost_func, layout, parent, range(1,i+1), x, range(i+1), None)
                if dp[i] == 0:
                    continue
                for j in range(i):
                    if dp_col:
                        cost = heavy_cost_function(cost_func, layout, parent, x, range(j+2,i+1), None, range(j+1,i+1))
                    else:
                        cost = heavy_cost_function(cost_func, layout, parent, range(j+2,i+1), x, range(j+1,i+1), None)
                    cost += dp[j]
                    if cost < dp[i]:
                        dp[i] = cost
                        path[i] = j
            if dp[-1] < min_cost:
                min_cost = dp[-1]
                if dp_col:
                    best_plan = (x, path.copy())
                else:
                    best_plan = (path.copy(), x)
        if dp_col:
            x, best_path = best_plan
        else:
            best_path, x = best_plan
        y = sorted(dfs(best_path, max_val-1))
        if dp_col:
            best_plan = (x, y)
        else:
            best_plan = (y, x)
        return best_plan
    
    if len(merge_rows) <= len(merge_cols):
        rows, cols = dp_func(merge_rows, non_roi_cols, M, True)
    else:
        rows, cols = dp_func(merge_cols, non_roi_rows, N, False)
    best_layout = merging(layout, None, rows, cols)
    return best_layout