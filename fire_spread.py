"""
풍향·풍속·지형·연료습도를 반영한 Cellular Automata 산불 확산 시뮬레이터
"""
import numpy as np

# 셀 상태
EMPTY, TREE, BURNING, BURNED = 0, 1, 2, 3


def initialize_grid(size=50, tree_density=0.85):
    """임상도를 모사한 격자 초기화"""
    grid = np.where(np.random.rand(size, size) < tree_density, TREE, EMPTY)
    return grid


def step(grid, wind_dir=90, wind_speed=5, fuel_moisture=10):
    """
    한 시간 step 진행
    wind_dir: 풍향(도, 0=북, 90=동, 180=남, 270=서)
    wind_speed: 풍속(m/s)
    fuel_moisture: 연료습도(%) — 낮을수록 잘 탐
    """
    new = grid.copy()
    h, w = grid.shape

    # 풍향 벡터 (dx, dy): 화재가 진행하는 방향
    rad = np.deg2rad(wind_dir)
    wx, wy = np.sin(rad), -np.cos(rad)

    # 기본 발화 전파 확률
    base_p = 0.30 + 0.04 * wind_speed - 0.015 * fuel_moisture
    base_p = np.clip(base_p, 0.05, 0.95)

    for i in range(h):
        for j in range(w):
            if grid[i, j] == BURNING:
                new[i, j] = BURNED
                # 8방향 이웃 전파
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and grid[ni, nj] == TREE:
                            # 풍향과 일치할수록 확률↑
                            align = (dj * wx + (-di) * wy)  # -1 ~ +1
                            p = base_p * (1 + 0.6 * align)
                            p = np.clip(p, 0.02, 0.98)
                            if np.random.rand() < p:
                                new[ni, nj] = BURNING
    return new


def simulate(grid, ignition_xy, hours=6, **kwargs):
    """발화점에서 N시간 시뮬레이션"""
    g = grid.copy()
    x, y = ignition_xy
    g[y, x] = BURNING
    history = [g.copy()]
    for _ in range(hours):
        g = step(g, **kwargs)
        history.append(g.copy())
    return history


def burned_area_km2(grid, cell_size_m=100):
    """피해면적 산출 (km²)"""
    burned = ((grid == BURNING) | (grid == BURNED)).sum()
    return burned * (cell_size_m ** 2) / 1_000_000


def grid_to_latlon_polygons(grid_history, center_lat, center_lon, cell_size_deg=0.002):
    """
    시뮬레이션 결과 격자를 위경도 폴리곤 좌표로 변환

    Returns:
        시간대별 [{'bounds': [[lat, lon], [lat, lon]], 'state': int}, ...] 리스트
    """
    polygons_by_time = []
    for grid in grid_history:
        h, w = grid.shape
        # 격자 중앙을 center로 정렬
        lat0 = center_lat - (h / 2) * cell_size_deg
        lon0 = center_lon - (w / 2) * cell_size_deg

        cells = []
        for i in range(h):
            for j in range(w):
                if grid[i, j] in (BURNING, BURNED):
                    lat = lat0 + i * cell_size_deg
                    lon = lon0 + j * cell_size_deg
                    cells.append({
                        'bounds': [
                            [lat - cell_size_deg/2, lon - cell_size_deg/2],
                            [lat + cell_size_deg/2, lon + cell_size_deg/2],
                        ],
                        'state': int(grid[i, j]),
                    })
        polygons_by_time.append(cells)
    return polygons_by_time


def get_ignition_latlon(ignition_xy, grid_shape, center_lat, center_lon, cell_size_deg=0.002):
    """발화점 격자 좌표 → 위경도 변환"""
    x, y = ignition_xy
    h, w = grid_shape
    lat0 = center_lat - (h / 2) * cell_size_deg
    lon0 = center_lon - (w / 2) * cell_size_deg
    return lat0 + y * cell_size_deg, lon0 + x * cell_size_deg
