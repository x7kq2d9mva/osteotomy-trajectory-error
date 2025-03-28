# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

files = ['Louie_mand.csv', 'Louie_max.csv', 'Lo_mand.csv', 'Lo_max.csv']

def find_min_dis(file_path):
    df = pd.read_csv(file_path)
    data = {}
    for _, row in df.iterrows():
        point = row['Point']  # e.g., 'LO L1-6'
        parts = point.split()
        if len(parts) != 2:
            continue  # 跳過不符合格式的資料
        
        prefix, name = parts
        line_id, idx_str = name.split('-')  # 'L1', '6'
        idx = int(idx_str) - 1  # 0-based index
        is_stl = 0 if prefix == 'LO' else 1

        if line_id not in data:
            data[line_id] = [[None]*10, [None]*10]  # index 0: LO, 1: CT

        coords = (
            row['Pre-Op 3D [X]'],
            row['Pre-Op 3D [Y]'],
            row['Pre-Op 3D [Z]']
        )
        data[line_id][is_stl][idx] = coords    
    
    
    for line_id in data:
        if (len(data[line_id][0]) != len(data[line_id][1])):
            print(f'Error: {line_id}')
            continue
        # for i in [0, 1]:
        #     print(f'data[{line_id}][{i}]: {data[line_id][i]}')
        
        min_dis = compute_distances(data[line_id][0], data[line_id][1])
        
        print(f"{line_id}: {min_dis}")
        min_dis = [x for x in min_dis if x is not None]
        x_values = np.linspace(0, 1, len(min_dis))
        plt.plot(x_values, min_dis, marker='o', label=line_id, linewidth=0.5)
        plt.axhline(y=2.0, color='red', linewidth=1)
        


def point_to_segment_dist(p, a, b):
    """計算點 p 到線段 ab 的最短距離"""
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    projection = a + t * ab
    return np.linalg.norm(p - projection)

def compute_distances(p_list, polyline):
    """
    p_list: 要測量的點
    polyline: 構成折線的點
    """
    dists = []
    for p in p_list:
        if p is None:
                continue
        min_dist = float('inf')
        for i in range(len(polyline) - 1):
            a, b = polyline[i], polyline[i+1]
            if a is None or b is None:
                continue
            dist = point_to_segment_dist(p, a, b)
            if dist < min_dist:
                min_dist = dist
        dists.append(min_dist)
    dists = remove_outliers(dists)
    return dists

def remove_outliers(dists):
    """將離群值標記為 None"""
    dists = np.array(dists)
    q1 = np.percentile(dists, 25)
    q3 = np.percentile(dists, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [dist if lower_bound <= dist <= upper_bound else None for dist in dists]

for f in files:
    plt.figure(figsize=(10, 6))
    print(f'Processing {f}...')
    find_min_dis('./raw/extracted' / Path(f))

    # 添加標題和標籤
    plt.title(f"Shortest Distances to Curve: {f}")
    plt.xticks([])  # 清空 x 軸刻度
    plt.ylabel("Error(mm)")
    plt.legend(title="Line ID", bbox_to_anchor=(1, 1))
    plt.grid(axis='x', visible=False)
    plt.grid(axis='y', visible=True, zorder=0)
    plt.savefig('./fig/' / Path(f).with_suffix(".png"))