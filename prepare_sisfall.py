"""Đọc toàn bộ SisFall, đổi sang đơn vị g, cắt 1 cửa sổ quanh cú va chạm mỗi bản ghi.

Chạy:  python prepare_sisfall.py --root duong/dan/toi/SisFall_dataset
Thư mục chỉ cần chứa các file dạng D01_SA01_R01.txt / F05_SE06_R03.txt ở bất kỳ cấp nào.
"""
import argparse
import glob
import os
import re

import numpy as np

from common import impact_window

NAME = re.compile(r"^([DF])(\d\d)_(S[AE]\d\d)_R(\d\d)\.txt$")
# Gia tốc kế ADXL345: dải ±16g, 13 bit -> đổi giá trị thô sang g.
ADXL_SCALE = (2 * 16) / (2 ** 13)


def read_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                rows.append([float(v) for v in parts[:3]])  # 3 cột đầu là ADXL345
    return np.array(rows) * ADXL_SCALE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/windows.npz")
    args = ap.parse_args()

    X, y, subj, act = [], [], [], []
    files = sorted(glob.glob(os.path.join(args.root, "**", "*.txt"), recursive=True))
    for path in files:
        m = NAME.match(os.path.basename(path))
        if not m:
            continue
        acc = read_file(path)
        if len(acc) < 100:
            continue
        X.append(impact_window(acc))
        y.append(1 if m.group(1) == "F" else 0)
        subj.append(m.group(3))
        act.append(m.group(1) + m.group(2))
    if not X:
        raise SystemExit(f"Không tìm thấy file SisFall nào trong {args.root}")

    X, y, subj, act = np.stack(X), np.array(y), np.array(subj), np.array(act)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, subject=subj, activity=act)

    elderly = np.char.startswith(subj, "SE")
    print(f"{len(y)} bản ghi từ {len(set(subj))} người -> {args.out}")
    print(f"  Người trẻ:     {(~elderly & (y == 0)).sum()} hoạt động thường, {(~elderly & (y == 1)).sum()} cú ngã")
    print(f"  Người cao tuổi: {(elderly & (y == 0)).sum()} hoạt động thường, {(elderly & (y == 1)).sum()} cú ngã")


if __name__ == "__main__":
    main()
