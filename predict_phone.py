"""Test mô hình đã train trên dữ liệu bạn tự ghi bằng điện thoại (app phyphox).

Đặt tên file theo đáp án:  fall_01.zip, fall_02.zip, adl_ngoi_01.zip, adl_dibo_01.zip ...
(nhận cả file .zip phyphox xuất ra lẫn file .csv)
Chạy:  python predict_phone.py --dir phone_data --model results/rf_50hz.pkl
"""
import argparse
import glob
import json
import os
import pickle
import re
import zipfile

import numpy as np
import pandas as pd

from common import FS, downsample, impact_window

G = 9.81


def load_phyphox(path):
    """Đọc CSV của phyphox: cột 1 là thời gian (s), 3 cột tiếp theo là gia tốc x, y, z (m/s²).

    Phải dùng thí nghiệm "Acceleration with g" (có trọng lực), giống cảm biến trong SisFall.
    """
    if path.lower().endswith(".zip"):
        # phyphox xuất ra file .zip chứa "Raw Data.csv"; đọc thẳng, không cần giải nén.
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv") and "meta" not in n.lower())
            with z.open(name) as f:
                df = pd.read_csv(f, sep=None, engine="python")
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    if "acceleration" not in df.columns[1].lower():
        raise ValueError(f"không phải dữ liệu gia tốc (cột: {df.columns[1]}). "
                         "Hãy ghi lại bằng thí nghiệm 'Acceleration with g'")
    t = df.iloc[:, 0].to_numpy(float)
    acc = df.iloc[:, 1:4].to_numpy(float) / G
    # Điện thoại lấy mẫu không đều -> nội suy về đúng 200Hz như SisFall.
    t_new = np.arange(t[0], t[-1], 1 / FS)
    return np.stack([np.interp(t_new, t, acc[:, k]) for k in range(3)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="phone_data")
    ap.add_argument("--model", required=True, help="file .pkl trong results/")
    args = ap.parse_args()

    model = pickle.load(open(args.model, "rb"))
    rate = float(re.search(r"_([\d.]+)hz", os.path.basename(args.model)).group(1))
    rate = int(rate) if rate == int(rate) else rate

    rows = []
    paths = sorted(glob.glob(os.path.join(args.dir, "*.csv")) + glob.glob(os.path.join(args.dir, "*.zip")))
    for path in paths:
        name = os.path.basename(path)
        if not name.lower().startswith(("fall", "adl")):
            print(f"Bỏ qua {name}: tên file phải bắt đầu bằng fall_ hoặc adl_")
            continue
        truth = 1 if name.lower().startswith("fall") else 0
        try:
            sig = load_phyphox(path)
        except ValueError as e:
            print(f"Bỏ qua {name}: {e}")
            continue
        w = impact_window(sig)[None]
        p = float(model.predict_proba(downsample(w, rate), rate)[0])
        rows.append((name, truth, round(p, 3)))
        print(f"{name:30s} đáp án={'NGÃ' if truth else 'thường':6s} "
              f"mô hình={'NGÃ' if p >= 0.5 else 'thường':6s} (xác suất ngã {p:.2f})")

    if not rows:
        raise SystemExit(f"Không có file .csv hoặc .zip nào trong {args.dir}")
    y = np.array([r[1] for r in rows])
    pred = np.array([r[2] >= 0.5 for r in rows]).astype(int)
    print()
    if (y == 1).any():
        print(f"Độ nhạy (phát hiện được ngã): {(pred[y == 1] == 1).mean():.2f}  ({(y == 1).sum()} cú ngã)")
    if (y == 0).any():
        print(f"Báo động giả:                 {(pred[y == 0] == 1).mean():.2f}  ({(y == 0).sum()} hoạt động thường)")

    # Lưu kết quả để make_figures.py vẽ biểu đồ so sánh SisFall và điện thoại.
    model_name = os.path.basename(args.model)[:-4]
    out = os.path.join(os.path.dirname(args.model), f"phone_{model_name}.json")
    json.dump({"model": model_name,
               "falls": int((y == 1).sum()), "falls_detected": int((pred[y == 1] == 1).sum()),
               "adl": int((y == 0).sum()), "false_alarms": int((pred[y == 0] == 1).sum()),
               "recordings": [{"file": n, "truth": t, "p_fall": p} for n, t, p in rows]},
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    main()
