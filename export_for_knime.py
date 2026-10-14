"""Xuất dữ liệu ra CSV để train và vẽ biểu đồ trong KNIME.

Chạy sau prepare_sisfall.py:  python export_for_knime.py
Tạo ra trong thư mục knime/:
  features.csv        1 dòng = 1 bản ghi: 10 đặc trưng + nhãn + người + nhóm tuổi + hoạt động + fold/split theo người
  signals_sample.csv  tín hiệu gia tốc của vài bản ghi mẫu (dạng dài) để vẽ Line Plot
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import FS, features, magnitude

FEATURES = ["mag_max", "mag_min", "mag_mean", "mag_std", "mag_range",
            "freefall_min", "post_std", "post_mean", "posture_change_deg", "time_above_2g"]

ACTIVITIES = {
    "D01": "Đi bộ chậm", "D02": "Đi bộ nhanh", "D03": "Chạy bộ chậm", "D04": "Chạy bộ nhanh",
    "D05": "Lên xuống cầu thang chậm", "D06": "Lên xuống cầu thang nhanh",
    "D07": "Ngồi chậm xuống ghế cao vừa", "D08": "Ngồi nhanh xuống ghế cao vừa",
    "D09": "Ngồi chậm xuống ghế thấp", "D10": "Ngồi nhanh xuống ghế thấp",
    "D11": "Cố đứng dậy rồi ngồi phịch xuống ghế", "D12": "Nằm xuống chậm rồi ngồi dậy",
    "D13": "Nằm xuống nhanh rồi ngồi dậy", "D14": "Trở mình khi nằm",
    "D15": "Khuỵu gối rồi đứng dậy", "D16": "Cúi người không gập gối",
    "D17": "Vào ngồi ô tô rồi ra", "D18": "Vấp khi đi bộ", "D19": "Nhảy nhẹ với đồ trên cao",
    "F01": "Trượt ngã về trước khi đi", "F02": "Trượt ngã ra sau khi đi", "F03": "Trượt ngã ngang khi đi",
    "F04": "Vấp ngã về trước khi đi", "F05": "Vấp ngã về trước khi chạy", "F06": "Ngất ngã thẳng xuống khi đi",
    "F07": "Ngất ngã, chống tay vào bàn", "F08": "Ngã về trước khi cố đứng dậy",
    "F09": "Ngã ngang khi cố đứng dậy", "F10": "Ngã về trước khi ngồi xuống",
    "F11": "Ngã ra sau khi ngồi xuống", "F12": "Ngã ngang khi ngồi xuống",
    "F13": "Ngất ngã về trước khi đang ngồi", "F14": "Ngất ngã ra sau khi đang ngồi",
    "F15": "Ngất ngã ngang khi đang ngồi",
}


def main():
    d = np.load("data/windows.npz")
    X, y, subj, act = d["X"], d["y"], d["subject"], d["activity"]
    os.makedirs("knime", exist_ok=True)

    df = pd.DataFrame(features(X, FS), columns=FEATURES).round(4)
    df.insert(0, "label", np.where(y == 1, "fall", "adl"))
    df.insert(1, "subject", subj)
    df.insert(2, "age_group", np.where(np.char.startswith(subj, "SE"), "elderly", "young"))
    df.insert(3, "activity", act)
    df.insert(4, "activity_name", [ACTIVITIES.get(a, a) for a in act])
    # Fold chia THEO NGƯỜI (giống train_eval.py); split: fold 1 làm test, còn lại train.
    fold = np.zeros(len(y), int)
    for k, (_, te) in enumerate(GroupKFold(5).split(X, y, subj)):
        fold[te] = k + 1
    df.insert(5, "fold", fold)
    df.insert(6, "split", np.where(fold == 1, "test", "train"))
    df.to_csv("knime/features.csv", index=False)

    # Tín hiệu mẫu: mỗi loại hoạt động lấy 1 bản ghi của 1 người trẻ và 1 người cao tuổi.
    rows = []
    for a in sorted(set(act)):
        for group in ("SA", "SE"):
            idx = np.where((act == a) & np.char.startswith(subj, group))[0]
            if not len(idx):
                continue
            i = idx[0]
            m = magnitude(X[i])
            for t in range(0, X.shape[1], 2):  # 100Hz là đủ để vẽ
                rows.append([f"{a}_{subj[i]}", a, ACTIVITIES.get(a, a), subj[i],
                             "fall" if y[i] else "adl", round(t / FS, 3),
                             *np.round(X[i, t], 4), round(float(m[t]), 4)])
    pd.DataFrame(rows, columns=["recording", "activity", "activity_name", "subject", "label",
                                "time_s", "acc_x_g", "acc_y_g", "acc_z_g", "acc_mag_g"]
                 ).to_csv("knime/signals_sample.csv", index=False)

    print(f"knime/features.csv: {len(df)} dòng, {len(df.columns)} cột")
    print(f"knime/signals_sample.csv: {len(rows)} dòng")


if __name__ == "__main__":
    main()
