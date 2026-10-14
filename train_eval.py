"""Train và đánh giá mô hình phát hiện ngã, chia dữ liệu THEO NGƯỜI (5-fold).

Ví dụ:
  python train_eval.py --model threshold
  python train_eval.py --model rf --rate 50
  python train_eval.py --model cnn --rate 50 --input mag
"""
import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
from sklearn.model_selection import GroupKFold

from common import RATES, downsample
from models import CNNModel, RFModel, ThresholdModel


def build(args):
    if args.model == "threshold":
        return ThresholdModel()
    if args.model == "rf":
        return RFModel()
    return CNNModel(args.input, args.epochs)


# ---------- Đánh giá ----------

def metrics(y, pred, subj, act):
    elderly = np.char.startswith(subj, "SE")
    adl = y == 0
    fa = pred[adl] == 1
    false_alarm_acts = Counter(act[adl][fa])
    total_acts = Counter(act[adl])
    # So sánh công bằng: chỉ tính các hoạt động mà phần lớn người cao tuổi đều làm.
    # (D06, D13, D18, D19 chỉ 1 người cao tuổi thực hiện.)
    shared = adl & ~np.isin(act, ["D06", "D13", "D18", "D19"])
    return {
        "sensitivity": float((pred[y == 1] == 1).mean()),
        "specificity": float((pred[adl] == 0).mean()),
        "false_alarm_young": float((pred[adl & ~elderly] == 1).mean()),
        "false_alarm_elderly": float((pred[adl & elderly] == 1).mean()) if (adl & elderly).any() else None,
        "false_alarm_young_shared": float((pred[shared & ~elderly] == 1).mean()),
        "false_alarm_elderly_shared": float((pred[shared & elderly] == 1).mean()) if (shared & elderly).any() else None,
        "f1": float(2 * ((pred == 1) & (y == 1)).sum()
                    / max((pred == 1).sum() + (y == 1).sum(), 1)),
        # Hoạt động nào hay bị nhầm là ngã nhất -> tư liệu cho phần thảo luận.
        "top_false_alarm_activities": [
            [str(a), int(n), round(n / total_acts[a], 3)] for a, n in false_alarm_acts.most_common(5)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/windows.npz")
    ap.add_argument("--model", choices=["threshold", "rf", "cnn"], default="rf")
    ap.add_argument("--rate", type=float, choices=list(RATES), default=200)
    ap.add_argument("--input", choices=["xyz", "mag"], default="xyz", help="chỉ dùng cho cnn")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    rate = int(args.rate) if args.rate == int(args.rate) else args.rate

    d = np.load(args.data)
    X = downsample(d["X"], rate)
    y, subj, act = d["y"], d["subject"], d["activity"]

    # Out-of-fold: mỗi người chỉ được dự đoán bởi mô hình chưa từng thấy người đó.
    prob = np.zeros(len(y), np.float32)
    for k, (tr, te) in enumerate(GroupKFold(args.folds).split(X, y, subj)):
        m = build(args).fit(X[tr], y[tr], rate)
        prob[te] = m.predict_proba(X[te], rate)
        print(f"  fold {k + 1}/{args.folds} xong ({len(set(subj[te]))} người test)")
    pred = (prob >= 0.5).astype(int)

    name = f"{args.model}_{rate}hz" + (f"_{args.input}" if args.model == "cnn" else "")
    res = {"name": name, "model": args.model, "rate_hz": rate,
           "input": args.input if args.model == "cnn" else "features",
           **metrics(y, pred, subj, act)}
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(args.out_dir, f"{name}.json"), "w"), indent=2)

    # Train lại trên toàn bộ dữ liệu để dùng cho test điện thoại.
    final = build(args).fit(X, y, rate)
    if args.model == "cnn":
        final.model = final.model.cpu()
        final.device = "cpu"
    pickle.dump(final, open(os.path.join(args.out_dir, f"{name}.pkl"), "wb"))

    fe = res["false_alarm_elderly"]
    print(f"\n[{name}] độ nhạy={res['sensitivity']:.3f}  độ đặc hiệu={res['specificity']:.3f}  F1={res['f1']:.3f}")
    print(f"  báo động giả: người trẻ={res['false_alarm_young']:.3f}  "
          f"người cao tuổi={'-' if fe is None else f'{fe:.3f}'}")
    print(f"  hoạt động hay bị nhầm là ngã: {res['top_false_alarm_activities']}")


if __name__ == "__main__":
    main()
