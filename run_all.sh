#!/usr/bin/env bash
# Chạy toàn bộ thí nghiệm sau khi đã có data/windows.npz (xem HUONG_DAN.md).
set -e
PY=${PY:-python}

# Thí nghiệm 1: so sánh 3 mô hình ở tần số gốc 200Hz
$PY train_eval.py --model threshold
$PY train_eval.py --model rf
$PY train_eval.py --model cnn --input xyz
$PY train_eval.py --model cnn --input mag

# Thí nghiệm 2: giảm tần số lấy mẫu (tiết kiệm pin)
for r in 100 50 25 12.5; do
  $PY train_eval.py --model rf --rate $r
  $PY train_eval.py --model cnn --input mag --rate $r
done

$PY summarize.py | tee results/results_table.md
