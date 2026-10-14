"""Gom kết quả trong results/ thành bảng Markdown (dán thẳng vào README)."""
import glob
import json
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = [json.load(open(p)) for p in glob.glob(f"{folder}/*.json")]
order = {"threshold": 0, "rf": 1, "cnn": 2}
rows.sort(key=lambda r: (order[r["model"]], r["input"], -r["rate_hz"]))


def fmt(v):
    return "-" if v is None else f"{v:.3f}"


print("| Mô hình | Đầu vào | Tần số (Hz) | Độ nhạy | Độ đặc hiệu | F1 | Báo động giả: người trẻ | Báo động giả: người cao tuổi |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['model']} | {r['input']} | {r['rate_hz']} | {fmt(r['sensitivity'])} | {fmt(r['specificity'])} | "
          f"{fmt(r['f1'])} | {fmt(r['false_alarm_young'])} | {fmt(r['false_alarm_elderly'])} |")
