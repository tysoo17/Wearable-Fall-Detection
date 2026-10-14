# Hướng dẫn A1: Phát hiện ngã cho người cao tuổi bằng cảm biến gia tốc

## 1. Tổng quan

```
SisFall (38 người, cảm biến ở thắt lưng, 200Hz)
   │  prepare_sisfall.py   đổi sang đơn vị g, cắt 3,2 giây quanh cú va chạm mạnh nhất
   ▼
data/windows.npz
   │  train_eval.py        3 mô hình × nhiều tần số lấy mẫu, đánh giá THEO NGƯỜI (5-fold)
   ▼
results/*.json, *.pkl  ──►  summarize.py   bảng kết quả cho README
   │
   └─ predict_phone.py     test mô hình trên dữ liệu bạn tự ghi bằng điện thoại
```

| File | Việc nó làm |
|---|---|
| `common.py` | Cắt cửa sổ, giảm tần số, tính đặc trưng |
| `models.py` | 3 mô hình: ngưỡng cố định, Random Forest, 1D-CNN |
| `prepare_sisfall.py` | Đọc dữ liệu SisFall |
| `train_eval.py` | Train và chấm điểm |
| `predict_phone.py` | Test trên dữ liệu điện thoại |
| `summarize.py`, `run_all.sh` | Chạy hết và gom bảng kết quả |

### 3 câu hỏi nghiên cứu và thí nghiệm tương ứng

| Câu hỏi | Xem ở đâu |
|---|---|
| AI có tốt hơn cách "vượt ngưỡng là ngã" mà thiết bị rẻ tiền hay dùng không? | So sánh dòng `threshold` với `rf` và `cnn` |
| Mô hình có báo động giả nhiều hơn với **người cao tuổi** không? | Cột "Báo động giả: người trẻ" so với "người cao tuổi" |
| Giảm tần số lấy mẫu (tiết kiệm pin) đến đâu thì vẫn ổn? | Các dòng 200 → 100 → 50 → 25 → 12,5Hz |
| Mô hình train ở phòng thí nghiệm có chạy được trên điện thoại thật không? | `predict_phone.py` |

### Vì sao có 2 kiểu đầu vào cho CNN (`xyz` và `mag`)

- `xyz`: dùng cả 3 trục. Có thể chính xác hơn, nhưng **phụ thuộc cách đeo thiết bị**: đeo xoay đi một góc là các trục bị đổi chỗ.
- `mag`: chỉ dùng độ lớn gia tốc tổng, **không phụ thuộc hướng đeo**. Random Forest cũng chỉ dùng đặc trưng không phụ thuộc hướng.

Khi chạy thử trên dữ liệu giả lập, mô hình `xyz` báo động giả 100% khi điện thoại bị xoay trục, còn `mag` và `rf` thì không. Hãy kiểm tra xem dữ liệu thật có cho kết quả tương tự không: đây là một phát hiện đáng viết.

## 2. Tải dữ liệu SisFall

- Bài báo gốc: [SisFall: A Fall and Movement Dataset (Sucerquia et al., 2017)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5298771/).
- Trang tải gốc của trường Universidad de Antioquia đôi khi không truy cập được. Cách nhanh nhất: tìm **"SisFall"** trên Kaggle hoặc GitHub và tải bản sao (khoảng vài trăm MB khi giải nén).
- Bản đúng có các thư mục `SA01`…`SA23` (người trẻ) và `SE01`…`SE15` (người cao tuổi), trong đó là file dạng `D01_SA01_R01.txt` (D = hoạt động thường) và `F01_SA01_R01.txt` (F = ngã).
- Mình chưa kiểm tra được link tải nào từ môi trường của mình. Nếu không tải được, báo mình ngay để chuyển phương án.

## 3. Cài đặt

```bash
python -m venv venv
venv\Scripts\activate                 # macOS/Linux: source venv/bin/activate
pip install torch          # bản CPU là đủ: mô hình rất nhỏ
pip install -r requirements.txt
```

## 4. Chạy

```bash
# 1. Chuẩn bị dữ liệu (đổi đường dẫn tới chỗ bạn giải nén SisFall)
python prepare_sisfall.py --root D:/data/SisFall_dataset

# 2a. Chạy tất cả thí nghiệm (Git Bash trên Windows)
bash run_all.sh

# 2b. Hoặc chạy từng cái
python train_eval.py --model threshold
python train_eval.py --model rf
python train_eval.py --model cnn --input mag
python train_eval.py --model rf --rate 25
python summarize.py > results/results_table.md
```

**Thời gian:** trên dữ liệu giả lập (~1.100 bản ghi), toàn bộ `run_all.sh` mất ~5 phút trên CPU 4 nhân. SisFall thật có ~4.500 bản ghi nên mình ước tính 15–30 phút trên laptop của bạn; GPU giúp phần CNN nhanh hơn.

## 5. Tự ghi dữ liệu bằng điện thoại

1. Cài app **phyphox** (miễn phí, Android/iOS). Chọn thí nghiệm **"Acceleration with g"** (bắt buộc bản *with g*).
2. Đeo điện thoại ở **thắt lưng** (đai, túi bụng, hoặc túi quần trước) để giống vị trí cảm biến SisFall.
3. Mỗi lần ghi **khoảng 10 giây và chỉ chứa 1 sự kiện**. Bấm ghi, chờ 2–3 giây, làm hành động, đứng/nằm yên vài giây, dừng.
4. Xuất file: menu ⋮ → *Export Data* → **CSV (Comma, decimal point)**. Chép file vào thư mục `phone_data/` và đặt tên theo đáp án:
   - `fall_01.csv`, `fall_02.csv`, …
   - `adl_ngoi_01.csv` (ngồi phịch xuống ghế), `adl_nhay_01.csv`, `adl_cauthang_01.csv`, `adl_dibo_01.csv`, `adl_cui_01.csv`…
5. Gợi ý số lượng: **10 lần ngã và 15–20 hoạt động thường**, ưu tiên các hoạt động dễ bị nhầm (ngồi phịch, nhảy, ngồi xổm nhanh).
6. Chạy test:
   ```bash
   python predict_phone.py --dir phone_data --model results/rf_200hz.pkl
   python predict_phone.py --dir phone_data --model results/cnn_200hz_mag.pkl
   python predict_phone.py --dir phone_data --model results/cnn_200hz_xyz.pkl
   ```

⚠️ **An toàn:** chỉ ngã xuống **nệm dày**, bắt đầu từ tư thế quỳ hoặc ngồi trên mép giường. Không ngã từ tư thế đứng thẳng xuống sàn cứng.

## 6. Đọc kết quả

- **Độ nhạy:** % cú ngã phát hiện được. Bỏ sót ngã là nguy hiểm nhất.
- **Độ đặc hiệu:** % hoạt động thường được nhận đúng là không ngã. Báo động giả = 1 − độ đặc hiệu.
- **Báo động giả người trẻ và người cao tuổi:** câu hỏi nghiên cứu chính. Nếu cột người cao tuổi cao hơn hẳn thì đó là phát hiện quan trọng; nếu không thì cũng là kết quả có giá trị, cứ viết đúng như vậy.
- `top_false_alarm_activities` trong file JSON cho biết **hoạt động nào hay bị nhầm là ngã** (mã D01–D19 tra trong bài báo SisFall). Đây là tư liệu tốt cho phần thảo luận.
- Lưu ý: SisFall chỉ có **1 người cao tuổi** (SE06) thực hiện ngã, nên đừng kết luận gì về độ nhạy riêng cho người cao tuổi.

## 7. Đã kiểm tra những gì

Mình đã chạy toàn bộ quy trình trên **dữ liệu giả lập cùng định dạng SisFall** (38 người) và dữ liệu điện thoại giả lập (lấy mẫu không đều, trục bị xoay): cả 13 thí nghiệm và phần test điện thoại đều chạy thông. Môi trường của mình không tải được SisFall thật, nên **chưa có kết quả thật**; con số trên dữ liệu giả lập không có ý nghĩa.

## 8. Lịch còn lại

| Thời gian | Việc |
|---|---|
| 13/10 sáng | Tải SisFall, cài đặt, chạy `prepare_sisfall.py` |
| 13/10 chiều | `run_all.sh`; trong lúc chờ thì ghi dữ liệu điện thoại |
| 13/10 tối | `predict_phone.py`, gửi bảng kết quả cho mình để cùng phân tích |
| 14/10 | README dạng mini-paper, đưa lên GitHub, cập nhật CV |
| 14–15/10 | Bài luận ≥500 từ, nộp hồ sơ |
