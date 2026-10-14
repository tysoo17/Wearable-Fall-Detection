# Train và vẽ biểu đồ trên KNIME

KNIME hợp nhất cho **Random Forest trên bảng đặc trưng** và **vẽ biểu đồ**. Phần CNN và thí nghiệm giảm tần số vẫn nên chạy bằng Python (`train_eval.py`).

## File dữ liệu (thư mục `knime/`)

| File | Nội dung |
|---|---|
| `features.csv` | 4.505 dòng, mỗi dòng 1 bản ghi SisFall: 10 đặc trưng + `label` (fall/adl), `subject`, `age_group` (young/elderly), `activity`, `activity_name` (tiếng Việt), `fold` (1–5, chia theo người), `split` (train/test) |
| `signals_sample.csv` | Tín hiệu gia tốc 3,2 giây của 1 bản ghi mẫu cho mỗi loại hoạt động (người trẻ và người cao tuổi), dạng dài, để vẽ Line Plot |

`split` đã chia **theo người**: 31 người train, 7 người test (4 trẻ, 3 cao tuổi). Không dùng node Partitioning chia ngẫu nhiên, vì như vậy cùng một người sẽ nằm ở cả train và test, kết quả sẽ đẹp giả tạo.

Ý nghĩa 10 đặc trưng (đơn vị g; "mag" là độ lớn gia tốc tổng):

| Cột | Ý nghĩa |
|---|---|
| `mag_max`, `mag_min`, `mag_mean`, `mag_std`, `mag_range` | Thống kê của độ lớn gia tốc trong 3,2 giây |
| `freefall_min` | Giá trị nhỏ nhất trong 1 giây trước va chạm (rơi tự do làm giá trị tụt gần 0) |
| `post_std`, `post_mean` | Độ dao động và mức trung bình sau va chạm (ngã xong thường nằm yên) |
| `posture_change_deg` | Góc thay đổi hướng trọng lực trước và sau va chạm (đứng → nằm thì gần 90°) |
| `time_above_2g` | Tỉ lệ thời gian gia tốc vượt 2g |

## Workflow 1: Train Random Forest

```
CSV Reader ──► Row Splitter ──(train)──► Random Forest Learner ──► Random Forest Predictor ──► Scorer
                    └────────(test)────────────────────────────────────────┘                ──► ROC Curve
```

1. **CSV Reader**: chọn `knime/features.csv`. Ở tab *Encoding* chọn **UTF-8** để hiện đúng tiếng Việt.
2. **Row Splitter** (hoặc *Row Filter*): điều kiện `split` = `train` → cổng trên là train, cổng dưới là test.
3. **Random Forest Learner**: *Target column* = `label`. Chỉ chọn **10 cột đặc trưng** làm đầu vào; **bỏ** `subject`, `age_group`, `activity`, `activity_name`, `fold`, `split` (nếu để lại, mô hình sẽ "nhìn trộm" đáp án). Số cây: 300.
4. **Random Forest Predictor**: nối mô hình và tập test. Tích *Append individual class probabilities*.
5. **Scorer**: so `label` với `Prediction (label)` → ma trận nhầm lẫn, độ chính xác, độ nhạy (Sensitivity của lớp `fall`), độ đặc hiệu.
6. (Tùy chọn) **ROC Curve**: lớp dương = `fall`, cột xác suất = `P (label=fall)`.

**Muốn so sánh người trẻ và người cao tuổi:** sau Predictor, thêm *Row Filter* theo `age_group` rồi mỗi nhánh nối một *Scorer*.

**Muốn chấm theo cả 5 fold** (giống Python): dùng *Table Row To Variable Loop Start* lặp `fold` = 1…5; trong vòng lặp lọc `fold` = biến lặp làm test, còn lại làm train; kết thúc bằng *Loop End* rồi *Scorer*. Kết quả 1 fold có thể lệch vài phần trăm so với bảng 5 fold trong Python, vì chỉ có 7 người test.

## Workflow 2: Biểu đồ cho README

| Biểu đồ | Node | Thiết lập | Dùng để nói gì |
|---|---|---|---|
| Tín hiệu ngã và hoạt động thường | **Line Plot** trên `signals_sample.csv` | Lọc `recording` = `F01_SA01` và `D11_SA01`; trục X `time_s`, Y `acc_mag_g` | Ngã: tụt gần 0 (rơi tự do) rồi vọt lên cao (va chạm). Ngồi phịch: chỉ có cú sốc nhỏ hơn |
| Phân bố gia tốc lớn nhất | **Box Plot** trên `features.csv` | Cột `mag_max`, nhóm theo `label` | Vì sao "vượt ngưỡng" hay báo nhầm: hai hộp chồng lên nhau |
| Hai đặc trưng tách hai lớp | **Scatter Plot** | X `mag_max`, Y `posture_change_deg`, màu theo `label` (*Color Manager* trước) | Thêm "đổi tư thế" giúp tách ngã khỏi hoạt động mạnh |
| Hoạt động bị nhầm nhiều nhất | **Bar Chart** sau Predictor | Lọc `label` = `adl` và dự đoán sai, đếm theo `activity_name` | Cho phần thảo luận |
| Đặc trưng quan trọng nhất | Cổng thống kê của **Random Forest Learner** | Cột đếm số lần mỗi đặc trưng được chọn để tách | Mô hình dựa vào gì để quyết định |

Xuất hình: chuột phải vào node → *Open view* → biểu tượng tải ảnh (PNG/SVG), rồi đưa vào README.

## Tạo lại file CSV

Nếu chạy lại `prepare_sisfall.py`, tạo lại CSV bằng:
```
python export_for_knime.py
```
