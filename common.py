"""Hàm dùng chung: cắt cửa sổ quanh cú va chạm, giảm tần số, tính đặc trưng."""
import numpy as np

FS = 200                 # tần số gốc của SisFall
WIN = 640                # cửa sổ 3,2 giây ở 200Hz
BEFORE = 240             # 1,2 giây trước đỉnh va chạm, 2 giây sau
RATES = {200: 1, 100: 2, 50: 4, 25: 8, 12.5: 16}


def magnitude(acc):
    return np.linalg.norm(acc, axis=-1)


def impact_window(acc):
    """Lấy đoạn 3,2 giây quanh lúc gia tốc tổng lớn nhất.

    Với hoạt động thường, đây là khoảnh khắc "giống ngã nhất" (ví dụ ngồi phịch xuống),
    nên mô hình được kiểm tra ở tình huống khó nhất.
    """
    mag = magnitude(acc)
    peak = int(np.argmax(mag))
    start = min(max(peak - BEFORE, 0), max(len(acc) - WIN, 0))
    w = acc[start:start + WIN]
    if len(w) < WIN:  # bản ghi quá ngắn: lặp lại mẫu cuối cho đủ độ dài
        w = np.concatenate([w, np.repeat(w[-1:], WIN - len(w), axis=0)])
    return w.astype(np.float32)


def downsample(X, rate):
    """Giả lập thiết bị lấy mẫu thưa hơn: lấy trung bình từng nhóm mẫu liên tiếp."""
    k = RATES[rate]
    if k == 1:
        return X
    n, t, c = X.shape
    return X.reshape(n, t // k, k, c).mean(axis=2)


def features(X, rate=200):
    """Đặc trưng không phụ thuộc cách đeo thiết bị (chỉ dùng độ lớn và góc thay đổi),
    nhờ vậy mô hình dùng được cho điện thoại đặt theo hướng bất kỳ."""
    fs = rate
    mag = magnitude(X)
    peak = mag.argmax(axis=1)
    feats = []
    for i in range(len(X)):
        m, p = mag[i], peak[i]
        pre = m[max(0, p - int(fs)):p] if p > 0 else m[:1]
        post = m[p + int(0.5 * fs):]
        post = post if len(post) else m[-1:]
        # Hướng trọng lực trước và sau va chạm: ngã thường làm cơ thể đổi tư thế (đứng -> nằm).
        g_before = X[i, :max(1, p - int(0.5 * fs))].mean(axis=0)
        g_after = X[i, min(len(m) - 1, p + int(fs)):].mean(axis=0)
        cos = np.dot(g_before, g_after) / (np.linalg.norm(g_before) * np.linalg.norm(g_after) + 1e-8)
        feats.append([
            m.max(), m.min(), m.mean(), m.std(), m.max() - m.min(),
            pre.min(),                   # rơi tự do trước va chạm: gia tốc tổng tụt gần 0
            post.std(), post.mean(),     # sau ngã thường nằm yên
            np.degrees(np.arccos(np.clip(cos, -1, 1))),
            (m > 2.0).mean(),            # tỉ lệ thời gian gia tốc > 2g
        ])
    return np.array(feats, np.float32)
