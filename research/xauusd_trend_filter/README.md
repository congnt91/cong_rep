# XAUUSD: lệnh chờ tại mốc phản ứng có tốt hơn khi lọc thuận Trend / Swing?

Backtest cho câu hỏi: với lệnh chờ đặt tại các mốc (PDH/PDL, pivot ngày, pivot phiên 8h, pivot 12h,
Fibo 50%, EMA) như module "Auto mốc phản ứng" của Nola-X, nếu chỉ đánh thuận Trend/Swing, hoặc lệnh
ngược trend thì TP ngắn + lot nhỏ, hiệu quả có tốt hơn không?

## Dữ liệu

| Giai đoạn | Khung | Thời gian | Nguồn |
|---|---|---|---|
| A | M15, giờ server MT5 | 05/2012 → 03/2022 | `ejtraderLabs/historical-data` |
| B | M1, UTC → giờ server | 10/2025 → 09/2026 | TradingView `FX:XAUUSD`, ghép từ lịch sử commit của `navtemmt/xauusd-autodata` |

Chưa có dữ liệu 03/2022 → 10/2025 (không tìm được nguồn công khai truy cập được).

## Quy tắc mô phỏng

- Mốc làm mới đầu mỗi phiên 8h (00/08/16 giờ server). Mốc dưới giá → Buy Limit, trên giá → Sell Limit.
  Gộp mốc cách nhau < 0,25 ATR, bỏ mốc < 0,15 ATR hoặc > 3 ATR. Lệnh chờ hết hạn sau 8h.
- SL = 0,75 × ATR(14) H1. TP = mốc kế tiếp theo hướng lệnh (tối đa 3R), bỏ lệnh nếu RR < 1.
  Đóng lệnh sau tối đa 24h.
- Spread 0,30 USD/lệnh; SL bị gap khớp ở giá mở; nến chạm cả SL và TP tính là thua.
- Bộ lọc tính trên nến đã đóng: Trend H4 (EMA50 vs EMA200), Trend D1 (giá vs EMA50),
  Swing H1/H4 (BOS với fractal 3/2 nến), EMA20/50 H1, "Trend mạnh" = Trend H4 + Trend D1 + Swing H4 đồng thuận.

## Kết quả chính (TP cấu trúc, E = kỳ vọng R/lệnh)

| | A: 10.072 lệnh | B: 998 lệnh |
|---|---|---|
| Tất cả lệnh | −0,162 | −0,022 |
| Thuận / ngược Trend H4 | −0,162 / −0,162 | −0,024 / −0,021 |
| Thuận / ngược Swing H1 | −0,148 / −0,177 | −0,027 / −0,013 |
| Thuận / ngược Trend mạnh | −0,131 / −0,165 | +0,001 / +0,047 |
| Ngược trend: lot ½ + TP 0,5R (lọc Trend H4) | −0,184 | −0,052 |

- Lọc thuận Trend gần như không đổi kỳ vọng mỗi lệnh; chủ yếu làm giảm số lệnh.
- TP ngắn cho lệnh ngược trend làm kết quả xấu đi: lệnh ngược chạy được ≥ 1R với tỷ lệ bằng lệnh thuận (47,7% vs 47,7% ở A).
- Trend tạo khác biệt rõ hơn với lệnh Stop (breakout) và SL rộng: ở B, Stop thuận Trend H4, SL 1,5 ATR,
  TP 1,5–2R đạt khoảng +0,18R/lệnh (ngược ≈ 0); ở A cùng cấu hình vẫn âm sau phí.

## Chạy lại

```bash
pip install pandas numpy
python fetch_data.py        # clone dữ liệu public -> data/
python run_sim.py old       # cấu hình giống Nola-X -> out/ex_old_*.pkl
python run_sim.py new
python run_grid.py old      # lưới SL x Limit/Stop -> out/grid_*.pkl
python run_grid.py new
python analysis.py          # -> out/results.json
```

## Phần 2: phương án cải thiện Auto mốc (improve.py, run_improve.py, run_combo.py)

Thử 37 cấu hình theo quy trình chọn trên A1 (2012–2017), kiểm tra trên A2 (2017–2022) và B (2025–26).

| Kỳ vọng R/lệnh | A1 | A2 | B | Lệnh/tháng |
|---|---|---|---|---|
| Hiện tại: Limit tại mốc | −0,165 | −0,159 | −0,022 | 82–87 |
| Xác nhận nến M15 (vào ở giá đóng nến, SL 0,75 ATR sau mốc) | −0,067 | −0,107 | −0,026 | 36–40 |
| Xác nhận M15 + ATR H1 ≥ trung vị 20 ngày + chốt 50% tại 1R | −0,023 | −0,035 | +0,026 | 16–19 |

- Lợi ích của xác nhận nến đến từ việc loại nhóm lệnh giá xuyên thẳng qua mốc (−0,25R/lệnh ở A1).
- Không có tác dụng nhất quán: dời SL hoà vốn, trailing, time stop, TP ngắn/xa, SL rộng, Limit lệch mốc,
  lọc phiên, thứ, số mốc gộp, RSI, khoảng cách EMA20, tốc độ tiếp cận.
- Cấu hình đề xuất vẫn có khoảng tin cậy chứa 0 ở cả ba giai đoạn: giảm lỗ, chưa phải hệ thống có lãi.

```bash
python run_improve.py       # -> out/improve.json (thử nghiệm đơn lẻ, bộ lọc)
python run_combo.py         # -> out/combo.json (cấu hình kết hợp, đường vốn)
```
