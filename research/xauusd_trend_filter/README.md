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

## Phần 3: lọc theo khung giờ (run_hours.py)

9 khung giờ định nghĩa trước theo lịch thị trường (giờ server MT5 = NY + 7; giờ VN = server + 4 mùa hè / + 5 mùa đông).
Chọn khung cấm trên A1, kiểm tra trên A2 và B.

| Kỳ vọng R/lệnh | A1 | A2 | B |
|---|---|---|---|
| Limit hiện tại | −0,165 | −0,159 | −0,022 |
| Limit, cấm 01–02h và 10–12h server (chọn trên A1) | −0,148 | −0,145 | +0,021 |
| Lệnh trong 2 khung bị cấm | −0,248 | −0,226 | −0,236 |
| Xác nhận M15, chỉ PDH/PDL, chỉ 02–05h server | +0,127 | +0,081 | +0,584 (19 lệnh) |

- Nên cấm: 01–02h server (mở cửa lại, tỷ lệ thắng 30–37%) và 10–12h server (London mở cửa, RR trung vị chỉ 1,34–1,38).
- Giờ Mỹ vỡ mốc nhiều hơn (32–43% so với 13–26% phiên Á) nhưng kỳ vọng không xấu hơn; không nên cấm.
  Lệnh Stop giờ Mỹ cũng không lãi.
- Với cấu hình xác nhận + lọc ATR + chốt 50%, lọc giờ không cải thiện thêm ở 2012–2022.

```bash
python run_hours.py         # -> out/hours.json
```

## Phần 4: kết hợp mốc với chỉ báo (indicators.py, run_indicators.py, run_confluence.py)

46 chỉ báo thuộc 7 nhóm (xu hướng, vị trí, dao động, biến động, volume, hành vi giá, cấu trúc), đo trên nến
đã đóng tại lúc vào lệnh, quy đổi theo hướng lệnh. Mỗi chỉ báo: chọn trên A1 giữ 40% lệnh ở đuôi thấp hoặc cao,
kiểm tra trên A2 và B. "Vững" = cải thiện ở A2 có khoảng tin cậy > 0 và cải thiện ở B > 0
(may rủi thuần tuý: khoảng 11/46 cùng dấu, 0–2/46 vững).

| | Cùng dấu A2 và B | Vững |
|---|---|---|
| Limit tại mốc | 17/46 | 4/46: khoảng cách tới VWAP, mốc trùng swing, RSI M15, NR7 D1 (+0,02 đến +0,07R/lệnh ở A2) |
| Xác nhận nến M15 | 9/46 | 0/46 |

Mốc trùng đường chỉ báo (Limit, E trùng / không trùng):

| | A1 | A2 | B |
|---|---|---|---|
| Trùng đỉnh/đáy swing H1 gần đây | −0,262 / −0,137 | −0,201 / −0,147 | −0,130 / +0,003 |
| Trùng dải Bollinger ngoài H1 | −0,212 / −0,157 | −0,252 / −0,142 | −0,071 / −0,012 |
| Trùng EMA50 H1 | −0,273 / −0,155 | −0,259 / −0,151 | +0,108 / −0,033 |
| Trùng số tròn 50 USD | −0,261 / −0,162 | −0,249 / −0,155 | −0,035 / −0,018 |

Cấu hình ghép (E R/lệnh):

| | A1 | A2 | B |
|---|---|---|---|
| Limit hiện tại | −0,165 | −0,159 | −0,022 |
| Limit, tránh mốc trùng swing H1 và dải Bollinger ngoài | −0,137 | −0,135 | +0,006 |
| Như trên + cấm 01–02h, 10–12h server | −0,120 | −0,122 | +0,043 |
| Xác nhận M15 + ATR ≥ trung vị + chốt 50% | −0,023 | −0,035 | +0,026 |
| Như trên + tránh dải Bollinger ngoài | −0,030 | −0,038 | +0,073 |

- RSI/Stochastic/CCI quá bán, MACD, Supertrend, Ichimoku, PSAR, phân kỳ RSI: cải thiện 0,00–0,03R, không phân biệt được với may rủi.
- Nhóm biến động (ATR, ADX, độ rộng Bollinger) giúp ở 2012–22, không rõ ở 2025–26; bộ lọc ATR trong cấu hình đề xuất đã dùng phần này.
- Mốc trùng swing gần đây, dải Bollinger ngoài, EMA50 hay số tròn không mạnh hơn mà yếu hơn (nơi tập trung stop).
- Mô hình máy học (logistic, ridge, gradient boosting) giữ 30% lệnh tốt nhất: gradient boosting đạt +0,41 đến +0,57R trên dữ liệu
  huấn luyện nhưng −0,07 đến +0,03R ở B; không mô hình nào ổn định trên cả hai giai đoạn kiểm tra.
- Các bộ lọc "tránh" được chọn sau khi xem cả ba giai đoạn nên kết quả ghép lạc quan hơn thực tế.

```bash
pip install scikit-learn
python run_indicators.py    # -> out/indicators.json (quét 46 chỉ báo, mô hình)
python run_confluence.py    # -> out/confluence.json (trùng mốc, cấu hình ghép)
```

## Phần 5: tỉ lệ bật tại mốc và chiến lược ăn đoạn bật (bounce.py, run_bounce.py)

Mỗi lần chạm mốc (mọi mốc tĩnh, không lọc RR): giá đi thuận x trước khi đi ngược y hay không, trong 24h.
So với mốc giả (mức giá ngẫu nhiên cùng phiên/hướng/khoảng cách, cách mốc thật ≥ 0,25 ATR) và lý thuyết
giá ngẫu nhiên P = y / (x + y). B dùng nến M1; A chỉ có M15 nên đo thiếu các cú bật nhỏ trong nến.
Sai số đo như nhau với mốc thật và mốc giả, nên dùng chênh lệch thật − giả; kỳ vọng ước tính = chênh lệch − chi phí.

Tỉ lệ bật trước khi ngược 0,75 ATR (≈ 14 USD), 2025–26, chưa trừ spread:

| Bật ít nhất | 1 USD | 2 USD | 5 USD | 10 USD |
|---|---|---|---|---|
| Mốc thật, Limit | 86,8% | 82,3% | 69,9% | 57,7% |
| Mốc giả, Limit | 88,3% | 83,2% | 72,1% | 57,0% |
| Mốc thật, xác nhận M15 | 92,3% | 86,6% | 73,3% | 58,8% |
| Lý thuyết | 93,3% | 87,4% | 73,8% | 58,9% |

Cấu hình TP mỏng (kỳ vọng ước tính, R = 0,75 ATR; cột cuối USD/lệnh 0,01 lot năm 2025–26):

| | A1 | A2 | B | B, USD |
|---|---|---|---|---|
| Limit, TP 0,1 ATR (≈ 2 USD), SL 0,75 ATR | −0,038 | −0,030 | −0,037 | −0,71 |
| Limit, TP 0,1 ATR, không SL (đóng sau 24h) | −0,023 | −0,050 | +0,001 | −0,02 |
| Xác nhận M15, TP 0,1 ATR, SL 0,75 ATR | −0,024 | −0,020 | −0,020 | −0,12 |
| Mốc giả (đối chứng) | −0,022 | −0,022 | −0,022 | −0,30 |

- Mốc thật không bật nhiều hơn giá ngẫu nhiên (kém 1–2 điểm % với cú bật nhỏ). Lệnh xác nhận M15 bật đúng bằng lý thuyết.
- TP mỏng cho tỉ lệ thắng 75–96% nhưng luôn thấp hơn tỉ lệ cần để hoà vốn; kỳ vọng ≈ −spread.
- Không SL: 2025–26 có 3,8% lệnh không bật nổi 0,1 ATR trong 24h, lỗ trung bình 72 USD/0,01 lot (≈ 39 lệnh thắng), tệ nhất 264 USD.
- PDH/PDL bật kém giá ngẫu nhiên 6–10 điểm %, khung 10–12h server kém 5–15 điểm %; pivot phiên 8h ngang ngẫu nhiên.

```bash
python run_bounce.py        # -> out/bounce.json
```

## Xuất dữ liệu từng lệnh (export_results.py)

```bash
python export_results.py    # -> out/export/lenh_limit.csv, lenh_xac_nhan.csv, cham_moc.csv
```

- `lenh_limit.csv`: lệnh Limit tại mốc như Nola-X (SL 0,75 ATR, TP mốc kế tiếp, RR ≥ 1), kèm trend/swing và đặc trưng lúc khớp, kết quả R và USD / 0,01 lot.
- `lenh_xac_nhan.csv`: xác nhận nến M15 + chốt 50% tại 1R; lọc `atr_rel >= 1` là cấu hình đề xuất Phần 2.
- `cham_moc.csv`: mọi lần chạm mốc (thật/giả, Limit/xác nhận), độ bật trước khi ngược 0,5/0,75/1,0 ATR và độ ngược trước khi bật 0,05/0,1/0,2 ATR.
- Cột `thoi_gian_server` là giờ MT5 (NY + 7), `gio_vn` là giờ Việt Nam. File UTF-8 có BOM để Excel đọc đúng tiếng Việt.
