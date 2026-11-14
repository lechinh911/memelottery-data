# 2 Đáy / 2 Đỉnh → Gãy → EMA Momentum

Nghiên cứu và tối ưu chỉ báo TradingView (Pine v6) phát hiện cụm nến 2 đáy/2 đỉnh, chờ gãy neckline, rồi xác nhận momentum EMA20/50.

## Kết quả chính (v3 champion)

Walk-forward OOS trên BTC / ETH / Gold / EURUSD **1h**:

| Metric | v1 | v2 strict | **v3 Quality** |
|--------|----|-----------|----------------|
| Winrate | ~58%* | 58% | **67%** |
| Profit factor | ~1.6* | 1.45 | **1.81** |
| Expectancy (R) | ~0.22* | 0.18 | **0.25** |
| Markets PF≥1 | 3/3* | 4/4 | **4/4** |

\* v1 bị thổi bởi Gold mẫu OOS nhỏ.

Chi tiết: [`research/FINDINGS.md`](research/FINDINGS.md) · params: [`results/best_params.json`](results/best_params.json)

## Dùng trên TradingView

1. Mở [`pine/v2_strategy.pine`](pine/v2_strategy.pine)
2. Paste vào Pine Editor → Add to chart
3. Giữ preset **Quality v3 (khuyến nghị)**
4. Chạy Strategy Tester trên khung **1h** (đã tối ưu chủ yếu ở TF này)
5. Bản gốc (chỉ báo): [`pine/v1_original.pine`](pine/v1_original.pine)

## Chạy lại backtest Python

```bash
pip install -r requirements.txt
python -u src/run_research.py
python -u src/hillclimb.py
```

## Cải tiến then chốt của v3

1. **ATR stop cap (3×ATR)** — stop mô hình trên 1h thường rộng 6–7 ATR, làm WR thấp
2. **RSI mềm** — lọc long khi RSI quá cao / short khi RSI quá thấp
3. **Volume + HTF EMA + quality score + ADX/spread chặt**
4. **BE sau 1R + trail** — giữ lệnh thắng, cắt lỗ giả sớm hơn
