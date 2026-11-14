# Nghiên cứu 2 Đáy / 2 Đỉnh → Gãy → EMA Momentum

## Mục tiêu

Tăng chất lượng hệ thống qua vòng lặp nghiên cứu → backtest → chỉnh filter.

**Bài học:** tối đa winrate (Quality) làm ít lệnh, đợi lâu, lãi thực thấp.  
Default chuyển sang **Active** (nhiều lệnh hơn, vẫn expectancy dương). **Core** cho lãi full-sample tốt nhất.

## Universe backtest

| Symbol | TF | Kỳ |
|--------|----|----|
| BTC-USD | 1h | ~730 ngày |
| ETH-USD | 1h | ~730 ngày |
| GC=F (vàng) | 1h | ~730 ngày |
| EURUSD=X | 1h | ~730 ngày |

Chi phí mô phỏng: fee 4 bps + slip 2 bps mỗi phía. Walk-forward: 60% train / 40% test.

## Tiến triển (OOS, thị trường đủ mẫu)

| Version | OOS WR | OOS PF | OOS ExpR | Thị trường PF≥1 |
|---------|--------|--------|----------|-----------------|
| v1 original | ~58.5%* | ~1.60* | ~0.22* | 3/3 (Gold mẫu nhỏ) |
| v2 strict | 57.9% | 1.45 | 0.18 | **4/4** |
| v3 wr3 (champion) | **67.0%** | **1.81** | **0.25** | **4/4** |

\* v1 bị thổi phồng bởi Gold OOS chỉ 5 lệnh (WR 100%).

### Champion v3 OOS theo thị trường

| Market | Trades | WR | PF | ExpR | MaxDD |
|--------|--------|----|----|------|-------|
| BTC-USD 1h | 15 | 60% | 1.05 | +0.02 | -8.7% |
| ETH-USD 1h | 18 | 72% | 2.42 | +0.39 | -5.2% |
| GC=F 1h | 18 | 67% | 2.11 | +0.37 | -4.0% |
| EURUSD 1h | 26 | 69% | 1.68 | +0.21 | -3.0% |

Full-sample cũng dương cả 4 thị trường (BTC PF 1.10, ETH 1.26, Gold 2.66, EURUSD 1.14).

## Chẩn đoán lỗ quan trọng

1. **Stop theo đáy/đỉnh mô hình quá rộng** (~6–7 ATR trên 1h) → hầu hết lệnh chạm stop trước khi tới TP.  
   **Fix:** `atr_stop_cap = 3.0` (siết stop tối đa 3×ATR khỏi entry).
2. **EURUSD long yếu** khi không có RSI / cancel fake break.  
   **Fix:** RSI soft (long ≤72 / short ≥28) + `cancel_retest = true`.
3. **Lọc quá chặt xếp chồng** (extension + close-dir + RSI tight + quality cao) → hết lệnh.  
   **Fix:** chỉ giữ filter có edge rõ, không stack tối đa.

## Filter thắng cuộc (v3)

- Volume xác nhận lúc gãy (≥1.15× SMA20)
- HTF EMA50 bias (4h)
- Pattern quality ≥ 0.48
- minScore = 3, spread EMA ≥ 0.35×ATR, ADX ≥ 22
- minRetrace 0.60, minHeight 1.75×ATR, breakBuf 0.22
- RSI filter mềm
- ATR stop cap 3.0
- TP 1.5R + move BE sau 1R + trail ATR
- oppCancel + cancelRetest + cooldown 8

## Cách chạy lại nghiên cứu

```bash
pip install -r requirements.txt
python -u src/run_research.py   # baseline / grid
python -u src/hillclimb.py      # vòng hillclimb
```

Kết quả ghi vào `results/`.
