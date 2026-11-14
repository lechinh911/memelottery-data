# 2 Day / 2 Dinh -> Gay -> EMA Momentum

Nghien cuu / toi uu chi bao TradingView (Pine v6): cum nen 2 day/2 dinh, gay neckline, xac nhan EMA20/50.

## Ket luan quan trong

**Winrate cao != lai nhieu.** Preset Quality (~67% WR) qua khat -> it lenh, doi lau, full-sample lai thap.  
**Default moi: Active** — nhieu lenh hon, van expectancy duong.

| Preset | ~Lenh/thang/market (1h) | OOS WR | Full-sample net | Khi nao dung |
|--------|-------------------------|--------|-----------------|--------------|
| **Active (default)** | ~3.3 | ~55% | ~+7% | Can bang tan suat / edge |
| Core | ~2.2 | ~60% | **~+19%** | Toi da lai, chap nhan it lenh hon Active |
| Freq | ~3.7 | ~55% | ~+5% | Muon nhieu tin hieu nhat con edge |
| Quality | ~1.9 | ~67% | ~+3% | Chi trade setup dep, khong ngai doi |

Neu noi long hon Freq (pivot 3, filter yeu): so lenh tang nhung **mat edge** (nhat EURUSD).

## Dung tren TradingView

1. Copy file raw: [`COPY_PASTE_TV.pine`](https://github.com/lechinh911/memelottery-data/blob/cursor/double-pattern-momentum-rd-259e/COPY_PASTE_TV.pine) (bam **Raw** -> Ctrl+A -> Ctrl+C)
2. Pine Editor -> paste -> Save -> Add to chart
3. Chon preset **Active (khuyen nghi)** (hoac Core neu uu tien lai)

Khuyen nghi: BTC/ETH/Gold 1h. EURUSD thuong keo net xuong — co the tat SHORT/LONG tuy thi truong.

## Chay lai backtest

```bash
pip install -r requirements.txt
python -u src/run_research.py
python -u src/hillclimb.py
```

Chi tiet: [`research/FINDINGS.md`](research/FINDINGS.md)
