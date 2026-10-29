# NN-price-prediction

AAPL next-day direction with a feed-forward neural network. **Report: [`report.pdf`](report.pdf)**

Software: Python 3.11.5 with Jupyter. Packages: numpy 2.3.4, pandas 2.3.3, scikit-learn 1.7.2 (`MLPClassifier`), matplotlib 3.9.2, yfinance 0.2.59.
The versions used are also saved in `results/software_versions.csv`.

## Files
| Path | Content |
|---|---|
| `nn_stock_direction.ipynb` | Full source code, explanations, figures and answers (the report) |
| `report.pdf` | Concise report with all required tables, figures and answers |
| `make_report.py` | Builds `report.pdf` from `results/` and `figures/` (`python3 make_report.py`) |
| `report.html` | The executed notebook exported as a static report |
| `data/AAPL.csv` | Cached adjusted daily close prices (Yahoo Finance) |
| `results/*.csv` | Exported tables: optimizer and architecture comparison, convergence curves, test metrics, confusion matrix, strategy performance |
| `figures/*.png` | Exported figures |

## How to reproduce
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace nn_stock_direction.ipynb   # or open it and use "Run All"
jupyter nbconvert --to html nn_stock_direction.ipynb --output report           # optional: rebuild report.html
```
Running everything takes about 20 seconds on a laptop. All randomness uses `SEED = 42`, and the price data is read from `data/AAPL.csv`.
Delete that file to download the data again (Yahoo may revise adjusted prices slightly). Every setting is in the first code cell.
