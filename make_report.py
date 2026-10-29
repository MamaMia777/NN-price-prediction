"""Builds report.pdf from results/*.csv and figures/*.png. Run from the project folder."""
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=13.5, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=10, spaceAfter=4, keepWithNext=1,
                    textColor=colors.HexColor("#1f3b57"))
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=16, leading=20, alignment=TA_CENTER)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=9, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=10.5, spaceAfter=0)
W = A4[0] - 4 * cm

story = []
p = lambda text, style=BODY: story.append(Paragraph(text, style))


def table(rows, header=True, widths=None, highlight=None):
    rows = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
             ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                  ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6"))]
    if highlight is not None:
        style.append(("BACKGROUND", (0, highlight), (-1, highlight), colors.HexColor("#fff4d6")))
    t.setStyle(TableStyle(style))
    story.append(KeepTogether([t, Spacer(1, 6)]))


def figure(path, caption, width=W):
    img = Image(path)
    img.drawWidth, img.drawHeight = width, width * img.imageHeight / img.imageWidth
    story.append(KeepTogether([img, Paragraph(caption, CAP), Spacer(1, 6)]))


def cv_rows(csv, selected):
    df = pd.read_csv(csv, index_col=0)
    rows = [["Model", "Mean train bal. acc.", "Mean val. bal. acc. (sd)", "Time (s)", "Mean iterations"]]
    for name, r in df.iterrows():
        label = f"<b>{name}</b> (selected)" if name.split()[0] == selected else name
        rows.append([label, f"{r.train_bal_acc:.3f}", f"{r.val_bal_acc:.3f} ({r.val_bal_acc_std:.3f})",
                     f"{r.time_s:.2f}", f"{r.mean_n_iter:.0f}"])
    return rows, list(df.index.str.split().str[0]).index(selected) + 1


# ---------------------------------------------------------------- title
p("Predicting the Direction of AAPL Daily Returns with a Feed-Forward Neural Network", TITLE)
p("Python 3.11.5 / scikit-learn 1.7.2 &nbsp;|&nbsp; Data: Yahoo Finance, adjusted daily close, 2015-2024", SUB)

p("<b>Summary.</b> A two-hidden-layer neural network using ten lagged returns does not predict the next-day direction of "
  "AAPL. Validation balanced accuracy is about 0.50 for every optimizer and architecture, and the final test balanced "
  "accuracy is 0.485, below the no-skill level of 0.5. The network trading strategy underperforms buy and hold both before "
  "and after transaction costs.")

# ---------------------------------------------------------------- 1
p("1. Data and experimental design", H1)
p("<b>Stock.</b> Apple Inc. (AAPL): one of the most liquid stocks (tight spreads, realistic low trading costs) with a long, "
  "clean price history. Prices are split- and dividend-adjusted.")
p("<b>Definitions.</b> Daily return r<sub>t</sub> = P<sub>t</sub> / P<sub>t-1</sub> - 1. "
  "Inputs x<sub>t</sub> = (r<sub>t-1</sub>, ..., r<sub>t-10</sub>). "
  "Target y<sub>t</sub> = 1 if r<sub>t</sub> &gt; 0 (Up), else 0 (Down). All inputs are known at the previous close.")
table([["", "Period", "Observations", "Up / Down"],
       ["Full sample", "2015-01-20 to 2024-12-31", "2,505", "53.3% / 46.7%"],
       ["Model-fitting period (80%)", "2015-01-20 to 2023-01-03", "2,004", "52.5% / 47.5%"],
       ["Final test period (20%)", "2023-01-04 to 2024-12-31", "501", "56.3% / 43.7%"]],
      widths=[4.6 * cm, 4.8 * cm, 2.8 * cm, 2.8 * cm])
p("Observations stay in time order and are never shuffled. Model choices use 5-fold expanding-window time-series "
  "cross-validation inside the fitting period. Each fold trains on all earlier data and validates on the next block of "
  "about 334 days. Inputs are standardised inside each training fold. The test period is used only once. "
  "<b>Metric:</b> balanced accuracy = (hit rate on Up days + hit rate on Down days) / 2. Any constant forecast scores 0.5, "
  "so 0.5 is the no-skill benchmark.")

# ---------------------------------------------------------------- 2
p("2. Neural-network model", H1)
p("scikit-learn <i>MLPClassifier</i>: inputs (10) -&gt; hidden layers with ReLU activation -&gt; one sigmoid output "
  "p<sub>t</sub> = P(y<sub>t</sub> = 1 | x<sub>t</sub>); prediction Up if p<sub>t</sub> &gt; 0.5. The objective is "
  "binary cross-entropy (log loss) with a small L2 penalty (alpha = 10<super>-4</super>). The baseline has 32 and 16 "
  "hidden neurons. All optimizers are available in scikit-learn: <b>SGD</b> uses mini-batches of 200 with momentum 0.9 and "
  "learning rate 0.01; <b>Adam</b> uses learning rate 0.001; <b>L-BFGS</b> is a full-batch quasi-Newton method. "
  "Training stops after 300 iterations, or earlier if the loss improves by less than 10<super>-4</super> for 10 "
  "consecutive iterations. For SGD and Adam an iteration is one epoch (a full pass over the data); for L-BFGS it is "
  "one quasi-Newton step. Random seed: 42.")

# ---------------------------------------------------------------- 3
p("3. Optimizer comparison (baseline 32-16 network)", H1)
rows, hl = cv_rows("results/optimizer_comparison.csv", "adam")
table(rows, widths=[3.6 * cm, 3.2 * cm, 3.8 * cm, 2 * cm, 2.6 * cm], highlight=hl)
p("All three optimizers fit the training data well above chance, and L-BFGS almost memorises it (0.99). Yet no optimizer "
  "beats 0.5 on validation data by more than about one fold standard deviation. L-BFGS is the fastest on this small data "
  "set but overfits the most. SGD and Adam usually stop at the 300-epoch cap. <b>Adam</b> has the highest mean "
  "validation score and is selected, but the gap between optimizers is within noise.")

# ---------------------------------------------------------------- 4
p("4. Architecture comparison (Adam)", H1)
rows, hl = cv_rows("results/architecture_comparison.csv", "16-8")
table(rows, widths=[3.6 * cm, 3.2 * cm, 3.8 * cm, 2 * cm, 2.6 * cm], highlight=hl)
p("Larger networks fit the training data more closely (0.66 for 16-8 up to 0.95 for 64-32-16), but validation accuracy "
  "does not improve and is lowest for the deepest network. The smallest network, <b>16-8</b>, has the highest validation "
  "score (0.506) and is selected. All differences are within about one standard deviation of no skill.")

# ---------------------------------------------------------------- 5
p("5. Convergence analysis (SGD and Adam)", H1)
figure("figures/03_convergence.png",
       "Figure 1. Baseline 32-16 network trained epoch by epoch on fold 5 (train 2015-2021, validate 2021-2023).")
p("The training loss falls steadily for both methods; Adam falls faster (0.78 to 0.47, compared with 0.78 to 0.54 for SGD). "
  "Validation balanced accuracy fluctuates around 0.50 throughout. Its peaks of about 0.53 (SGD at epoch 45, Adam at epoch "
  "203) do not persist, and both end near 0.49. This is overfitting: lower training loss reflects noise memorised in past "
  "returns, not a repeatable signal.")

# ---------------------------------------------------------------- 6
p("6. Final evaluation on the untouched test period (2023-2024)", H1)
m = pd.read_csv("results/test_metrics.csv", index_col=0)
cmx = pd.read_csv("results/confusion_matrix.csv", index_col=0).values
table([["", "NN 16-8, Adam", "Always Up"],
       ["Accuracy", f"{m.iloc[0, 0]:.3f}", f"{m.iloc[0, 1]:.3f}"],
       ["Balanced accuracy", f"{m.iloc[1, 0]:.3f}", f"{m.iloc[1, 1]:.3f}"],
       ["Share of days predicted Up", f"{m.iloc[2, 0]:.1%}", f"{m.iloc[2, 1]:.0%}"]],
      widths=[5 * cm, 3.5 * cm, 3.5 * cm])
table([["Confusion matrix", "Predicted Down", "Predicted Up", "Hit rate"],
       ["Actual Down", cmx[0, 0], cmx[0, 1], f"{cmx[0, 0] / cmx[0].sum():.1%}"],
       ["Actual Up", cmx[1, 0], cmx[1, 1], f"{cmx[1, 1] / cmx[1].sum():.1%}"]],
      widths=[5 * cm, 3.5 * cm, 3.5 * cm, 2.5 * cm])
p("The network predicts Up on 81% of days. It catches 80% of Up days but only 17% of Down days, so its balanced "
  "accuracy (0.485) is <i>below</i> 0.5, and its accuracy (0.525) is lower than the naive Always Up rule (0.563). "
  "The standard error of balanced accuracy on 501 days is about 0.022, so the result is statistically indistinguishable "
  "from no skill. <b>The network gives no meaningful out-of-sample improvement.</b> This is consistent with weak-form "
  "market efficiency: past returns alone carry no usable information about the next day's direction.")

# ---------------------------------------------------------------- 7
p("7. Investment strategy", H1)
p("<b>Buy and hold:</b> R<sub>t</sub> = r<sub>t</sub>. <b>NN strategy:</b> s<sub>t</sub> = 1 if p<sub>t</sub> &gt; 0.5, "
  "else 0 (cash); R<sub>t</sub> = s<sub>t</sub> r<sub>t</sub> - c |s<sub>t</sub> - s<sub>t-1</sub>|, with c = 0 (before "
  "costs) or c = 10 bp per buy or sell (after costs). Wealth W<sub>T</sub> = product of (1 + R<sub>t</sub>).")
s = pd.read_csv("results/strategy_performance.csv", index_col=0)
rows = [["Test period 2023-2024", "Final wealth", "Ann. return", "Ann. vol.", "Sharpe", "Max DD", "In market", "Trades"]]
for name, r in s.iterrows():
    rows.append([name, f"{r.iloc[0]:.2f}", f"{r.iloc[1]:.1%}", f"{r.iloc[2]:.1%}", f"{r.iloc[3]:.2f}",
                 f"{r.iloc[4]:.1%}", f"{r.iloc[5]:.0%}", f"{r.iloc[6]:.0f}"])
table(rows, widths=[3.6 * cm, 1.9 * cm, 1.8 * cm, 1.6 * cm, 1.4 * cm, 1.6 * cm, 1.7 * cm, 1.6 * cm])
figure("figures/05_strategy_growth.png", "Figure 2. Growth of one monetary unit over the test period.")
p("Buy and hold turns 1 into 2.02 (Sharpe 1.77). The NN strategy holds the stock on 81% of days. Before costs it reaches "
  "1.70 (Sharpe 1.45). Its slightly lower volatility and drawdown come from holding the stock less often, not from "
  "successful timing. It trades 147 times, and 10 bp per trade reduces wealth to 1.47 (Sharpe 1.08) and makes its drawdown "
  "worse than buy and hold. <b>Buy and hold dominates the NN strategy both before and after transaction costs.</b>")

# ---------------------------------------------------------------- 8
p("8. Conclusion and limitations", H1)
p("A feed-forward network on ten lagged returns learns the training data but not a signal that holds out of sample. "
  "The conclusion does not depend on the choice of optimizer or architecture. These results come from a single stock, "
  "a single random seed and a single test window, and the exact numbers vary slightly between runs. The no-skill "
  "conclusion is stable.")

# ---------------------------------------------------------------- reproducibility
p("Reproducibility", H1)
v = pd.read_csv("results/software_versions.csv", index_col=0)["version"]
p("<b>Software:</b> " + ", ".join(f"{k} {x}" for k, x in v.items()) + ", Jupyter.")
p("<b>Files:</b> <i>nn_stock_direction.ipynb</i> (all source code); <i>data/AAPL.csv</i> (cached prices); "
  "<i>results/*.csv</i> (all tables); <i>figures/*.png</i> (all figures); <i>requirements.txt</i>; <i>README.md</i>.")
p("<b>To reproduce:</b> <font face='Courier'>pip install -r requirements.txt</font>, then "
  "<font face='Courier'>jupyter nbconvert --to notebook --execute --inplace nn_stock_direction.ipynb</font> "
  "(or Run All in Jupyter). The run takes about 20 seconds. All settings are in the first code cell.")


def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, str(doc.page))


SimpleDocTemplate("report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                  bottomMargin=1.8 * cm, title="AAPL direction prediction with a neural network").build(
    story, onFirstPage=footer, onLaterPages=footer)
