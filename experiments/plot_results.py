import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter, NullFormatter

R = "results"
FIG = f"{R}/figures"
os.makedirs(FIG, exist_ok=True)

# ---------- 폰트 설정 ----------
available = {f.name for f in fm.fontManager.ttflist}
for _name in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"]:
    if _name in available:
        plt.rcParams["font.family"] = _name
        print("사용 폰트:", _name)
        break
else:
    print("한글 폰트를 찾지 못함 (한글이 깨질 수 있음)")

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["figure.dpi"] = 100


# ---------- 유틸 ----------
def _fmt(v, _pos=None):
    """로그축 눈금을 일반 숫자로 표기 (지수 표기/유니코드 마이너스 회피)"""
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:g}"
    return f"{v:.10f}".rstrip("0").rstrip(".")


def plain_log(ax, axis="y"):
    """로그축 눈금 숫자를 일반 표기로 변경"""
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(FuncFormatter(_fmt))
    target.set_minor_formatter(NullFormatter())


def save(name):
    plt.tight_layout()
    plt.savefig(f"{FIG}/{name}", dpi=200)
    plt.close()
    print("saved", name)


def pos(series, floor=1e-5):
    """로그축에서 0이 사라지지 않도록 하한 적용"""
    return np.maximum(np.asarray(series, dtype=float), floor)


def human(b):
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}TB"


# ---------- 데이터 로드 ----------
bloom_main = pd.read_csv(f"{R}/bloom_main.csv").iloc[0]
cms_main = pd.read_csv(f"{R}/cms_main.csv")
scatter = pd.read_csv(f"{R}/cms_stock_scatter.csv", dtype={"key": str})
bits = pd.read_csv(f"{R}/bloom_params_bits.csv")
kdf = pd.read_csv(f"{R}/bloom_params_k.csv")
w_stock = pd.read_csv(f"{R}/cms_params_width.csv")
d_stock = pd.read_csv(f"{R}/cms_params_depth.csv")
eqmem = pd.read_csv(f"{R}/cms_params_equalmem.csv")
w_pair = pd.read_csv(f"{R}/cms_params_pair_width.csv")


# ---------- Fig1: Bloom 비트/원소 vs FPR ----------
def fig01():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bits["bits_per_elem"], pos(bits["fpr_measured"]), "o-", label="실측 FPR")
    ax.plot(bits["bits_per_elem"], pos(bits["fpr_theory"]), "s--", label="이론 FPR")
    ax.set_yscale("log")
    plain_log(ax, "y")
    ax.set_xlabel("비트/원소 (m/n), k는 각각 최적값")
    ax.set_ylabel("False Positive Rate (log)")
    ax.set_title("Bloom Filter: 비트/원소 vs FPR")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    save("fig01_bloom_bits_fpr.png")


# ---------- Fig2: Bloom k vs FPR (U자형) ----------
def fig02():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for b, g in kdf.groupby("bits_per_elem"):
        line, = ax.plot(g["k"], pos(g["fpr_measured"]), "o-", label=f"실측 (비트/원소={b})")
        ax.plot(g["k"], pos(g["fpr_theory"]), "--", color=line.get_color(), alpha=0.6,
                label=f"이론 (비트/원소={b})")
        best = g.loc[g["fpr_measured"].idxmin()]
        ax.annotate(f"최저 k={int(best['k'])}", (best["k"], best["fpr_measured"]),
                    textcoords="offset points", xytext=(8, -14), fontsize=8)
    ax.set_yscale("log")
    plain_log(ax, "y")
    ax.set_xlabel("해시 함수 개수 k")
    ax.set_ylabel("False Positive Rate (log)")
    ax.set_title("Bloom Filter: k vs FPR (U자형)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    save("fig02_bloom_k_fpr.png")


# ---------- Fig3: Bloom k vs 삽입 시간 ----------
def fig03():
    fig, ax = plt.subplots(figsize=(6, 4))
    g = kdf[kdf["bits_per_elem"] == 10]
    ax.plot(g["k"], g["insert_time_s"], "o-")
    ax.set_xlabel("해시 함수 개수 k")
    ax.set_ylabel("삽입 시간 (초)")
    ax.set_title("Bloom Filter: k vs 삽입 시간 (비트/원소=10)")
    ax.grid(True, alpha=0.3)
    save("fig03_bloom_k_time.png")


# ---------- Fig4: 메모리 비교 ----------
def fig04():
    s = cms_main[cms_main["target"] == "stock"].iloc[0]
    p = cms_main[cms_main["target"] == "pair"].iloc[0]
    labels = ["Bloom vs set\n(쌍 22만개)", "CMS vs Counter\n(상품 3,665개)",
              "CMS vs Counter\n(쌍 26.7만개)"]
    approx = [bloom_main["bloom_bytes"], s["cms_bytes"], p["cms_bytes"]]
    exact = [bloom_main["set_bytes"], s["counter_bytes"], p["counter_bytes"]]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    b1 = ax.bar(x - 0.2, approx, 0.4, label="근사 자료구조")
    b2 = ax.bar(x + 0.2, exact, 0.4, label="정확한 방법 (set/Counter)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), human(r.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    plain_log(ax, "y")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("메모리 (bytes, log)")
    ax.set_title("메모리 사용량 비교")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    save("fig04_memory_compare.png")


# ---------- Fig5: CMS width vs 오차 (stock) ----------
def fig05():
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(w_stock["width"], pos(w_stock["mean_rel_err"]), "o-", label="전체 평균")
    ax.plot(w_stock["width"], pos(w_stock["rel_err_head10pct"]), "s-", label="상위 10% (heavy)")
    ax.plot(w_stock["width"], pos(w_stock["rel_err_tail50pct"]), "^-", label="하위 50% (tail)")
    ax.axvline(3665, color="gray", linestyle=":", label="고유 상품 수 3,665")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log(ax, "x")
    plain_log(ax, "y")
    ax.set_xlabel("width (log)")
    ax.set_ylabel("평균 상대오차 (log)")
    ax.set_title("CMS: width vs 상대오차 (stock, depth=5)\n※ 오차 0은 그래프 하한 0.00001로 표시")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    save("fig05_cms_width_stock.png")


# ---------- Fig6: CMS depth vs 오차, 처리량 ----------
def fig06():
    fig, ax1 = plt.subplots(figsize=(6.8, 4.4))
    ax1.plot(d_stock["depth"], pos(d_stock["mean_rel_err"]), "o-", color="tab:blue")
    ax1.set_yscale("log")
    plain_log(ax1, "y")
    ax1.set_xlabel("depth")
    ax1.set_ylabel("평균 상대오차 (log)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(d_stock["depth"], d_stock["add_ops_per_s"], "s--", color="tab:red")
    ax2.set_yscale("log")
    plain_log(ax2, "y")
    ax2.set_ylabel("add 처리량 (ops/s, log)", color="tab:red")
    ax1.set_title("CMS: depth vs 오차 / 처리량 (width=2000)\n※ depth 증가 시 메모리도 함께 증가")
    ax1.grid(True, which="both", alpha=0.3)
    save("fig06_cms_depth.png")


# ---------- Fig7: 동일 메모리 비교 ----------
def fig07():
    labels = [f"w={w}\nd={d}" for w, d in zip(eqmem["width"], eqmem["depth"])]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(labels, eqmem["mean_rel_err"], color="tab:blue")
    for r, v in zip(bars, eqmem["mean_rel_err"]):
        ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("평균 상대오차")
    ax.set_title(f"CMS: 동일 메모리({human(eqmem['cms_bytes'].iloc[0])}) 조합 비교 (stock)")
    ax.grid(True, axis="y", alpha=0.3)
    save("fig07_cms_equal_memory.png")


# ---------- Fig8: pair width vs 오차 + 메모리 ----------
def fig08():
    counter_bytes = cms_main[cms_main["target"] == "pair"].iloc[0]["counter_bytes"]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(w_pair["width"], pos(w_pair["mean_rel_err"]), "o-", color="tab:blue", label="평균 상대오차")
    ax1.plot(w_pair["width"], pos(w_pair["rel_err_head10pct"]), "s-", color="tab:green",
             label="상위 10% 오차")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    plain_log(ax1, "x")
    plain_log(ax1, "y")
    ax1.set_xlabel("width (log, depth=5)")
    ax1.set_ylabel("상대오차 (log)")
    ax2 = ax1.twinx()
    ax2.plot(w_pair["width"], w_pair["cms_bytes"], "^--", color="tab:red", label="CMS 메모리")
    ax2.axhline(counter_bytes, color="black", linestyle=":",
                label=f"Counter 메모리 {human(counter_bytes)}")
    ax2.set_yscale("log")
    plain_log(ax2, "y")
    ax2.set_ylabel("메모리 (bytes, log)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax1.set_title("CMS: pair 키 width vs 오차 / 메모리\n(정확도 상승 시 메모리 이점 감소)")
    ax1.grid(True, which="both", alpha=0.3)
    save("fig08_cms_pair_width.png")


# ---------- Fig9: 실제 vs 추정 산점도 ----------
def fig09():
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(scatter["true"], scatter["est"], s=6, alpha=0.4, label="상품")
    lim = [1, max(scatter["true"].max(), scatter["est"].max()) * 1.2]
    ax.plot(lim, lim, "r--", label="y = x (정확)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log(ax, "x")
    plain_log(ax, "y")
    ax.set_xlabel("실제 빈도 (log)")
    ax.set_ylabel("CMS 추정 빈도 (log)")
    ax.set_title("CMS: 실제 vs 추정 (stock, w=2000, d=5)\n모든 점이 y=x 위 또는 선상 = 과대추정만 발생")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    save("fig09_cms_scatter.png")


# ---------- Fig10: 빈도 구간별 상대오차 ----------
def fig10():
    cols = ["rel_err_head10pct", "rel_err_mid40pct", "rel_err_tail50pct"]
    names = ["상위 10%", "중간 40%", "하위 50%"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i, (_, row) in enumerate(cms_main.iterrows()):
        vals = [row[c] for c in cols]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38,
                      label=f"{row['target']} (w={int(row['width'])}, d={int(row['depth'])})")
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("평균 상대오차")
    ax.set_title("CMS: 빈도 구간별 상대오차")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    save("fig10_cms_tiers.png")


# ---------- Fig11: 처리량 비교 ----------
def fig11():
    s = cms_main[cms_main["target"] == "stock"].iloc[0]
    p = cms_main[cms_main["target"] == "pair"].iloc[0]
    labels = ["Bloom vs set\n(삽입)", "CMS vs Counter\n(stock)", "CMS vs Counter\n(pair)"]
    approx = [bloom_main["bloom_throughput_rec_s"], s["cms_throughput_rec_s"], p["cms_throughput_rec_s"]]
    exact = [bloom_main["set_throughput_rec_s"], s["counter_throughput_rec_s"], p["counter_throughput_rec_s"]]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    b1 = ax.bar(x - 0.2, approx, 0.4, label="근사 자료구조")
    b2 = ax.bar(x + 0.2, exact, 0.4, label="정확한 방법")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(),
                    f"{r.get_height() / 1000:.0f}K", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("처리량 (records/sec)")
    ax.set_title("처리량 비교 (CSV 읽기 포함, 3회 평균)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save("fig11_throughput.png")


if __name__ == "__main__":
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11):
        f()
    print("ALL FIGURES DONE")