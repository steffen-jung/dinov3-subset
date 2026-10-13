import argparse
import statistics as st, math
from replay_search import (load_landscape, run_replay, write_records,
                           DEPTHS, BUDGETS, STATISTICS_FILENAME)
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="Generate the four-panel interaction and search figure.")
variants = parser.add_mutually_exclusive_group()
variants.add_argument("--no-baselines", action="store_true",
                      help="Save mean search curves without percentile bands or fixed baselines.")
variants.add_argument("--with-baselines", action="store_true",
                      help="Save mean search curves with truncation and additive-predictor baselines.")
variants.add_argument("--search-bands", action="store_true",
                      help="Generate the paper figure with percentile bands (the default).")
args = parser.parse_args()
args.search_bands = not (args.no_baselines or args.with_baselines)
if args.search_bands:
    args.no_baselines = True
HERE = Path(__file__).resolve().parent
OUT = HERE
B = load_landscape(HERE / "search.db")
D=dict(B)
pop=defaultdict(list)
for s,v in B: pop[sum(s)].append((s,v))
VAN=D[tuple([1]*12)]
OPT={k:max(v for _,v in pop[k]) for k in range(1,13)}
TR={k:D[tuple([1]*k+[0]*(12-k))] for k in range(1,13)}
sd={i:D[tuple([0 if j==i else 1 for j in range(12)])]-VAN for i in range(12)}

BLUE,ORANGE,GREY="#1f4e79","#e08214","0.55"
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "font.family":"serif","axes.labelsize":9,"xtick.labelsize":8,
                     "ytick.labelsize":8,"legend.fontsize":8})
fig,((a0,a1),(a2,a3))=plt.subplots(2,2,figsize=(6.9,4.6))

# The single-deletion curve supplies the scores used in panels (b), (c), and (d).
deleted_accuracy = [VAN + sd[i] for i in range(12)]
final_drop, first_drop = -sd[11], -sd[0]
a0.axhline(VAN, color="0.45", lw=1.1)
a0.text(1.15, VAN + 0.06, f"Full model: {VAN:.2f}%", color="0.35", fontsize=8)
a0.plot(range(1,13), deleted_accuracy, "-o", color=BLUE, ms=4, lw=1.5)
a0.scatter([12], [deleted_accuracy[-1]], s=35, color=ORANGE, zorder=4)
a0.annotate(f"Final block: {final_drop:.2f} pp drop\n{final_drop/first_drop:.1f} times the drop for block 1",
            xy=(12, deleted_accuracy[-1]), xytext=(2, 78.35), fontsize=8,
            color=ORANGE, va="center",
            arrowprops={"arrowstyle":"->", "color":ORANGE, "lw":1.1,
                        "connectionstyle":"arc3,rad=-0.15"})
a0.set_xlim(0.6,12.5); a0.set_ylim(77.55,81.35)
a0.set_xticks(range(1,13)); a0.set_yticks([78,79,80,81])
a0.set_xlabel("Deleted block"); a0.set_ylabel("Top-1 accuracy (%)")
a0.yaxis.grid(True,color="0.92",lw=0.6); a0.set_axisbelow(True)
a0.set_title("(a) Single-block deletions", fontsize=9, loc="left")

ex=[]
for i,j in combinations(range(12),2):
    s=tuple([0 if q in (i,j) else 1 for q in range(12)])
    ex.append((D[s]-VAN, sd[i]+sd[j]))
lo,hi=-6.0,-1.0
a1.plot([lo,hi],[lo,hi],ls="--",color="0.4",lw=1.0)
a1.fill_between([lo,hi],[lo,hi],lo,color=ORANGE,alpha=0.07)
a1.scatter([p for _,p in ex],[o for o,_ in ex],s=14,color=BLUE,alpha=0.75,edgecolors="none")
a1.set_xlim(lo,hi); a1.set_ylim(lo,hi)
a1.set_xlabel("predicted (sum of singles, pp)"); a1.set_ylabel("observed (pp)")
a1.grid(True,color="0.93",lw=0.6); a1.set_axisbelow(True)
a1.text(-1.3,-5.6,"super-additive\n66/66",fontsize=8,color=ORANGE,ha="right")
a1.set_title("(b) Two-block deletions",fontsize=9,loc="left")

def rank(x):
    o=sorted(range(len(x)),key=lambda i:x[i]); r=[0]*len(x)
    for p,i in enumerate(o): r[i]=p
    return r
sp=[]
for k in range(2,12):
    obs=[v for _,v in pop[k]]
    pred=[sum(sd[i] for i in range(12) if not s[i]) for s,_ in pop[k]]
    ro,rp=rank(obs),rank(pred); n=len(obs); mo,mp=st.mean(ro),st.mean(rp)
    num=sum((ro[i]-mo)*(rp[i]-mp) for i in range(n))
    den=math.sqrt(sum((ro[i]-mo)**2 for i in range(n))*sum((rp[i]-mp)**2 for i in range(n)))
    sp.append(num/den)
a2.axhline(0,color="0.35",lw=0.9)
a2.plot(range(2,12),sp,"-o",color=BLUE,ms=3,lw=1.4)
a2.set_xticks(range(2,12,2)); a2.set_xlabel("$k$ (blocks kept)")
a2.set_ylabel("Spearman $\\rho$"); a2.set_ylim(-0.55,1.05)
a2.yaxis.grid(True,color="0.92",lw=0.6); a2.set_axisbelow(True)
a2.set_title("(c) Additive scoring vs. measured",fontsize=9,loc="left")

# Every figure variant and the command-line tables use this same replay.
KS=list(DEPTHS); BUD=list(BUDGETS)
records=run_replay(HERE / "search.db", budgets=BUD)
write_records(records, OUT / STATISTICS_FILENAME)
for method,color in (("local search",BLUE),("random",GREY)):
    entries=records["methods"][method]
    means=[entry["mean"] for entry in entries]
    if args.search_bands:
        a3.fill_between(BUD,[entry["p10"] for entry in entries],
                        [entry["p90"] for entry in entries],
                        color=color,alpha=0.18,linewidth=0,zorder=1)
    a3.plot(BUD,means,marker=None,
            color=color,ms=3,lw=1.4,label=method,zorder=3)
if args.search_bands:
    a3.text(0.03,0.90,"Shading: 10th-90th\npercentiles",transform=a3.transAxes,
            fontsize=7,color="0.35",va="top")
if not args.no_baselines:
    # Fixed selection rule from Section 4.5: evaluate the subset with the
    # largest additive prediction, not the largest measured accuracy.
    # Python's max preserves database order if predicted scores tie.
    additive_selected = {
        k: max(pop[k], key=lambda sv: VAN + sum(sd[i] for i in range(12) if not sv[0][i]))
        for k in KS
    }
    additive_gap = st.mean(additive_selected[k][1] - OPT[k] for k in KS)
    predictor_color = "#397b63"
    a3.axhline(additive_gap,ls="--",color=predictor_color,lw=1.3)
    a3.text(11,additive_gap+0.4,"additive predictor",fontsize=8,color=predictor_color)
    print(f"Additive predictor: mean gap {additive_gap:.6f} pp")
    t=st.mean([TR[k]-OPT[k] for k in KS])
    a3.axhline(t,ls="--",color=ORANGE,lw=1.3)
    a3.text(11,t+0.25,"truncation",fontsize=8,color=ORANGE)
a3.axhline(0,color="0.3",lw=0.9)
ticks=[10,20,50,100,200]
a3.set_xscale("log"); a3.set_xticks(ticks); a3.set_xticklabels(ticks)
a3.set_xlabel("evaluation budget per $k$"); a3.set_ylabel("Selected - best observed (pp)")
a3.set_xlim(8.5,240)
a3.yaxis.grid(True,color="0.92",lw=0.6); a3.set_axisbelow(True)
if args.no_baselines:
    a3.legend(loc="lower right",frameon=False)
else:
    a3.legend(loc="upper right",bbox_to_anchor=(1,0.76),frameon=False)
a3.set_title("(d) Search methods",fontsize=9,loc="left")

if args.search_bands:
    filename = "fig4_interaction_and_search.pdf"
elif args.with_baselines:
    filename = "fig4_interaction_and_search_with_baselines.pdf"
else:
    filename = "fig4_interaction_and_search_no_baselines.pdf"
fig.tight_layout(h_pad=1.7, w_pad=1.8); fig.savefig(OUT/filename,bbox_inches="tight")
if args.search_bands:
    fig.savefig(OUT/Path(filename).with_suffix(".png"),dpi=220,bbox_inches="tight")
print("ok")
