import sqlite3, statistics as st
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE
# The release database contains only completed, nonempty binary schedules.
# Retain its insertion order: seeded sampling and tie handling depend on it.
with sqlite3.connect((HERE / "search.db").as_uri() + "?mode=ro", uri=True) as c:
    B = [(tuple(map(int, s.split(","))), v) for s, v in c.execute(
        "SELECT schedule_key, val_top1 FROM search_runs ORDER BY rowid"
    )]
D=dict(B)
pop=defaultdict(list)
for s,v in B: pop[sum(s)].append((s,v))
VAN=D[tuple([1]*12)]
OPT={k:max(pop[k],key=lambda x:x[1]) for k in range(1,13)}
TR={k:D[tuple([1]*k+[0]*(12-k))] for k in range(1,13)}
KS=list(range(1,13))
PAR={k:(2458216+1774848*(k-1))/1e6 for k in KS}

BLUE,ORANGE,GREY="#1f4e79","#e08214","0.55"
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,
                     "font.family":"serif","axes.labelsize":8,"xtick.labelsize":7.5,
                     "ytick.labelsize":7.5,"legend.fontsize":7.5})

# ---- Accuracy landscape and truncation ranking
fig,(a1,a2)=plt.subplots(1,2,figsize=(6.9,2.15))
for k in KS:
    vals=[v for _,v in pop[k]]
    a1.scatter([k]*len(vals),vals,s=1.2,color=GREY,alpha=0.25,edgecolors="none",zorder=1)
a1.plot(KS,[st.mean([v for _,v in pop[k]]) for k in KS],"-",color=GREY,lw=1.3,zorder=3,label="mean subset")
a1.plot(KS,[OPT[k][1] for k in KS],"-o",color=BLUE,ms=3,lw=1.5,zorder=4,label="best observed subset")
a1.plot(KS,[TR[k] for k in KS],"-s",color=ORANGE,ms=3,lw=1.5,zorder=4,label="truncation $[1{:}k]$")
a1.set_xticks(KS); a1.set_xlabel("$k$ (number of blocks kept)"); a1.set_ylabel("val top-1 (%)")
a1.set_ylim(0,90); a1.yaxis.grid(True,color="0.92",lw=0.6); a1.set_axisbelow(True)
a1.legend(loc="lower right",frameon=False)
a1.set_title("(a) All 4095 subsets",fontsize=8,loc="left")

pcts=[100*sum(1 for _,v in pop[k] if v<TR[k])/len(pop[k]) for k in KS]
a2.bar(KS[:-1],pcts[:-1],color=ORANGE,width=0.65)
a2.text(12, 2, "N/A", ha="center", va="bottom", fontsize=7, color="0.4")
a2.axhline(50,color="0.35",lw=0.9,ls="--")
a2.text(6.5,53,"population median",fontsize=7,color="0.35",ha="center")
a2.set_xticks(KS); a2.set_xlabel("$k$ (number of blocks kept)")
a2.set_ylabel("Subsets below truncation (%)")
a2.set_ylim(0,72); a2.yaxis.grid(True,color="0.92",lw=0.6); a2.set_axisbelow(True)
a2.set_title("(b) Subsets outperformed by truncation",fontsize=8,loc="left")
fig.tight_layout(); fig.savefig(OUT/"fig1_frontier.pdf",bbox_inches="tight")
plt.close(fig)

# Shared canvas and axes keep rows aligned at equal widths in the paper.
GRID_FIGSIZE=(3.35,2.6)
GRID_AXES=[0.145,0.21,0.65,0.64]

# ---- Retained-block comparison grid
fig=plt.figure(figsize=GRID_FIGSIZE)
ax=fig.add_axes(GRID_AXES)
def draw_kept_grid(ax):
    ks=list(range(2,12))
    CB,CO,CT,CN=BLUE,"#8db8d8",ORANGE,"#eeeeee"
    for r,k in enumerate(ks):
        ko={i+1 for i,x in enumerate(OPT[k][0]) if x}; kt=set(range(1,k+1))
        for L in range(1,13):
            o,t=L in ko,L in kt
            col=CB if o and t else CO if o else CT if t else CN
            ax.add_patch(Rectangle((L-0.45,r-0.43),0.9,0.86,facecolor=col,edgecolor="white",lw=0.7))
        ax.text(13.1,r,f"+{OPT[k][1]-TR[k]:.1f}",va="center",ha="left",fontsize=7,color=BLUE)
    ax.text(13.5,-0.065,"Gain over\ntruncation (pp)",transform=ax.get_xaxis_transform(),
            va="top",ha="center",fontsize=6.5,color="0.4")
    ax.set_xlim(0.5,12.5); ax.set_ylim(len(ks)-0.5,-0.5)
    ax.set_xticks(range(1,13)); ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"$k{{=}}{k}$" for k in ks])
    ax.set_xlabel("block index"); ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(handles=[Patch(color=CB,label="both"),Patch(color=CO,label="best only"),
                       Patch(color=CT,label="truncation only"),Patch(color=CN,label="neither")],
              loc="upper center",bbox_to_anchor=(0.32,-0.03),ncol=2,frameon=False,
              fontsize=7,handlelength=1.1,columnspacing=1.0)

draw_kept_grid(ax)
fig.savefig(OUT/"fig2_kept_dropped_grid.pdf",bbox_inches=None); plt.close(fig)

# ---- Marginal-value heatmap
mv=np.zeros((12,10))
for ci,k in enumerate(range(2,12)):
    for i in range(12):
        w=[v for s,v in pop[k] if s[i]]; wo=[v for s,v in pop[k] if not s[i]]
        mv[i,ci]=st.mean(w)-st.mean(wo)
fig=plt.figure(figsize=GRID_FIGSIZE)
ax=fig.add_axes(GRID_AXES)
def draw_marginal_grid(fig,ax,cax):
    m=np.abs(mv).max()
    im=ax.imshow(mv.T,cmap="RdBu_r",vmin=-m,vmax=m,aspect="auto")
    # White cell separators match the retained-block comparison grid.
    ax.set_xticks(np.arange(-0.5,12,1), minor=True)
    ax.set_yticks(np.arange(-0.5,10,1), minor=True)
    ax.set_axisbelow(False)
    ax.grid(which="minor",color="white",linewidth=1.5)
    ax.tick_params(which="minor",bottom=False,top=False,left=False,right=False)
    
    ax.set_xticks(range(12)); ax.set_xticklabels(range(1,13))
    ax.set_yticks(range(10)); ax.set_yticklabels([f"$k{{=}}{k}$" for k in range(2,12)])
    ax.set_xlabel("block index")
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    for i in range(12):
        for j in range(10):
            if abs(mv[i,j])>2.0:
                ax.text(i,j,f"{mv[i,j]:.0f}",ha="center",va="center",fontsize=5.5,
                        color="white")
    cb=fig.colorbar(im,cax=cax)
    cb.set_label("marginal value (pp)",fontsize=7); cb.ax.tick_params(labelsize=6.5)
    for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)
    ax.tick_params(which="both",length=0)

cax=fig.add_axes([0.84,GRID_AXES[1],0.022,GRID_AXES[3]])
draw_marginal_grid(fig,ax,cax)
fig.savefig(OUT/"fig3_marginal_value.pdf",bbox_inches=None); plt.close(fig)

# ---- Combined paper figure: identical row geometry in both panels.
combined=plt.figure(figsize=(6.9,2.85))
left=combined.add_axes([0.070,0.19,0.315,0.60])
right=combined.add_axes([0.565,0.19,0.315,0.60])
colorbar=combined.add_axes([0.915,0.19,0.0107,0.60])
draw_kept_grid(left)
draw_marginal_grid(combined,right,colorbar)
combined.text(0.070,0.985,"(a) Best observed vs. truncation",fontsize=9,va="top")
combined.text(0.565,0.985,"(b) Block performance across budgets",fontsize=9,va="top")
# The x coordinates differ by one half-cell in the two data conventions.
for row in range(10):
    assert np.isclose(left.transData.transform((1,row))[1],
                      right.transData.transform((0,row))[1])
combined.savefig(OUT/"fig2_structure_and_marginal_value.pdf",bbox_inches=None)
combined.savefig(OUT/"fig2_structure_and_marginal_value.png",dpi=220,bbox_inches=None)
plt.close(combined)

# ---- Two-panel diagnostic: super-additivity and additive scoring
sd={i:D[tuple([0 if j==i else 1 for j in range(12)])]-VAN for i in range(12)}
ex=[]
for i,j in combinations(range(12),2):
    s=tuple([0 if q in (i,j) else 1 for q in range(12)])
    ex.append((D[s]-VAN, sd[i]+sd[j]))
fig,(a1,a2)=plt.subplots(1,2,figsize=(6.9,2.1))
lo,hi=-6.0,-1.0
a1.plot([lo,hi],[lo,hi],ls="--",color="0.4",lw=1.0)
a1.fill_between([lo,hi],[lo,hi],lo,color=ORANGE,alpha=0.07)
a1.scatter([p for _,p in ex],[o for o,_ in ex],s=8,color=BLUE,alpha=0.75,edgecolors="none")
a1.set_xlim(lo,hi); a1.set_ylim(lo,hi); a1.set_aspect("equal")
a1.set_xlabel("predicted (sum of single drops, pp)"); a1.set_ylabel("observed (pp)")
a1.grid(True,color="0.93",lw=0.6); a1.set_axisbelow(True)
a1.text(-1.35,-5.5,"super-additive\n66/66 pairs",fontsize=7,color=ORANGE,ha="right")
a1.set_title("(a) two-block deletions",fontsize=8,loc="left")

def rank(x):
    o=sorted(range(len(x)),key=lambda i:x[i]); r=[0]*len(x)
    for p,i in enumerate(o): r[i]=p
    return r
import math
sp=[]
for k in range(2,12):
    obs=[v for _,v in pop[k]]
    pred=[sum(sd[i] for i in range(12) if not s[i]) for s,_ in pop[k]]
    ro,rp=rank(obs),rank(pred); n=len(obs); mo,mp=st.mean(ro),st.mean(rp)
    num=sum((ro[i]-mo)*(rp[i]-mp) for i in range(n))
    den=math.sqrt(sum((ro[i]-mo)**2 for i in range(n))*sum((rp[i]-mp)**2 for i in range(n)))
    sp.append(num/den)
a2.axhline(0,color="0.35",lw=0.9)
a2.plot(range(2,12),sp,"-o",color=BLUE,ms=3.5,lw=1.5)
a2.set_xticks(range(2,12)); a2.set_xlabel("$k$ (blocks kept)")
a2.set_ylabel("Spearman $\\rho$")
a2.set_ylim(-0.55,1.05); a2.yaxis.grid(True,color="0.92",lw=0.6); a2.set_axisbelow(True)
a2.set_title("(b) additive scoring vs. measured",fontsize=8,loc="left")
fig.tight_layout(); fig.savefig(OUT/"interaction.pdf",bbox_inches="tight"); plt.close(fig)
print("ok", [round(x,2) for x in sp])
