from style import *
import numpy as np
from matplotlib.patches import Patch
rows=[("Ali et al. [23]\n5-fold CV",83.59,2.11,None,None,0),
("Biswas & Tešić [24]\nTest split",85.78,None,84.20,None,0),
("Vuran et al. [16]\nSingle split",93.71,None,None,None,0),
("MSMP-Net [15]\noffline aug.\n5-fold CV",87.03,3.43,86.58,3.66,0),
("MSMP-Net [15]\noriginal only\n5-fold CV",79.45,5.66,None,None,0),
("DI-SESCA-MSNET [25]\n5-fold CV",89.38,4.20,88.83,4.41,0),
("Proposed\n3×5-fold CV",85.41,2.54,84.23,2.60,1),
("Proposed\nheld-out test (TTA)",88.50,None,86.21,None,1)]
fig,ax=plt.subplots(figsize=(14,5.4)); w=0.36
for i,(lab,a,asd,f,fsd,p) in enumerate(rows):
    if p: ax.axvspan(i-0.5,i+0.5,color="#e8f1e4",zorder=0)
    ax.bar(i-w/2,a,w,color=NAVY,edgecolor="black",linewidth=0.5,yerr=asd,capsize=3,zorder=3)
    ax.text(i-w/2,a+(asd or 0)+0.4,f"{a:.2f}",ha="center",fontsize=8)
    if f is not None:
        ax.bar(i+w/2,f,w,color=ORANGE,edgecolor="black",linewidth=0.5,yerr=fsd,capsize=3,zorder=3)
        ax.text(i+w/2,f+(fsd or 0)+0.4,f"{f:.2f}",ha="center",fontsize=8)
    else: ax.text(i+w/2,71,"n/r",ha="center",fontsize=8,color="gray",style="italic")
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows],fontsize=8.5)
for t,r in zip(ax.get_xticklabels(),rows):
    if r[5]: t.set_fontweight("bold")
ax.set_ylim(70,100); ax.set_ylabel("Performance (%)"); tidy(ax)
ax.legend(handles=[Patch(facecolor=NAVY,edgecolor="black",label="Accuracy"),
 Patch(facecolor=ORANGE,edgecolor="black",label="Macro-F1"),
 Patch(facecolor="#e8f1e4",edgecolor="gray",label="This study")],loc="upper left",frameon=False,ncol=3)
save(fig,"Fig10_comparison")
