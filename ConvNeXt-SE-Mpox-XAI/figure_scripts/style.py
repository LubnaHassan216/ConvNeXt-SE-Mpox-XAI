import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":11,
                     "axes.labelsize":10,"pdf.fonttype":42,"ps.fonttype":42})
NAVY,LIGHT,ORANGE,GREEN,GREY="#1f4e79","#7fa8c9","#e08a1e","#2e7d32","#8c8c8c"
def tidy(ax):
    ax.yaxis.grid(True,linestyle="--",alpha=0.4); ax.set_axisbelow(True)
    for s in("top","right"): ax.spines[s].set_visible(False)
def save(fig,name):
    fig.savefig(name+".png",dpi=600,bbox_inches="tight",facecolor="white")
    fig.savefig(name+".pdf",bbox_inches="tight",facecolor="white")
    plt.close(fig)
