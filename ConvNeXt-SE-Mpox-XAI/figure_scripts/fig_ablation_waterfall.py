from style import *
steps=["ConvNeXt\nbaseline","+ SE block","+ Label\nsmoothing","+ Weighted\nloss","+ TTA\nensemble","Full\nmodel"]
def panel(ax,vals,title,ylim):
    base=vals[0]; gains=[vals[i]-vals[i-1] for i in range(1,len(vals))]
    ax.bar(0,base,0.6,color=GREY,edgecolor="black",linewidth=0.5,zorder=3)
    ax.text(0,base+0.08,f"{base:.2f}",ha="center",va="bottom",fontsize=8.5)
    run=base
    for k,g in enumerate(gains,1):
        ax.bar(k,g,0.6,bottom=run,color=GREEN,edgecolor="black",linewidth=0.5,zorder=3)
        ax.text(k,run+g+0.08,f"+{g:.2f}",ha="center",va="bottom",fontsize=8.5,color=GREEN)
        ax.plot([k-1+0.3,k-0.3],[run,run],color="black",linewidth=0.6,linestyle=":")
        run+=g
    ax.bar(len(vals),vals[-1],0.6,color=NAVY,edgecolor="black",linewidth=0.5,zorder=3)
    ax.text(len(vals),vals[-1]+0.08,f"{vals[-1]:.2f}",ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps,fontsize=8.5)
    ax.set_ylim(*ylim); ax.set_title(title,loc="left",fontweight="bold"); ax.set_ylabel("Performance (%)"); tidy(ax)
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
panel(axes[0],[85.60,86.90,87.40,87.90,88.50],"(a) Accuracy",(82,89.5))
panel(axes[1],[83.10,84.40,85.10,85.70,86.21],"(b) Macro F1-score",(80,87.5))
fig.tight_layout(); save(fig,"Fig_Table10_ablation_waterfall")
