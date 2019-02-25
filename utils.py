
def autolabel(rects, ax,fontsize=16):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                f"{height:0.3}",
                ha='center', va='bottom', fontsize=fontsize)


