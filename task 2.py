import matplotlib.pyplot as plt
matplotlib.pyplot.figure(figsize=None, facecolor=None, edgecolor=None)
fig=plt.figur(figsize=[6.4, 4.8],facecolor='skyblue', edgecolor='black')
ax=fig.add_axes([0,0,1,1],projection='rectilinear',xlabel='X-label',ylabel='Y-label,title='Creating new Figure & axes')