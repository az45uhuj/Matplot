import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

def violin_plot(class0_1, class1_1, class2_1):
 
    #figure(figsize=(12, 6), dpi=80)
    fig, ax1 = plt.subplots()

    labels = []
    ticks = ['Data0', 'Data1', 'Data2']

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    positions = np.arange(1, 10, 3) # 3*3+1
    asian1 = np.array(asian1).transpose()

    add_label(ax1.violinplot(class0_1, positions, showmeans=True, showmedians=True), "Class0")

    positions = np.arange(2, 10, 3)
    black1 = np.array(black1).transpose()
    add_label(ax1.violinplot(class1_1, positions,showmeans=True, showmedians=True), "Class1")

    positions = np.arange(3, 10, 3)
    white1 = np.array(white1).transpose()
    add_label(ax1.violinplot(class2_1, positions,showmeans=True, showmedians=True), "Class2")

    plt.axvline(x=3.5, color='#bdbdbd', label='axvline - full height')
    plt.axvline(x=6.5, color='#bdbdbd', label='axvline - full height')
    ax1.set_xlabel('data')
    ax1.set_ylabel('evaluation metric', fontsize=22)
    plt.tight_layout()
    ax1.set_xticks(range(2, len(ticks) * 3, 3), ticks)
    ax1.tick_params(axis='x', labelsize=22)
    ax1.tick_params(axis='y', labelsize=15)
    plt.tick_params(axis='x', which='both', top= False, bottom=False, labelbottom=False)
    plt.tick_params(axis='y', labelsize=15)
    plt.legend(*zip(*labels), loc=2, prop={'size': 20})
    plt.savefig('violin_plot.pdf', bbox_inches='tight')
    plt.show()

class0 = [data0[0], data2[0], data3[0]]
class1 = [data0[1], data2[1], data3[1]]
class2 = [data0[2], data2[2], data3[2]]

violent_plot(class0, class1, class2)
