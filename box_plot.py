import json
from matplotlib import pyplot as plt
import numpy as np

def box_plot(dis1):
    class0_1 = dis1[0]
    class1_1 = dis1[1]
    class2_1 = dis1[2]
    class3_1 = dis1[3]

    ticks = ['data0', 'data1', 'data2']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=(6,6))

    bpl0 = plt.boxplot(class0_1, positions=np.array(range(len(class0_1))) * 2.0 - 0.6, sym='', widths=0.3)
    bpr0 = plt.boxplot(class1_1, positions=np.array(range(len(class1_1))) * 2.0 - 0.3, sym='', widths=0.3)
    bpl1 = plt.boxplot(class2_1, positions=np.array(range(len(class2_1))) * 2.0 + 0.3, sym='', widths=0.3)
    bpr1 = plt.boxplot(class3_1, positions=np.array(range(len(class3_1))) * 2.0 + 0.6, sym='', widths=0.3)

    set_box_color(bpl0, '#feb24c')  # colors are from http://colorbrewer2.org/
    set_box_color(bpr0, '#636363')
    set_box_color(bpl1, '#2c7fb8')  # colors are from http://colorbrewer2.org/
    set_box_color(bpr1, '#31a354')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#feb24c', label='Class0')
    plt.plot([], c='#636363', label='Class1')
    plt.plot([], c='#2c7fb8', label='Class2')
    plt.plot([], c='#31a354', label='Class3')
    plt.legend(prop={'size':20}, loc='upper left')

    plt.axvline(x = 1, color='#636363', ls='--', lw=0.5)
    plt.axvline(x = 3, color='#636363', ls='--', lw=0.5)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=22)
    plt.tick_params(axis='y',  labelsize=15)

    #plt.xlim(0, len(ticks) * 2)
    #plt.ylim(0, 8)
    plt.ylabel('Evaluation_metrix', fontsize=22)
    plt.tight_layout()
    plt.savefig('box_plot_example.pdf')
    plt.show()



dis0 = np.random.normal(3, 2.2, (4,3,5)).tolist()
box_plot(dis0)