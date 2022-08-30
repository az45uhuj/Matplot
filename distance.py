import pandas as pd
import numpy as np
from numpy import linalg as LA
import math
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

def plot2(asian, black, white):
    ticks = ['RAF', 'Synthetic Gaussian Data', 'Synthetic Uniform Data']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    plt.figure(figsize=(6, 6))

    print(len(asian))
    bpl0 = plt.boxplot(asian, positions=np.array(range(len(asian))) * 2.0 - 0.4, sym='', widths=0.3)
    bpm = plt.boxplot(black, positions=np.array(range(len(black))) * 2.0, sym='', widths=0.3)
    bpr0 = plt.boxplot(white, positions=np.array(range(len(white))) * 2.0 + 0.4, sym='', widths=0.3)

    set_box_color(bpl0, '#feb24c')  # colors are from http://colorbrewer2.org/
    set_box_color(bpm, '#636363')
    set_box_color(bpr0, '#2c7fb8')  # colors are from http://colorbrewer2.org/

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#feb24c', label='Asian')
    plt.plot([], c='#636363', label='Black')
    plt.plot([], c='#2c7fb8', label='White')
    plt.legend()
    #plt.axvline(x = 1, color='#636363', ls='--', lw=0.5)

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    #plt.xlim(0, len(ticks) * 2)
    #plt.ylim(0, 8)
    #plt.title('3D FAN ')
    plt.xlabel('Test Dataset')
    plt.ylabel('NME')
    plt.tight_layout()
    plt.savefig('box_dlib.pdf', bbox_tight=True)

def p2p_mse(gth_ldmks, pre_ldmks):
    dscale = LA.norm([np.amax(gth_ldmks[2:, 0]) - np.amin(gth_ldmks[2:, 0]), np.amax(gth_ldmks[2:, 1]) - np.amin(gth_ldmks[2:, 1])])
    dis = LA.norm(gth_ldmks[2:,] - pre_ldmks[2:,]) / math.sqrt(3) / dscale
    return dis

def calculate_distance(df):
    distance = []
    for i in range(len(df)):
        g = np.array(df.iloc[i][1:11]).reshape(-1, 2)
        p = np.array(df.iloc[i][11:]).reshape(-1, 2)
        distance.append(p2p_mse(g, p))
    return distance

def get_distance(headpose):
    asian_dlib = pd.read_csv('dlib_Asian_' + headpose + '.csv')
    black_dlib = pd.read_csv('dlib_Black_' + headpose + '.csv')
    white_dlib = pd.read_csv('dlib_White_' + headpose + '.csv')

    asian_gth = pd.read_csv(headpose + '_Asian_gth_ldmks.csv')
    black_gth = pd.read_csv(headpose + '_Black_gth_ldmks.csv')
    white_gth = pd.read_csv(headpose + '_White_gth_ldmks.csv')

    asian_merge = asian_dlib.merge(asian_gth, left_on='Image', right_on='Image')
    black_merge = black_dlib.merge(black_gth, left_on='Image', right_on='Image')
    white_merge = white_dlib.merge(white_gth, left_on='Image', right_on='Image')

    asian_dis = calculate_distance(asian_merge.sample(3000))
    black_dis = calculate_distance(black_merge.sample(3000))
    white_dis = calculate_distance(white_merge.sample(3000))

    asian_dis = [x for x in asian_dis if x < 1]
    black_dis = [x for x in black_dis if x < 1]
    white_dis = [x for x in white_dis if x < 1]
    return asian_dis, black_dis, white_dis

def calculate_raf_distance(gth, pre):
    distance = []
    for i in range(len(gth)):
        g = np.array(gth.iloc[i]).reshape(-1, 2)
        p = np.array(pre.iloc[i]).reshape(-1, 2)
        distance.append(p2p_mse(g, p))
    return distance

def get_raf_data():
    # 0:white, 1:black, 2:asian
    landmaks = ['Image', 'left_eye_left_cornerx', 'left_eye_left_cornery', 'right_eye_right_cornerx', 'right_eye_right_cornery',
                    'nose_tipx', 'nose_tipy', 'mouth_left_cornerx', 'mouth_left_cornery', 'mouth_right_cornerx', 'mouth_right_cornery']
    raf_df = pd.read_csv('dlib_raf_merge.csv')
    raf_asian_pre = raf_df[raf_df['race']==2].loc[:, 'left_eye_left_cornerx':'mouth_right_cornery']
    raf_black_pre = raf_df[raf_df['race']==1].loc[:, 'left_eye_left_cornerx':'mouth_right_cornery']
    raf_white_pre = raf_df[raf_df['race']==0].loc[:, 'left_eye_left_cornerx':'mouth_right_cornery']
    raf_asian_gth = raf_df[raf_df['race']==2].loc[:, 'left_eye_centerx':'mouth_right_cornery_y']
    raf_black_gth = raf_df[raf_df['race']==1].loc[:, 'left_eye_centerx':'mouth_right_cornery_y']
    raf_white_gth = raf_df[raf_df['race']==0].loc[:, 'left_eye_centerx':'mouth_right_cornery_y']

    raf_asian_dis = calculate_raf_distance(raf_asian_gth, raf_asian_pre)
    raf_black_dis = calculate_raf_distance(raf_black_gth, raf_black_pre)
    raf_white_dis = calculate_raf_distance(raf_white_gth, raf_white_pre)
    raf_asian_dis = [x for x in raf_asian_dis if x < 1]
    raf_black_dis = [x for x in raf_black_dis if x < 1]
    raf_white_dis = [x for x in raf_white_dis if x < 1]

    return [raf_asian_dis, raf_black_dis, raf_white_dis]


def violent_plot(asian1, black1, white1):
    from matplotlib.pyplot import figure

    #figure(figsize=(12, 6), dpi=80)
    fig, ax1 = plt.subplots()

    labels = []
    ticks = ['RAF', 'Synthetic Data']#, 'Synthetic Uniform Data']

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    positions = np.arange(1, 7, 3)
    asian1 = np.array(asian1).transpose()

    add_label(ax1.violinplot(asian1, positions, showmeans=True, showmedians=True), "Asian")

    positions = np.arange(2, 7, 3)
    black1 = np.array(black1).transpose()
    add_label(ax1.violinplot(black1, positions,showmeans=True, showmedians=True), "Black")

    positions = np.arange(3, 7, 3)
    white1 = np.array(white1).transpose()
    add_label(ax1.violinplot(white1, positions,showmeans=True, showmedians=True), "White")

    cell_text = [["77.0%", "71.8%", "80.7%", "92.8%", "64.1%", "91.4%", "95.4%", "62.6%", "93.9%"]]
    cell0 = [77.0, 71.8, 80.7, 92.8, 64.1, 91.4] # 95.4, 62.6, 93.9]
    x = range(1, 10, 1)
    ax2 = ax1.twinx()
    ax2.plot(x[:3],cell0[:3], 'o--', c='#8856a7')
    ax2.plot(x[3:6],cell0[3:6], 'o--', c='#8856a7')
    #ax2.plot(x[6:],cell0[6:], 'o--', c='#8856a7')
    ax2.set_ylim(50,100)
    ax2.set_ylabel('detection rates (%)', fontsize=22)

    '''
    #cell_text = [["77.0%", "71.8, 80.7, 92.8, 64.1, 91.4, 95.4, 62.6, 93.9]]
    rows = ["detection rates"]
    col = ('RAF Asian', 'RAF Black', 'RAF White', 'Gaussian Asian', 'Gaussian Black', 'Gaussian White',
           'Uniform Asian', 'Uniform Black', 'Uniform White')
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          colLabels=col,
                          cellLoc='center',
                          loc='bottom')
    the_table.auto_set_font_size(False)

    for (row, col), cell in the_table.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    the_table.set_fontsize(6)
    #the_table.scale(1.2, 1.2)
    #plt.subplots_adjust(left=0.2, top=0.8)
    '''

    plt.axvline(x=3.5, color='#bdbdbd', label='axvline - full height')
    #plt.axvline(x=6.5, color='#bdbdbd', label='axvline - full height')
    #ax1.set_xlabel('Test Dataset')
    ax1.set_ylabel('NME', fontsize=22)
    plt.tight_layout()
    ax1.set_xticks(range(2, len(ticks) * 3, 3), ticks)
    ax1.tick_params(axis='x', labelsize=22)
    ax1.tick_params(axis='y', labelsize=15)
    plt.tick_params(axis='x', which='both', top= False, bottom=False, labelbottom=False)
    plt.tick_params(axis='y', labelsize=15)
    plt.legend(*zip(*labels), loc=2, prop={'size': 20})
    plt.savefig('dlib_violin_7.pdf')
    plt.show()

def check():
    a = np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
    print(a.reshape(-1, 3))
    print(a.transpose())
check()


gaussian0 = get_distance('gaussian')
uniform0 = get_distance('uniform')
raf = get_raf_data()
print(len(raf[0]), len(raf[1]), len(raf[2]))
print(len(gaussian0[0]), len(gaussian0[1]), len(gaussian0[2]))
#asian0 = [raf[0][:780], gaussian0[0][:780], uniform0[0][:780]]
#black0 = [raf[1][:780], gaussian0[1][:780], uniform0[1][:780]]
#white0 = [raf[2][:780], gaussian0[2][:780], uniform0[2][:780]]
asian0 = [raf[0], gaussian0[0]]#, uniform0[0]]
black0 = [raf[1], gaussian0[1]]#, uniform0[1]]
white0 = [raf[2], gaussian0[2]]#, uniform0[2]]

#plot2(asian0, black0, white0)
#plt.show()


violent_plot(asian0, black0, white0)
