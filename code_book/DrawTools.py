# coding: utf-8
from itertools import chain
from collections import Iterable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import matplotlib.patheffects as path_effects
import matplotlib
import numpy as np
import math
import venn
import imp

def draw_scatter_RQ2(SemMT_Report, SIT_Report, TransRepair_Report):
    x_axis = np.arange(0,1.1,0.1)
    x_axis_SIT = list(range(0,17))
    x_axis_trans = np.arange(0.0, 1, 0.1)
    figure_k = 1
    plt.figure(figsize=(16,8))

    SMALL_SIZE = 12
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20
    LINEWIDTH=2

    ATITLE_SIZE = 20
    LABEL_SIZE = 20
    XTICK_SIZE = 20
    YTICK_SIZE = 20
    LEGEND_SIZE = 15
    FTITLA_SIZE = 20
    SCATTER_FONT_SIZE = 14
    SCATTER_LINE_SIZE = 2.5
    SCATTER_AREA = 100

    # plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=ATITLE_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=LABEL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=XTICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=YTICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=FTITLA_SIZE)  # fontsize of the figure title

    t_list = ["LEVEN", "DFA", "HYB", "SBERT"]

    ax1 = plt.subplot(1,1,1)
    # ax1.set_xlabel('# of issues')
    # ax1.set_ylabel('precision')
    ax1.set_ylim(bottom=0, top=1.001)
    ax1.set_xlim(left=-10, right=230)
    ax1.tick_params(axis='y')
    PatInv_Report = {}
    PatInv_Report["issues_list"] = [16]
    PatInv_Report["precision_list"] = [0.5652]
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylim(bottom=0, top=160)
    # original color: #C00001, #44546A
    fi_color = "#C00001"
    color_1 = fi_color
    color_2 = fi_color
    color_3 = fi_color
    # color_2 = "#C00001"
    # color_3 = "#C00001"


    se_color = '#44546A'
    color_4 = se_color
    color_5 = se_color
    color_6 = se_color

    # color_7 = '#9DC3E5'
    color_7 = se_color
    color_8 = se_color
    # ax2.set_ylabel('# of issues', color=color_2)  # we already handled the x-label with ax1
    # ax2.tick_params(axis='y',labelcolor=color_2)
    style_1 = "x"
    style_2 = "o"
    style_3 = "+"
    style_4 = "o"
    style_5 = "x"
    style_6 = "o"
    style_7 = "+"
    style_8 = "*"
    line_style = '-.'
    lns1 = ax1.scatter(SemMT_Report[t_list[0]]["issues_list"], SemMT_Report[t_list[0]]["precision_list"]\
                       , marker=style_1, color=color_1, label="SemMT-R: [0, 0.1, ..., 1.0)",\
                       linewidth=SCATTER_LINE_SIZE, s=SCATTER_AREA)
    lns2 = ax1.scatter(SemMT_Report[t_list[1]]["issues_list"], SemMT_Report[t_list[1]]["precision_list"]\
                       , marker=style_2, color='',edgecolors=color_2, label="SemMT-D: [0, 0.1, ..., 1.0)",\
                       linewidth=SCATTER_LINE_SIZE, s=SCATTER_AREA)
    lns3 = ax1.scatter(SemMT_Report[t_list[2]]["issues_list"], SemMT_Report[t_list[2]]["precision_list"]\
                       , marker=style_3, color=color_3, label="SemMT-H: [0, 0.1, ..., 1.0)",\
                       linewidth=SCATTER_LINE_SIZE, s=SCATTER_AREA*1.3)
    # lns4 = ax1.scatter(n_issues_dict[t_list[3]], precision_dict[t_list[3]]\
    #                    , marker=style_4, color=color_4, label="Sbert: [0.1, 0.2, ..., 1.0]",\
    #                    linewidth=SCATTER_LINE_SIZE, s=SCATTER_AREA)
    lns5 = ax1.scatter(SIT_Report["issues_list"], SIT_Report["precision_list"]\
                      , marker=style_5, color=color_5, label="SIT: [1, 2, ..., 17]",\
                       linewidth=SCATTER_LINE_SIZE, s=SCATTER_AREA)
    lns6 = ax1.scatter(PatInv_Report["issues_list"], PatInv_Report["precision_list"]\
                      , marker=style_6, color='', edgecolors=color_6, label="PatInv",\
                       linewidth=SCATTER_LINE_SIZE, s=SCATTER_AREA)
    lns7 = ax1.scatter(TransRepair_Report["BLEU_Score"]["issues_list"],\
                       TransRepair_Report["BLEU_Score"]["precision_list"]\
                      , marker=style_7, color=color_7, label="TransRepair(B): [0, 0.1, ..., 1.0)",\
                       linewidth=SCATTER_LINE_SIZE, s=SCATTER_AREA*1.3)

    lns8 = ax1.scatter(TransRepair_Report["LEVEN_Score"]["issues_list"],\
                       TransRepair_Report["LEVEN_Score"]["precision_list"]\
                      , marker=style_8, color='', edgecolors=color_8, label="TransRepair(L): [0, 0.1, ..., 1.0)",\
                       linewidth=SCATTER_LINE_SIZE-1, s=SCATTER_AREA*1.3)

    x_r_axis = np.arange(0.1,300, 1)
    for k in np.arange(5, 50,10):
        y_r_axis = [k/x for x in x_r_axis]
        ax1.plot(x_r_axis, y_r_axis, "--", color='k', linewidth=1)
    ax1.grid(linestyle='--', linewidth=1.5)
    ax1.legend(ncol=3,edgecolor='k')
    # pdf = PdfPages('../en_mutate_rules/plot/ScatterPlot.pdf')
    # pdf.savefig(bbox_inches='tight')
    # pdf.close()

def draw_linechart_RQ1(all_report, simis):
    plt.figure(figsize=(30, 20))

    metrics = ["accuracy","precision", "recall","F1Score", "F2Score", "sensitivity", "specificity"]
    plotid=[331, 332, 333,334, 335, 336, 337]

    for i, metric in enumerate(metrics):
        plt.subplot(plotid[i])
        for sim in simis:
            rep = all_report[sim]
            plt.plot(np.arange(0., len(rep[metric]), 1), rep[metric], label=sim)
            
        plt.grid()
        plt.title(metric)
    plt.legend()

def draw_venn_RQ3(issues_dict, target):
    imp.reload(venn)
    leven_text = "LEV ({})".format(len(issues_dict["LEVEN_Norm"]))
    dep_text = "DEP ({})".format(len(issues_dict["DEP_Norm"]))
    bleu_text = "BLEU ({})".format(len(issues_dict["BLEU_Norm"]))
    sbert_text = "SBERT ({})".format(len(issues_dict["SBERT_Norm"]))
    dfa_text = "DFA ({})".format(len(issues_dict["DFA_Norm"]))
    reg_text = "REG ({})".format(len(issues_dict["REG_Norm"]))
    hyb_text = "HYB ({})".format(len(issues_dict["HYB_Norm"]))
    replace = {"LEVEN_Norm":leven_text,
               "DEP_Norm": dep_text,
               "BLEU_Norm": bleu_text,
               "SBERT_Norm": sbert_text,
               "DFA_Norm": dfa_text,
               "REG_Norm": reg_text,
               "HYB_Norm": hyb_text}
    new_issues_dict = {replace[k]: set(map(lambda x: str(x), v)) for k, v in issues_dict.items()}
    cmap = matplotlib.cm.get_cmap("viridis_r", 5)
    color_list = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        color_list.append(matplotlib.colors.rgb2hex(rgb))
    color_list[0] = "#E68C8D"
    
    if target == "DFA":
        _ = new_issues_dict.pop(reg_text)
        _ = new_issues_dict.pop(hyb_text)
        labels = venn.get_labels([new_issues_dict[leven_text], new_issues_dict[dep_text], new_issues_dict[bleu_text], \
                                 new_issues_dict[sbert_text], new_issues_dict[dfa_text]], fill=['number'])
        fig, ax = venn.venn5(labels, names=[leven_text, dep_text, bleu_text, sbert_text, dfa_text]
                             , colors=color_list
                            )
    elif target == "REG":
        _ = new_issues_dict.pop(dfa_text)
        _ = new_issues_dict.pop(hyb_text)
        labels = venn.get_labels([new_issues_dict[leven_text], new_issues_dict[dep_text], new_issues_dict[bleu_text], \
                                 new_issues_dict[sbert_text], new_issues_dict[reg_text]], fill=['number'])
        fig, ax = venn.venn5(labels, names=[leven_text, dep_text, bleu_text, sbert_text, reg_text]
                             , colors=color_list
                            )
    elif target == "HYB":
        _ = new_issues_dict.pop(dfa_text)
        _ = new_issues_dict.pop(reg_text)
        labels = venn.get_labels([new_issues_dict[leven_text], new_issues_dict[dep_text], new_issues_dict[bleu_text], \
                                 new_issues_dict[sbert_text], new_issues_dict[hyb_text]], fill=['number'])
        fig, ax = venn.venn5(labels, names=[leven_text, dep_text, bleu_text, sbert_text, hyb_text]
                             , colors=color_list
                            )
