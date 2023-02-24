import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style('darkgrid') font='Times New Roman'
sns.set_theme(style="white",font_scale=0.8)
#自定义画图
def draw_inner(df):
    ax_mcc = sns.catplot(x="Model",
                     y="MCC",
                     # 添加order参数，指定顺序
                     order=["LR","SVM","RF","GBDT","XGB"],# 自定义
                     hue="FP",
                     kind="bar",
                     palette="muted",
                     errorbar='sd',
                     legend=None,
                     data=df)
    ax_mcc.set_ylabels("Mcc")
    #添加水平线
    #ax.refline(y=0.338,color='r',lw=2)
    # sns.move_legend(ax, 'center left')
    plt.legend(loc='lower left')
    plt.savefig("../../results/merge_fp_test/Inner_Mcc.svg", dpi=600, format='svg')
    plt.show()

    ax_auc = sns.catplot(x="Model",
                     y="ROC_AUC",
                     # 添加order参数，指定顺序
                     order=["LR","SVM","RF","GBDT","XGB"],# 自定义
                     hue="FP",
                     kind="bar",
                     palette="muted",
                     errorbar='sd',
                     legend=None,
                     data=df)
    ax_auc.set_ylabels("ROC_AUC")
    #添加水平线
    #ax.refline(y=0.338,color='r',lw=2)
    # sns.move_legend(ax, 'center left')
    plt.legend(loc='lower left')
    plt.savefig("../../results/merge_fp_test/Inner_AUC.svg",dpi=600,format='svg')
    plt.show()

def draw_external(df,name="",std_value=0.0):
    ax_mcc = sns.catplot(x="Model",
                     y="MCC",
                     # 添加order参数，指定顺序
                     order=["LR","SVM","RF","GBDT","XGB"],# 自定义
                     hue="FP",
                     kind="bar",
                     palette="muted",
                     errorbar='sd',
                     legend=None,
                     data=df)
    ax_mcc.set_ylabels("Mcc")
    #添加水平线
    ax_mcc.refline(y=std_value,color='r',lw=2)
    # sns.move_legend(ax, 'center left')
    plt.legend(loc='lower left')
    plt.savefig("../../results/merge_fp_test/%s_Mcc.svg" % (name), dpi=600, format='svg')
    plt.show()
    # draw ROC_AUC
    ax_auc = sns.catplot(x="Model",
                     y="ROC_AUC",
                     # 添加order参数，指定顺序
                     order=["LR","SVM","RF","GBDT","XGB"],# 自定义
                     hue="FP",
                     kind="bar",
                     palette="muted",
                     errorbar='sd',
                     legend=None,
                     data=df)
    ax_auc.set_ylabels("ROC_AUC")
    #添加水平线
    #ax_auc.refline(y=std_value,color='r',lw=2)
    #sns.move_legend(ax_auc, 'center left')
    plt.legend(loc='lower left')
    plt.savefig("../../results/merge_fp_test/%s_AUC.svg" % (name), dpi=600, format='svg')
    plt.show()

if __name__ == "__main__":
    df_Inner = pd.read_csv("../../results/merge_fp_test/Inner_test-all.csv")
    df_Berry =pd.read_csv("../../results/merge_fp_test/Berry_test-all.csv")
    df_Oat = pd.read_csv("../../results/merge_fp_test/Oat_test-all.csv")
    draw_inner(df_Inner)
    #yang 等人的GT-predict的效果是0.338和0。319
    draw_external(df_Berry,name="Berry",std_value=0.338)
    draw_external(df_Oat, name="Oat",std_value=0.319)
