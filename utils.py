import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

def calc_interactions_novel(df, TPR_d, FPR_d, P_sec_read, findings, sig_findings):
    """Simulates novel pipeline and calculates percentage of findings in each leaf
    Parameters
    ----------
    df : DataFrame
        Patient data (contains AI predictions and final report inclusions)
    TPR_d : float
        Disagreement model True Positive Rate [0,1]
    FPR_d : float
        Disagreement model False Positive Rate [0,1]
    P_sec_read : float
        Probability that a prediction on a significant finding is marked as 'high quality' [0,1]
    findings : list of str
        list of all findings in df
    sig_findings : list of str
        list of clinically significant findings in df
    Returns
    -------
    interactions : np.array
        Percentage of data points in each leaf of the novel pipeline
    """
    leaf_1 = 0
    leaf_2 = 0
    leaf_3 = 0
    leaf_4 = 0
    leaf_6 = 0
    leaf_7 = 0
    leaf_8 = 0
    leaf_10 = 0
    leaf_11 = 0
    leaf_12 = 0
    leaf_5 = 0
    leaf_9 = 0
    leaf_13 = 0
    for finding in findings:
        q = np.random.binomial(1, P_sec_read, size=len(df))
        dhat = np.zeros(len(df))
        s = np.ones(len(df))* int(finding in sig_findings)
        d = df[finding] != df['ai_'+finding]
        for i in range(len(d)):
            if d[i]:
                dhat[i] = np.random.binomial(1, TPR_d)
            else:
                dhat[i] = np.random.binomial(1, FPR_d)
        f = df[finding]
        fhat = df['ai_'+finding]
        leaf_1 += np.sum((dhat == 1) & (s == 1) & (d == 0))
        leaf_2 += np.sum((dhat == 1) & (s == 1) & (d == 1) & (q == 1))
        leaf_3 += np.sum((dhat == 1) & (s == 1) & (d == 1) & (q == 0))
        leaf_4 += np.sum((dhat == 1) & (s == 0) & (f == 1))
        leaf_5 += np.sum((dhat == 1) & (s == 0) & (f == 0))
        leaf_6 += np.sum((dhat == 0) & (fhat == 1) & (d == 1) & (s == 1) & (q == 1))
        leaf_7 += np.sum((dhat == 0) & (fhat == 1) & (d == 1) & (s == 1) & (q == 0))
        leaf_8 += np.sum((dhat == 0) & (fhat == 1) & (d == 1) & (s == 0))
        leaf_9 += np.sum((dhat == 0) & (fhat == 1) & (d == 0))
        leaf_10 += np.sum((dhat == 0) & (fhat == 0) & (d == 1) & (s == 1) & (q == 1))
        leaf_11 += np.sum((dhat == 0) & (fhat == 0) & (d == 1) & (s == 1) & (q == 0))
        leaf_12 += np.sum((dhat == 0) & (fhat == 0) & (d == 1) & (s == 0))
        leaf_13 += np.sum((dhat == 0) & (fhat == 0) & (d == 0))
    interactions = np.array([leaf_1, leaf_2, leaf_3, leaf_4, leaf_5, leaf_6, 
                             leaf_7, leaf_8, leaf_9, leaf_10, leaf_11, leaf_12, leaf_13])
    interactions = interactions / (len(df) * len(findings))
    return interactions

def calc_interactions_base(df, findings):
    """Simulates baseline pipeline and calculates percentage of findings in each leaf
    Parameters
    ----------
    df : DataFrame
        Patient data (contains AI predictions and final report inclusions)
    findings : list of str
        list of findings in df
    Returns
    -------
    interactions_base : np.array
        Percentage of data points in each leaf of the baseline pipeline
    """
    leaf_1_base = 0
    leaf_1_2_base = 0
    leaf_2_base = 0
    leaf_2_2_base = 0
    for finding in findings:
        d = (df[finding] != df['ai_'+finding]).astype(int)
        f = df[finding]
        fhat = df['ai_'+finding]
        leaf_1_base += np.sum((fhat == 1) & (d == 0))
        leaf_1_2_base += np.sum((fhat == 1) & (d == 1))
        leaf_2_base += np.sum((fhat == 0) & (d == 1))
        leaf_2_2_base += np.sum((fhat == 0) & (d == 0))
    interactions_base = np.array([leaf_1_base, leaf_2_base, leaf_1_2_base, leaf_2_2_base])
    interactions_base = interactions_base / (len(df) * len(findings))
    return interactions_base

def pipeline_comparison(df, TPR_d, FPR_d, P_sec_read, findings, sig_findings, constants, print_it=False):
    """Comares burden in novel pipeline to novel pipline through interactions
    Parameters
    ----------
    df : DataFrame
        Patient data (contains AI predictions and final report inclusions)
    TPR_d : float
        Disagreement model True Positive Rate [0,1]
    FPR_d : float
        Disagreement model False Positive Rate [0,1]
    P_sec_read : float
        Probability that a prediction on a significant finding is marked as 'high quality' [0,1]
    findings : list of str
        list of all findings in df
    sig_findings : list of str
        list of clinically significant findings in df
    Returns
    -------
    burden_ratio : float
        Percentage of data points in each leaf of the novel pipeline
    """
    interactions = calc_interactions_novel(df, TPR_d, FPR_d, P_sec_read, findings, sig_findings)
    S, click, search, AI, QA, RR = constants
    weights = np.array([RR + click, #1
                    RR + click + RR + click, #2
                    RR + click, #3
                    QA + click, #4
                    QA,#5
                    AI + click + RR + click, #6
                    AI + click, #7
                    AI + click, #8
                    AI,#9
                    S + search + RR + click, #10
                    S + search, #11
                    S + search, #12
                    S,#13
                   ])
    tot_inter = 0
    for i in range(len(weights)):
        curr_inter = interactions[i] * weights[i]
        tot_inter += curr_inter
        if print_it:
            print('Leaf', i+1,':',round(curr_inter, 3))
    interactions_base = calc_interactions_base(df, findings)
    w1 = QA + click
    w2 = S + search
    w1_2 = QA
    w2_2 = S
    weights_base = np.array([w1, w2, w1_2, w2_2])
    tot_inter_base = 0
    for i in range(len(interactions_base)):
        tot_inter_base += interactions_base[i]*weights_base[i]
    burden_ratio = tot_inter / tot_inter_base
    return burden_ratio

def pipeline_comparison_by_leaf(df, TPR_d, FPR_d, P_sec_read, findings, sig_findings, constants, weighted=False):
    """Break down novel pipline interaction contribution by leaf
    Parameters
    ----------
    df : DataFrame
        Patient data (contains AI predictions and final report inclusions)
    TPR_d : float
        Disagreement model True Positive Rate [0,1]
    FPR_d : float
        Disagreement model False Positive Rate [0,1]
    P_sec_read : float
        Probability that a prediction on a significant finding is marked as 'high quality' [0,1]
    findings : list of str
        list of all findings in df
    sig_findings : list of str
        list of clinically significant findings in df
    weighted : boolean
        whether to return fraction of data points in each leaf weighted by interaction cost
    Returns
    -------
    leaf_inters(_weighted) : float
        Percentage of data points in each leaf of the novel pipeline (may be weighted s.t. sum != 1)
    """
    interactions = calc_interactions_novel(df, TPR_d, FPR_d, P_sec_read, findings, sig_findings)
    S, click, search, AI, QA, RR = constants
    weights = np.array([RR + click, #1
                    RR + click + RR + click, #2
                    RR + click, #3
                    QA + click, #4
                    QA,#5
                    AI + click + RR + click, #6
                    AI + click, #7
                    AI + click, #8
                    AI,#9
                    S + search + RR + click, #10
                    S + search, #11
                    S + search, #12
                    S,#13
                   ])
    leaf_inters_weighted = []
    for i in range(len(weights)):
        curr_inter = interactions[i] * weights[i]
        leaf_inters_weighted.append(curr_inter)
    leaf_inters = interactions
    leaf_inters_weighted = np.array(leaf_inters_weighted)
    if weighted:
        return leaf_inters_weighted
    else:
        return leaf_inters

def pipeline_comparison_by_leaf_base(df, findings, constants, weighted=False):
    """Break down baseline pipline interaction contribution by leaf
    Parameters
    ----------
    df : DataFrame
        Patient data (contains AI predictions and final report inclusions)
    findings : list of str
        list of all findings in df
    weighted : boolean
        whether to return fraction of data points in each leaf weighted by interaction cost
    Returns
    -------
    leaf_inters(_weighted) : float
        Percentage of data points in each leaf of the baseline pipeline (may be weighted s.t. sum != 1)
    """
    S, click, search, AI, QA, RR = constants
    w1 = QA + click
    w2 = S + search
    w1_2 = QA
    w2_2 = S
    weights_base = np.array([w1, w2, w1_2, w2_2])
    interactions_base = calc_interactions_base(df, findings)
    leaf_inters_base_weighted = []
    for i in range(len(weights_base)):
        curr_inter = interactions_base[i] * weights_base[i]
        leaf_inters_base_weighted.append(curr_inter)
    if weighted:
        return leaf_inters_base_weighted
    else:
        return interactions_base

def fill_above_and_below_line(max_num=2, min_num=0, ax=None, alpha=.2, color1='green', color2='red', **kwargs):
    """Fill above and below line at 1 (2 different colors)
    Parameters
    ----------
    max_num : float/int
        highest point for top color (default 2)
    min_num : float/int
        lowest point for bottom color (default 0)
    ax : Axis Obj (optional)
    alpha : float
        alpha value for fill in colors (transparency) (default 0.2)
    color1 : str
        bottom color (default green)
    color2 : str
        top color (default red)
    """
    if ax is None:
        ax = plt.gca()
    line = ax.lines[-1]
    x, y = line.get_xydata().T
    if max_num > 1:
        ax.fill_between(x, y, max_num, color=color2, alpha=alpha, **kwargs)
    if min_num < 1:
        ax.fill_between(x, min_num, y, color=color1, alpha=alpha, **kwargs)
        
def plot_Disagreement_sim_TPR_line(df_all, FPR_d, P_sec_read, findings, sig_findings, constants, TPR_ds = np.arange(0.3, 1, 0.02), 
                                   text=False, n_trials = 5, slope_rot = -13, ytick_size = 0.005, save_fig = False,
                                   subset_size=1):
    """Plots burden reduction as a function of disagreement model true positive rate
    Parameters
    ----------
    TPR_ds : np.array
        Disagreement model True Positive Rates [0,1]
    FPR_d : float
        Default disagreement model False Positive Rate [0,1]
    P_sec_read : float
        Default probability that a prediction on a significant finding is marked as 'high quality' [0,1]
    findings : list of str
        list of all findings in df
    sig_findings : list of str
        list of clinically significant findings in df
    text : boolean
        whether to include text on the plot
    n_trials : int
        number of trials (data is sampled at full size with replacement)
    slope_rot : int
        how much to rotate slope text (unused in this version)
    ytick_size : float
        how far apart yticks are (unused in this version)
    save_fig : boolean
        whether to save the figure
    subset_size : float
        fraction of data size to sample with replacement (default 1)
    """
    sns.set_style("white")
    sns.despine()
    score = []
    trial = []
    TPR_d = []
    for t in range(n_trials):
        for TPR_d_ in range(len(TPR_ds)):
            trial.append(t)
            TPR_d.append(TPR_ds[TPR_d_])
            df = df_all.sample(frac=subset_size, replace=True).reset_index()
            score.append(pipeline_comparison(df, TPR_ds[TPR_d_], FPR_d, P_sec_read, findings, sig_findings, constants))
    df_scores = pd.DataFrame(np.array([trial, TPR_d, score]).T, columns = ['trial', 'TPR_d', 'score'])
    sns.lineplot(data=df_scores, x='TPR_d', y='score', color='black')
    sns.lineplot(TPR_ds, np.ones(len(TPR_ds)),linestyle='--', color='black', alpha=0.5)
    fill_above_and_below_line(max_num=1.2, min_num = min(0.9, min(df_scores['score'])))
    plt.yticks(np.arange(0.9,1.201,0.05))
    plt.xticks(np.round(np.linspace(df_scores['TPR_d'].min(), df_scores['TPR_d'].max(), 8),2), rotation=0) 
    slope, b = np.polyfit(df_scores['TPR_d'], df_scores['score'], 1)
    if text:
        vert = (max(1, df_scores['score'].max()) - df_scores['score'].min())/25
        hor = (df_scores['TPR_d'].max() - df_scores['TPR_d'].min())/50
        plt.text(df_scores['TPR_d'].max()-hor*20, 1+vert, "Burden Increase", horizontalalignment='left',
                 verticalalignment='bottom', size='medium', color='black', weight='semibold')
        plt.text(df_scores['TPR_d'].max()-hor*20, 1-vert*1.23, "Burden Reduction", horizontalalignment='left', 
                 verticalalignment='top', size='medium', color='black', weight='semibold')
    sns.set_context("talk")
    sns.despine(trim=True, offset={'bottom':-10, 'left':-13})
    plt.xlabel('Disagreement Model TPR',fontsize=13)
    plt.ylabel('Novel Pipeline to Baseline Burden Ratio',fontsize=13)
    plt.title('Burden Ratio as a Function of Disagreement TPR',fontsize=13, fontweight="bold")
    if save_fig:
        plt.tight_layout()
        plt.savefig('Disagreement_sim_TPR_line.png', dpi=300)
    plt.show()
    sns.set()
    
def plot_Disagreement_sim_FPR_line(df_all, TPR_d, P_sec_read, findings, sig_findings, constants, FPR_ds = np.arange(0.01, 0.5, 0.05),
                                   text=False, n_trials = 5, slope_rot = 32, ytick_size = 0.02, save_fig=False, 
                                   subset_size=1):
    """Plots burden reduction as a function of disagreement model false positive rate
    Parameters
    ----------
    FPR_ds : np.array
        Disagreement model False Positive Rates [0,1]
    TPR_d : float
        Default disagreement model True Positive Rate [0,1]
    P_sec_read : float
        Default probability that a prediction on a significant finding is marked as 'high quality' [0,1]
    findings : list of str
        list of all findings in df
    sig_findings : list of str
        list of clinically significant findings in df
    text : boolean
        whether to include text on the plot
    n_trials : int
        number of trials (data is sampled at full size with replacement)
    slope_rot : int
        how much to rotate slope text (unused in this version)
    ytick_size : float
        how far apart yticks are (unused in this version)
    save_fig : boolean
        whether to save the figure
    subset_size : float
        fraction of data size to sample with replacement (default 1)
    """
    sns.set_style("white")
    score = []
    trial = []
    FPR_d = []
    for t in range(n_trials):
        for FPR_d_ in range(len(FPR_ds)):
            trial.append(t)
            FPR_d.append(FPR_ds[FPR_d_])
            df = df_all.sample(frac=subset_size, replace=True).reset_index()
            score.append(pipeline_comparison(df, TPR_d, FPR_ds[FPR_d_], P_sec_read, findings, sig_findings, constants))
    df_scores = pd.DataFrame(np.array([trial, FPR_d, score]).T, columns = ['trial', 'FPR_d', 'score'])
    sns.lineplot(data=df_scores, x='FPR_d', y='score', color='black')
    sns.lineplot(FPR_ds, np.ones(len(FPR_ds)),linestyle='--', color='black', alpha=0.5)
    fill_above_and_below_line(max_num=1.2, min_num = min(0.9, min(df_scores['score'])))
    plt.yticks(np.arange(0.9,1.201,0.05))
    plt.xticks(np.round(np.linspace(df_scores['FPR_d'].min(), df_scores['FPR_d'].max(), 5),2), rotation=0) 
    slope, b = np.polyfit(df_scores['FPR_d'], df_scores['score'], 1)
    if text:
        vert = (max(1,df_scores['score'].max()) - df_scores['score'].min())/25
        hor = (df_scores['FPR_d'].max() - df_scores['FPR_d'].min())/50
        plt.text(df_scores['FPR_d'].max()-hor*20, 1+vert, "Burden Increase", horizontalalignment='left',
                 verticalalignment='bottom', size='medium', color='black', weight='semibold')
        plt.text(df_scores['FPR_d'].max()-hor*20, 1-vert*1.23, "Burden Reduction", 
                 horizontalalignment='left', verticalalignment='top', size='medium', color='black', weight='semibold')
    sns.set_context("talk")
    sns.despine(trim=True, offset={'bottom':-10, 'left':-13})
    plt.xlabel('Disagreement Model FPR',fontsize=13)
    plt.ylabel('Novel Pipeline to Baseline Burden Ratio',fontsize=13)
    plt.title('Burden Ratio as a Function of Disagreement FPR',fontsize=13, fontweight="bold")
    if save_fig:
        plt.tight_layout()
        plt.savefig('Disagreement_sim_FPR_line.jpg', dpi=300)
    plt.show()
    sns.set()

def plot_Disagreement_sim_Pred_Qual_line(df_all, TPR_d, FPR_d, findings, sig_findings, constants, P_sec_reads = np.arange(0, 1, 0.05),
                                         text=False, n_trials = 5, slope_rot = 32, ytick_size = 0.005, 
                                         save_fig = False, subset_size=1):
    """Plots burden reduction as a function of second read prevalence (among disagreement on severe findings)
    Parameters
    ----------
    P_sec_reads : float
        Probabilities that a prediction on a significant finding is marked as 'high quality' [0,1]
    TPR_d : np.array
        Disagreement model True Positive Rate [0,1]
    FPR_d : float
        Default disagreement model False Positive Rate [0,1]
    findings : list of str
        list of all findings in df
    sig_findings : list of str
        list of clinically significant findings in df
    text : boolean
        whether to include text on the plot
    n_trials : int
        number of trials (data is sampled at full size with replacement)
    slope_rot : int
        how much to rotate slope text (unused in this version)
    ytick_size : float
        how far apart yticks are (unused in this version)
    save_fig : boolean
        whether to save the figure
    subset_size : float
        fraction of data size to sample with replacement (default 1)
    """
    sns.set_style("white")
    score = []
    trial = []
    P_sec_read = []
    for t in range(n_trials):
        for P_sec_read_ in range(len(P_sec_reads)):
            trial.append(t)
            P_sec_read.append(P_sec_reads[P_sec_read_])
            df = df_all.sample(frac=subset_size, replace=True).reset_index()
            score.append(pipeline_comparison(df, TPR_d, FPR_d, P_sec_reads[P_sec_read_], findings, sig_findings, constants))
    df_scores = pd.DataFrame(np.array([trial, P_sec_read, score]).T, columns = ['trial', 'P_sec_read', 'score'])
    sns.lineplot(data=df_scores, x='P_sec_read', y='score', color='black')
    sns.lineplot(P_sec_reads, np.ones(len(P_sec_reads)),linestyle='--', color='black', alpha=0.5)
    fill_above_and_below_line(max_num=1.2, min_num = min(0.9, min(df_scores['score'])))
    plt.yticks(np.arange(0.9,1.201,0.05))
    plt.xticks(np.round(np.linspace(df_scores['P_sec_read'].min(), df_scores['P_sec_read'].max(), 8),2), rotation=0) 
    slope, b = np.polyfit(df_scores['P_sec_read'], df_scores['score'], 1)
    if text:
        vert = (max(1, df_scores['score'].max()) - df_scores['score'].min())/25
        hor = (df_scores['P_sec_read'].max() - df_scores['P_sec_read'].min())/50
        plt.text(df_scores['P_sec_read'].min()+hor*2, 1+vert, "Burden Increase", horizontalalignment='left',
                 verticalalignment='bottom', size='medium', color='black', weight='semibold')
        plt.text(df_scores['P_sec_read'].min()+hor*2, 1-vert*1.23, "Burden Reduction", 
                 horizontalalignment='left', verticalalignment='top', size='medium', color='black', 
                 weight='semibold')
    sns.set_context("talk")
    sns.despine(trim=True, offset={'bottom':-10, 'left':-13})
    plt.xlabel('P_sec_read',fontsize=13)
    plt.ylabel('Novel Pipeline to Baseline Burden Ratio',fontsize=13)
    plt.title(r'Burden Ratio as a Function of $\bf{P_{Second\_Read}}$',fontsize=13, fontweight="bold")
    if save_fig:
        plt.tight_layout()
        plt.savefig('Disagreement_sim_P_sec_read_line.jpg', dpi=300)
    plt.show()
    sns.set()