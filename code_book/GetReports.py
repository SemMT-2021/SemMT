import numpy as np
import pandas as pd
from collections import defaultdict
import operator

# Get Efficacy against threshold
def FScore(precision, recall, alpha = 1):
    if (alpha ** 2 * precision + recall) == 0:
        return 0.0
    return (alpha ** 2 + 1) * precision * recall / (alpha ** 2 * precision + recall)

def get_SIT(SIT_Label, Threshold_list):
    max_distance = max(SIT_Label["Distance"].to_list())
    #threshold_list = np.arange(1,max_distance+1, 1)
    SIT_Report = {"threshold_list": [], "issues_list": [], "precision_list": [], "accuracy_list": [], "recall_list": [], "F1score_list": []}
    n_total_issues = len(SIT_Label)
    for threshold in Threshold_list:
        SIT_Label_b = SIT_Label[SIT_Label["Distance"]>=threshold]# calculate precision
        SIT_Label_c = SIT_Label[SIT_Label["Distance"]<threshold]
        TP = len(SIT_Label_b[SIT_Label_b["Label_On_Mutants_Consistent"] == "bug"])
        TN = len(SIT_Label_c[SIT_Label_c["Label_On_Mutants_Consistent"] == "correct"])
        FP = len(SIT_Label_b[SIT_Label_b["Label_On_Mutants_Consistent"] == "correct"])
        FN = len(SIT_Label_c[SIT_Label_c["Label_On_Mutants_Consistent"] == "bug"])
        
        n_issues = len(SIT_Label_b)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1Score = FScore(precision, recall, 1)
        SIT_Report["threshold_list"].append(threshold)
        SIT_Report["accuracy_list"].append(accuracy)
        SIT_Report["recall_list"].append(recall)
        SIT_Report["precision_list"].append(precision)
        SIT_Report["F1score_list"].append(F1Score)
        SIT_Report["issues_list"].append(n_issues)
        #print("after filtering issues using threshold {}, the precision is {}".format(threshold, precision))
        #print("the number of issues / total issues : {}/{}, number of buggy issues: {}".format(n_issues, n_total_issues, TP))
    return SIT_Report

def get_TransRepair(TransRepair_Label, Threshold_list):
    #threshold_list = np.arange(0.0,1, 0.1)
    #print(threshold_list)
    n_total_issues = len(TransRepair_Label)
    t_list = ["LEVEN_Score", "BLEU_Score"]
    TransRepair_Report = {}    
    for sim in t_list:
        TransRepair_Report[sim] = {"threshold_list": [], "issues_list": [], "precision_list": [], "accuracy_list": [], "recall_list": [], "F1score_list": []}
        for threshold in Threshold_list:
            TransRepair_Label_b = TransRepair_Label[TransRepair_Label[sim]<=threshold]
            TransRepair_Label_c = TransRepair_Label[TransRepair_Label[sim]>threshold]
            total_num = len(TransRepair_Label_b)
            if total_num == 0:
                # skip this part
                #print("no issues reported on threshold {}".format(threshold))
                TransRepair_Report[sim]["threshold_list"].append(threshold)
                TransRepair_Report[sim]["accuracy_list"].append(0)
                TransRepair_Report[sim]["recall_list"].append(0)
                TransRepair_Report[sim]["precision_list"].append(0)
                TransRepair_Report[sim]["F1score_list"].append(0)
                TransRepair_Report[sim]["issues_list"].append(0)
                continue
            TP = len(TransRepair_Label_b[TransRepair_Label_b["Translation_Label"] == "bug"])
            FP = len(TransRepair_Label_b[TransRepair_Label_b["Translation_Label"] == "correct"])
            TN = len(TransRepair_Label_c[TransRepair_Label_c["Translation_Label"] == "correct"])
            FN = len(TransRepair_Label_c[TransRepair_Label_c["Translation_Label"] == "bug"])
            n_issues = len(TransRepair_Label_b)
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1Score = FScore(precision, recall, 1)
            TransRepair_Report[sim]["threshold_list"].append(threshold)
            TransRepair_Report[sim]["accuracy_list"].append(accuracy)
            TransRepair_Report[sim]["recall_list"].append(recall)
            TransRepair_Report[sim]["precision_list"].append(precision)
            TransRepair_Report[sim]["F1score_list"].append(F1Score)
            TransRepair_Report[sim]["issues_list"].append(n_issues)
            #print("the precision of {} is {}, threshold is {}".format(sim, TP/total_num, threshold))
            #print("# of reported issues is {}, # of buggy issues is {}".format(total_num, TP))
    return TransRepair_Report

def get_SemMT(SemMT_Label, Threshold_list):
    t_list = ["LEVEN", "DFA", "HYB"]
    # SemMT_Label = SemMT_Label[SemMT_Label["Translation_Label"]!="wrong_mutant"]
    SemMT_Report = {}
    for sim in t_list:
        SemMT_Report[sim] = {"threshold_list": [], "issues_list": [], "precision_list": [], "accuracy_list": [], "recall_list": [], "F1score_list": []}
        for threshold in Threshold_list:
            SemMT_Label_b = SemMT_Label[SemMT_Label[sim]<=threshold]
            SemMT_Label_c = SemMT_Label[SemMT_Label[sim]>threshold]
            total_num = len(SemMT_Label_b)
            if total_num == 0:
                # skip this part
                #print("no issues reported on threshold {}".format(threshold))
                SemMT_Report[sim]["threshold_list"].append(threshold)
                SemMT_Report[sim]["accuracy_list"].append(0)
                SemMT_Report[sim]["recall_list"].append(0)
                SemMT_Report[sim]["precision_list"].append(0)
                SemMT_Report[sim]["F1score_list"].append(0)
                SemMT_Report[sim]["issues_list"].append(0)
                continue
            TP = len(SemMT_Label_b[SemMT_Label_b["Translation_Label"] == "bug"])
            FP = len(SemMT_Label_b[SemMT_Label_b["Translation_Label"] == "correct"])
            TN = len(SemMT_Label_c[SemMT_Label_c["Translation_Label"] == "correct"])
            FN = len(SemMT_Label_c[SemMT_Label_c["Translation_Label"] == "bug"])
            n_issues = len(SemMT_Label_b)
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1Score = FScore(precision, recall, 1)
            SemMT_Report[sim]["threshold_list"].append(threshold)
            SemMT_Report[sim]["accuracy_list"].append(accuracy)
            SemMT_Report[sim]["recall_list"].append(recall)
            SemMT_Report[sim]["precision_list"].append(precision)
            SemMT_Report[sim]["F1score_list"].append(F1Score)
            SemMT_Report[sim]["issues_list"].append(n_issues)
            #print("the precision of {} is {}, threshold is {}".format(sim, TP/total_num, threshold))
            #print("# of reported issues is {}, # of buggy issues is {}".format(total_num, TP))
    return SemMT_Report

def get_BugDetection_RQ1(simis, df,threshold_list):
    all_report = {}

    for sim in simis:
        print("calculating on {}".format(sim))
        temp_dict = defaultdict(list)
        for val in threshold_list:
            TP = len(df[(df[sim] <= val) & (df["Translation_Label_Evaluator1"] == "bug")])
            TN = len(df[(df[sim] > val) & (df["Translation_Label_Evaluator1"] == "correct")])
            FP = len(df[(df[sim] <= val) & (df["Translation_Label_Evaluator1"] == "correct")])
            FN = len(df[(df[sim] > val) & (df["Translation_Label_Evaluator1"] == "bug")])

            accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else 0.0
            precision = TP / (TP + FP) if TP + FP != 0 else 0.0
            recall = TP / (TP + FN) if TP + FN != 0 else 0.0
            F1Score = FScore(precision, recall, 1)
            F2Score = FScore(precision, recall, 2)

            sensitive = TP / (TP + FN) if TP + FN != 0 else 0.0
            specificity = TN / (FP + TN) if FP + TN != 0 else 0.0

            temp_dict["accuracy"].append(accuracy)
            temp_dict["precision"].append(precision)
            temp_dict["recall"].append(recall)
            temp_dict["F1Score"].append(F1Score)
            temp_dict["F2Score"].append(F2Score)
            temp_dict["sensitivity"].append(sensitive)
            temp_dict["specificity"].append(specificity)
            temp_dict["TP"].append(TP)
            temp_dict["TN"].append(TN)
            temp_dict["issues"].append(TP+FP)

        all_report[sim] = temp_dict
    return all_report

def get_issues_RQ3(all_report, simis,df):
    best_threshold = {}
    for sim in simis:
        rep = (np.array(all_report[sim]["TP"][1:-1])) * np.array(all_report[sim]["TN"][1:-1])
        index, value = max(enumerate(rep), key=operator.itemgetter(1))
        index += 1
        #print(index)
        best_threshold[sim] = "%.4f" % (index * 0.01)
        print("\n[{}]\nbest threshold: {}, value: {}".format(sim, best_threshold[sim], value))
        assert rep[index-1] == value

    df_bug = df[df["Translation_Label_Consistent"] == "bug"]
    issues_dict = defaultdict(set)
    # for val in np.arange(0.00, 1.01, 0.01):
    for sim in simis:
        issues_dict[sim] = set(df_bug[df_bug[sim] <= float(best_threshold[sim])].index)
    return issues_dict

def get_best_RQ2(SIT_Report, TransRepair_Report, SemMT_Report):
    row_name = ["SIT", "TransRepair(L)", "TransRepair(B)", "PatInv", "SemMT-R", "SemMT-D"\
                         , "SemMT-H"]
    column_name = ["# Issues", "Acc", "Recall", "Prec", "F-Score"]
    measurements_name = ["issues_list", "accuracy_list", "recall_list", "precision_list", "F1score_list"]
    Best_Report = pd.DataFrame(index=row_name, columns=column_name)

    SIT_Index = SIT_Report["F1score_list"].index(max(SIT_Report["F1score_list"]))
    for column, measure in zip(column_name, measurements_name):
        Best_Report.at["SIT", column] = SIT_Report[measure][SIT_Index]

    t_list = ["LEVEN_Score", "BLEU_Score"]
    for i, sim in enumerate(t_list):
        TransRepair_Index = TransRepair_Report[sim]["F1score_list"].index(max(TransRepair_Report[sim]["F1score_list"]))
        for column,measure in zip(column_name, measurements_name):
            Best_Report.at[row_name[1+i], column] = TransRepair_Report[sim][measure][TransRepair_Index]
    Best_Report.at["PatInv", "# Issues"] = 16
    Best_Report.at["PatInv", "Prec"] = 0.5625
    t_list = ["LEVEN", "DFA", "HYB"]
    for i, sim in enumerate(t_list):
        SemMT_Index = SemMT_Report[sim]["F1score_list"].index(max(SemMT_Report[sim]["F1score_list"]))
        for column,measure in zip(column_name, measurements_name):
            Best_Report.at[row_name[4+i], column] = SemMT_Report[sim][measure][SemMT_Index]
    return Best_Report