import numpy as np
import pandas as pd

############# Load Reports of SIT ##################
save_folder = "../data/RQ2-Existingwork-Comparison/"
def read_SIT():
    label_result = pd.read_csv(save_folder+"SIT/SIT_Issues_Label.csv")
    print("total issues: {}".format(len(label_result)))

    # load issues.txt
    with open(save_folder+"SIT/issues.txt", encoding="utf-8") as file:
        content = file.read().split("\n\n")[:-1]
        
    flag = 0
    content_index = 0
    for row_index in range(len(label_result)):
        block1 = content[content_index]
        line = block1.split("\n")
        label_result.at[row_index, "ID"] = line[0][4:]
        label_result.at[row_index, "Initial_Seeds"] = line[2]
        label_result.at[row_index, "Translated_Seeds"] = line[4]
        content_index += 1
        block2 = content[content_index]
        line = block2.split("\n")
        label_result.at[row_index, "Distance"] = float(line[0][10:])
        label_result.at[row_index, "Mutants"] = line[1]
        label_result.at[row_index, "Translated_Mutants"] = line[2] # only keep the top 1 result
        content_index += 1

    target_index = (label_result["Remark_Evaluator2"]!="wrong_mutants")
    label_result = label_result[target_index]
    print("after check inconsistency: {}".format(len(label_result)))
    return label_result

def read_TransRepair():
    TransRepair_Label = pd.read_csv(save_folder+"TransRepair/TransRepair_Issues_Label.csv", encoding="utf-8")
    print("number of total mutants is {}".format(len(TransRepair_Label)))
    TransRepair_Label = TransRepair_Label[TransRepair_Label["Translation_Label_Remark"]!="wrong mutants"]
    print("number after manual check {}".format(len(TransRepair_Label)))
    return TransRepair_Label

def read_SemMT():
    SemMT_Label = pd.read_csv(save_folder+"SemMT/SemMT_Issues_Label.csv", encoding="utf-8")
    mask = SemMT_Label["DFA"] == "False"
    SemMT_Label["DFA"].loc[mask] = 0
    SemMT_Label["DFA"] = SemMT_Label.apply(lambda x: float(x["DFA"]), axis=1)
    mask = SemMT_Label["DFA"] == -1
    SemMT_Label["DFA"].loc[mask] = 0

    SemMT_Label["HYB"] = SemMT_Label.apply(lambda x: 0.5*x["LEVEN"]\
                                                                +0.5*x["DFA"], axis=1)
    return SemMT_Label