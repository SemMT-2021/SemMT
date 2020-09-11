# SemMT

## Prerequisite

Python 3.x

```shell
pip3 install -r requirements.txt
```



## Training Data

The training data of our regular expression transformer can be found at [NL-RX-Synth-Augmented.txt](./data/Training-Data/NL-RX-Synth-Augmented.txt)

The original data can be found at [NL-RX-Synth.txt](./data/Training-Data/NL-RX-Synth.txt)



## RQ1 and RQ3

For the dataset and manual label result of Bug Detection in RQ1 and RQ3, please see [bug_detection_label](./data/RQ1-RQ3-Bug-Detection/bug_detection_label.csv)



An executable demo can be found [here](https://github.com/SemMT-ICSE21/SemMT/blob/master/code_book/metrics_calculation.ipynb), it includes:

- A snippet of how similarity metrics are calculated over a pair of sentences
- Visualization of Accuracy, Precision, Recall, Fscore, Sensitivity and Specificity (RQ1)
- Venn figure of bugs detected by each similarity metrics (RQ3) 



## RQ2

In RQ2, the 200 initial seeds are in [Seeds](./data/RQ2-Existingwork-Comparison/Seeds/initial_seed_200.txt).

For the experimental result of each work, please see the following folder:

- [PatInv](./data/RQ2-Existingwork-Comparison/PatInv/)

- [SIT](./data/RQ2-Existingwork-Comparison/SIT/)

- [TransRepair](./data/RQ2-Existingwork-Comparison/TransRepair/)

- [SemMT](./data/RQ2-Existingwork-Comparison/SemMT/)

An executable demo can be found [here](https://github.com/SemMT-ICSE21/SemMT/blob/master/code_book/comparison_with_existing_work.ipynb), it includes:

- Read in bug report
- Visualization of comparisons
- Best result presentation



## Notes

### Preprocessing

- The sentences in the dataset are pre-process by the following steps:
  - Lower all the cases in the sentence
  - Replace the keywords as follows:
    - \<M0\> -> \'dog\'
    - \<M1\> -> \'truck\'
    - \<M2\> -> \'ring\'
    - \<M3\> -> \'lake\'

- The regular expressions generated are in raw forms. They can be transformed to normal regular expressions by the following replacing rules:

```python
regex = regex.replace("<VOW>", " ".join('AEIOUaeiou'))
regex = regex.replace("<NUM>", " ".join('0-9'))
regex = regex.replace("<LET>", " ".join('A-Za-z'))
regex = regex.replace("<CAP>", " ".join('A-Z'))
regex = regex.replace("<LOW>", " ".join('a-z'))

regex = regex.replace("<M0>", " ".join('dog'))
regex = regex.replace("<M1>", " ".join('truck'))
regex = regex.replace("<M2>", " ".join('ring'))
regex = regex.replace("<M3>", " ".join('lake'))
```


