Steps to execute the program 
1. Open command prompt/Terminal and navigate to the folder where the *.py script is kept.
2. Execute the following, replacing the paths to CSVs in 3rd, 4rth, 5th argument, and whether to print the decision tree or not in the 6th argument.
$ python DecisionTree.py  <L> <K> <training-set> <validation-set> <test-set> <to-print>

Examples for reference:
To output without printing trees, with L and K both 10:
$ python DecisionTree.py 10 10 "D://MS//training_set.csv" "D://MS//validation_set.csv" "D://MS//test_set.csv" "no"

To print the tree (with L and K both set to 10) command is 
$ python DecisionTree.py 10 10 "D://MS//training_set.csv" "D://MS//validation_set.csv" "D://MS//test_set.csv" "yes"

3. please note for path - "//" is being used to specify the subfolders
