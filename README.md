# shinra2020ml_char_cnn
Character-level CNN for [SHINRA2020ML](character-level CNN) task.

# Requirement

- Python3 (>=3.6)

# Preprocessing

Please specify the path to the directory containing the training data, such as `trial_en.zip` or `minimum_en.zip`.

~~~
$ bash script/run_preprocess.sh /Path/to/train_dir 
~~~

# Training

~~~
$ bash script/run_train.sh
~~~

# Get Prediction

Please specify the path to the directory containing the target data, such as `minimum_en.zip` or `leaderboard.zip`.

~~~
$ bash script/run_predict.sh /Path/to/target_dir
~~~
