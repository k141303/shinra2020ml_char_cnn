# shinra2020ml_char_cnn
Character-level CNN for [SHINRA2020ML](http://shinra-project.info/shinra2020ml/howtoparticipate/?lang=en)(character-level CNN) task.  
All 30 languages are supported.

## Requirement

- Python3 (>=3.6)
- requirement.txt

## Preprocessing

Please specify the path to the directory containing the **training data**, such as `trial_en.zip` or `minimum_en.zip`.

~~~
$ bash script/run_preprocess.sh /Path/to/train_dir 
~~~

## Training

~~~
$ bash script/run_train.sh
~~~

## Get Prediction

Please specify the path to the directory containing the **target data**, such as `minimum_en.zip` or `leaderboard.zip`.

~~~
$ bash script/run_predict.sh /Path/to/target_dir
~~~
