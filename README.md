# CS 229 (Machine Learning, Spring 2019) Final Project.

This is the repository for my final project for CS 229. It is joint with Justin Lundgren. 

# Paper
Our final project is available at http://cs229.stanford.edu/proj2019/report/31.pdf.

# Setup
Make sure that you have Tensorflow 1.13 installed instead of Tensorflow 2.0. We use Keras for our code with a tensorflow backend. Our version of Keras (2.2.4) is not compatible with Tensorflow 2.0.

# Hyperparameter Tuning
Here's something to play around with, namely hyperparameter tuning via a random grid search. First clone the repository, then go to src/LSTM
and run

```
python hyp_tuning.py. 
```

Make sure that in the python file, the boolean ```rand_tuning``` is set to ```True```. Now when you run ```hyp_tuning.py```, it randomly selects the hyperparameters

* Number of LSTM units
* Lookback period
* Learning rate

and saves the MSE value obtained on the validation set of our data into a csv file. In the file ```hyp_tuning.py```, on about line 140, the quantity ```num_trials``` is the number of times we want python to generate a random choice of these parameters. 

The CSV file that is generated is saved in src/output/LSTM_tuning/random_samples. If you set  ```num_trials``` to be ```4```, then the CSV file will have 4 rows. 

