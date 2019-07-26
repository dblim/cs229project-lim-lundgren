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

and saves the MSE value obtained on the validation set of our data into a CSV file. In the file ```hyp_tuning.py```, on about line 140, the quantity ```num_trials``` is the number of times we want python to generate a random choice of these parameters. We save the CSV file in  src/output/LSTM_tuning/random_samples. To distinguish the  different files each time we execute ```python hyp_tuning.py```, the CSV files contain a random number in their name.

Finally, to obtain Fig. 2 in our CS 229 project linked above, go to src/output/LSTM_training/ and run the jupyter notebook ```Tuning_plot.ipynb```. Run all the code in their, and the figure generated will be a plot of hyperparameters against MSE value.

