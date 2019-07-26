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

Make sure that in the python file, the boolean ```rand_tuning``` is set to ```True```.


