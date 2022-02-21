"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

import regression
import numpy as np

def test_updates():
	"""

	"""
	# Check that your gradient is being calculated correctly
	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training


	X_train, X_test, y_train, y_test = regression.utils.loadDataset(
		features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen',
			      'Plain chest X-ray (procedure)', 'Low Density Lipoprotein Cholesterol',
                  'Creatinine'], split_percent=0.8)
	lr = regression.LogisticRegression(X_train.shape[1])
	lr.train_model(X_train, y_train, X_test, y_test)

	## Test 1: check that loss is generally decreasing over iterations
	halfway = int(len(lr.loss_history_train) / 2)
	assert np.mean(lr.loss_history_train[0:halfway]) > np.mean(lr.loss_history_train[halfway:halfway*2])

	## Test 2: check that loss is low (below 1)
	assert lr.loss_history_train[-1] < 1

	## Test 3: check that loss gets lower if we allow more iterations
	num_iter = [10, 100, 1000, 10000]
	prev = np.inf
	for iter in num_iter:
		lr = regression.LogisticRegression(X_train.shape[1], max_iter=iter)
		lr.train_model(X_train, y_train, X_test, y_test)
		assert lr.loss_history_train[-1] < prev
		prev = lr.loss_history_train[-1] < prev

def test_predict():
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

    X_train, X_test, y_train, y_test = regression.utils.loadDataset(
        features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen',
                  'Plain chest X-ray (procedure)', 'Low Density Lipoprotein Cholesterol',
                  'Creatinine'], split_percent=0.8)

    ## Test 1: check that weights update with more iterations 
    ## Test 2: check that more predictions are correct with more iterations
    num_iter = [1, 50]
    prevw = np.array([0, 0, 0, 0, 0, 0])
    prev_bce = np.inf
    
    X_test_wcons = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    for iter in num_iter:
        lr = regression.LogisticRegression(X_train.shape[1], max_iter=iter)
        lr.train_model(X_train, y_train, X_test, y_test)
        assert not np.all(np.equal(lr.W, prevw))
        prevw = lr.W

        bce = lr.loss_function(X_test_wcons, y_test)
        assert prev_bce > bce
        prev_bce = bce
        

