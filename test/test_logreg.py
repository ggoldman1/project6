"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

def test_updates():
	"""

	"""
	# Check that your gradient is being calculated correctly
	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training

	X_train, X_test, y_train, y_test = utils.loadDataset(
		features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen',
				  -                          'Plain chest X-ray (procedure)', 'Low Density Lipoprotein Cholesterol',
				  -                          'Creatinine'], split_percent=0.8)
	lr = LogisticRegression(X_train.shape[1])
	lr.train_model(X_train, y_train, X_test, y_test)

	## Test 1: check that loss is generally decreasing over iterations
	halfway = int(len(lr.loss_train_history) / 2)
	assert np.mean(lr.loss_train_history[0:halfway]) > np.mean(lr.loss_train_history[halfway:halfway*2])

	## Test 2: 

def test_predict():
	# Check that self.W is being updated as expected
	# and produces reasonable estimates for NSCLC classification

	# Check accuracy of model after training

	pass