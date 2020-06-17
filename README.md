# Ordinary Least Squares (OLS) Regression

This package is a simple project to showcase an implementation of the OLS
regression algorithm. In this particular implementation the following occurs:

	1. input independent and dependent variable data is split into 
	   "training" and "prediction" sets. 
	2. Within the training set, k-fold crossvalidation is used to 
	   generate an Akaike Information Criteria (AIC) value for each 1-p 
	   combinations of independent variables. 
	3. The model with the lowest AIC is selected and fit to the entire 
	   training set.
	4. The newly fit champion model is used to predict values using the 
	   independent variables of the "prediction" set, withheld in (1).
	5. The predicted values are compared against the actual values of the
	   dependent variable withheld in (1), and common regression 
	   statistics are generated and stored in an Excel File, named 
	   "prediction.xlsx"

The main script in the packages is "ols_main.py". This script's main function
loads Boston housing price data from Python's sklearn package as example data.
The results of the regression are stored in the "output" directory.

# Installation

"ols_main.py" is dependent on the following python packages:

	Package   | Function
	--------------------------------------------------------
	NumPy     | Matrix calculations.
	Pandas    | Data storage and reporting.
	random    | Random rearrangement of data.
	sklearn   | Example "toy" datasets for testing.
	itertools | Quickly create list of possible 1-p 
		  | combinations of independent variables. 
	math      | "Ceiling" and "Floor" rounding functions.
	time	  | Track execution time.
	scipy	  | "f" and "t" to determine significance of
		  | the f_statistics and p_values, respectively 
	--------------------------------------------------------

Lines 8-17 of "ols_main.py" import these packages. If there is an issue with
importing these libraries, simply use the "pip" or "conda" command from the
Python or Anaconda shell, respectively, to properly install the libraries.

# Usage

"ols_main.py" can be run as is using the preferred IDE. The user may make edits
to the main function to run the algorithm on different data sets, so long as
the new dataset meets the following conditions:

	1. Dependent variables are stored in a n-by-p NumpyArray of floats,
	   where n is the number of observations and p is the number of
	   independent variables.
	2. Independent variable is stored in a n-by-1 NumpyArray of floats.

If a new dataset is chosen, line 54 of "ols_main.py" is no longer necessary,
and the user may specify their new data in lines 55 and 56. From there, the
script can be run as per usual. If the user would like to rename the output 
file at runtime, they may change line 19 to reflect the new name of the
output xlsx file.

# Changelog

	Version | Date       | Notes
	------------------------------------------------------------
	1.0.0   | 06-16-2020 | First end-to-end implementation of 
			       the OLS Regression Algorithm for this
			       project 
	-------------------------------------------------------------

# Planned Improvements

Process Flow

	1. Running the procedure on novel data requires an understanding of
	   Python and code. In the future, a user interface could be
	   implemented to ease the application of the procedure to new data.
	2. Currently, champion model selection by AIC is hardcoded in the 
	   procedure. The user should be able to choose the selection 
	   criteria by which their champion model is selected. 

Efficiency
	
	1. A Q-R decomposition is a more efficient method of deriving OLS 
	   coefficients and could be used in lieu of standard matrix 
	   multiplication.
	2. The current procedure uses a dataset with p = 13 regressors.
	   Sluething through the resulting model combinations to find the
	   champion model currently takes ~23 seconds. As p increases, the
	   amount of models to investigate would increase exponentially.
	   The native "multiprocessor" library could disperse the k-fold
	   crossvalidation among several computing cores in parallel, and
	   reduce runtime significantly.
	3. Currently, a simple bubble sort (O(n^2)) is used to identify the 
	   model with the lowest AIC. An optimized bubble sort (slightly 
	   better) or a merge sort (O(n*log n) should be implemented instead
	   to identify the champion model more efficiently.
	4. The methodology used to assign regressor names to their indexes is
	   redundant. A more efficient method of keeping track of these names
	   should be implemented.

# Authors

Brandon C Lyman

# Other Notes

An example of the "prediction.xlsx" file can be found in the "examples" 
directory.
