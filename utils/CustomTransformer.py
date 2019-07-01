from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
class CustomTransformer( BaseEstimator, TransformerMixin ):

	#Class constructor method that takes in a list of values as its argument
	def __init__(self, ):
		pass

	#Return self nothing else to do here
	def fit( self, X, y = None  ):
		return self

	#Transformer method we wrote for this transformer 
	def transform(self, X , y = None ):
		#Depending on constructor argument break dates column into specified units
		#using the helper functions written above 

		#Transform numeric data
		X.drop(['recurrentamountsum'],axis=1,inplace=True)
		X['balancemeanCombined'] = X[['balancemean','balancemeanafterpayday0','balancemeanafterpayday1']].mean()
		X.drop(['balancemean','balancemeanafterpayday0','balancemeanafterpayday1'],axis=1,inplace=True)
			
		#Transform Categorical data
		X = X.join(pd.get_dummies(X['highestpayfrequency'],prefix='payFreq_'))
		X = X.join(pd.get_dummies(X['signup_source'],prefix='signUpSource_'))
		X = X.join(pd.get_dummies(X['location_state'],prefix='signUpState_'))
		X.drop(['highestpayfrequency','signup_source','location_state'],axis=1, inplace=True)
		X.drop(['user_id'],axis=1, inplace=True)
		#returns numpy array
		return X.values 