from sklearn.base import BaseEstimator, TransformerMixin
class CustomImputer( BaseEstimator, TransformerMixin ):

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

		#Impute categorical data
		X['location_state'] = X['location_state'].fillna('Unknown')
		X['signup_source'] = X['signup_source'].fillna('Unknown')
		X['highestpayfrequency'] = X.groupby('user_id')['highestpayfrequency'].fillna(method = 'bfill')
		X['highestpayfrequency'] = X.groupby('user_id')['highestpayfrequency'].fillna(method = 'Ffill')
		X['highestpayfrequency'] = X.groupby('user_id')['highestpayfrequency'].fillna('Unknown')

		#Impute users with ls than 50% of data entered
		drop_columns_that_are_auto_populated = ['user_id','default_flag','amount']
		numerator = ((X==0) | (X=='Unknown') | (X.isna())).sum(axis=1)
		denominator = X.drop(drop_columns_that_are_auto_populated,axis=1).fillna(0).count(axis=1)
		pct_missing = numerator.divide(denominator).sort_values(ascending=False)
		X.loc[pct_missing > .5] = X.loc[pct_missing > .5].fillna(0)


		X['credittodebitratiomean'] = X.groupby('user_id')['credittodebitratiomean'].transform(lambda x: x.fillna(x.mean())).isna()
		X['credittodebitratiomean'] = X['credittodebitratiomean'].fillna(X['credittodebitratiomean'].mean())

		X['daystopayday'] = X.groupby('user_id')['daystopayday'].transform(lambda x: x.fillna(x.mean())).isna()
		X['daystopayday'] = X['daystopayday'].fillna(X['daystopayday'].mean())

		X['monthswithfeesrate'] = X.groupby('user_id')['monthswithfeesrate'].transform(lambda x: x.fillna(x.mean())).isna()
		X['monthswithfeesrate'] = X['monthswithfeesrate'].fillna(X['monthswithfeesrate'].mean())
			#returns numpy array
		return X.values 