import pandas as pd
import os

## constants

# Get directories
base_dir = os.path.abspath(os.path.dirname('__file__'))
data_dir = os.path.join(base_dir,'data')
raw_data_dir =  os.path.join(data_dir,'raw')

def main():
	df = load_data()
	df = dataImputer(df)
	df = dataTransform(df)
	df.to_csv(os.path.join(data_dir,'processed/loan_tape_processed.csv'))
	return 0



def load_data():
	# Import the raw data
	raw_data = os.path.join(raw_data_dir,'loan_tape.csv')
	raw_df = pd.read_csv(raw_data,index_col=0)
	raw_df.set_index('id', inplace = True)
	return raw_df

def dataImputer(X):
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


	X['credittodebitratiomean'] = X.groupby('user_id')['credittodebitratiomean'].transform(lambda x: x.fillna(x.mean()))
	X['credittodebitratiomean'] = X['credittodebitratiomean'].fillna(X['credittodebitratiomean'].mean())

	X['daystopayday'] = X.groupby('user_id')['daystopayday'].transform(lambda x: x.fillna(x.mean()))
	X['daystopayday'] = X['daystopayday'].fillna(X['daystopayday'].mean())

	X['monthswithfeesrate'] = X.groupby('user_id')['monthswithfeesrate'].transform(lambda x: x.fillna(x.mean()))
	X['monthswithfeesrate'] = X['monthswithfeesrate'].fillna(X['monthswithfeesrate'].mean())
		#returns numpy array
	return X 

def dataTransform(X):
	X.drop(['recurrentamountsum'],axis=1,inplace=True)
	X['balancemeanCombined'] = X[['balancemean','balancemeanafterpayday0','balancemeanafterpayday1']].mean(axis=1)
	X.drop(['balancemean','balancemeanafterpayday0','balancemeanafterpayday1'],axis=1,inplace=True)
		
	#Transform Categorical data
	X = X.join(pd.get_dummies(X['highestpayfrequency'],prefix='payFreq_'))
	X = X.join(pd.get_dummies(X['signup_source'],prefix='signUpSource_'))
	X = X.join(pd.get_dummies(X['location_state'],prefix='signUpState_'))
	X.drop(['highestpayfrequency','signup_source','location_state'],axis=1, inplace=True)
	X.drop(['user_id'],axis=1, inplace=True)
	return X

if __name__ == '__main__':
	main()