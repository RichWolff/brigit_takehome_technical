import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def ecdf(data):
	x = np.sort(data)
	n = len(x)
	y = np.arange(1,n+1)/n
	return ( x, y )

def roc_auc_plot(x_test,y_test,y_pred,model,ax,label='Logistic Regression'):
	logit_roc_auc = roc_auc_score(y_test, model.predict(x_test))
	fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
	ax.plot(fpr, tpr, label=label+'(area = %0.2f)' % logit_roc_auc)

	ax.plot([0, 1], [0, 1],'r--')
	ax.set_xlim([0.0, 1.0])
	ax.set_ylim([0.0, 1.05])
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title('Receiver operating characteristic')
	ax.legend(loc="lower right")
	return ax