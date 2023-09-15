import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data = pd.read_csv('bank1.csv')
# data=data.drop(['contact'],axis=1)
# data=data.drop(['month'],axis=1)
# data=data.drop(['poutcome'],axis=1)
# data=data.drop(['job'],axis=1)
# data=data.drop(['education'],axis=1)
# data['marital']=data['marital'].replace({'married':'2','single':'1','divorced':'0'})
# data['default']=data['default'].replace({'no':'0','yes':'1'})
# data['loan']=data['loan'].replace({'no':'0','yes':'1'})
# data['deposit']=data['deposit'].replace({'no':'0','yes':'1'})
# data['housing']=data['housing'].replace({'no':'0','yes':'1'})
#
# X = data[['age','default','marital','balance','housing','day','duration','campaign','pdays','previous','deposit']]
# Y = data.loan

data = pd.read_csv('dataset.csv')
X=data[['X']]
Y=data.Y
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
logreg = LogisticRegression()
logreg.fit(X_Train,Y_Train)
y_pred=logreg.predict(X_Test)

cnf_matrix = metrics.confusion_matrix(Y_Test, y_pred)
cnf_matrix

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(Y_Test, y_pred))
print("Precision:",metrics.precision_score(Y_Test, y_pred,average='micro'))
print("Recall:",metrics.recall_score(Y_Test, y_pred,average='micro'))

y_pred_proba = logreg.predict_proba(X_Test)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_Test,  y_pred_proba,pos_label='your_label')
auc = metrics.roc_auc_score(Y_Test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()