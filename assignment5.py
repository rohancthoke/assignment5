import pandas as pd
import seaborn as sns
df=pd.read_csv("Social_Network_Ads.csv")
shape=df.shape
print(shape)
sns.heatmap(df.corr(),annot=True)
X=df[['Age']]
Y=df[['Purchased']]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random__state=42)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)
print('Model Score:'model.score(X_test,Y_test))
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
y_pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cf_matrix=confusion_matrix(Y_test,y_pred)
print(cf_matrix)
from sklearn.metrics import precision_recall_fscore_support
score=precision_recall_fscore_support(Y_test,y_pred,average='micro')
print('precision:',score[0])
print('recall:',score[1])
print('f-score:',score[2])