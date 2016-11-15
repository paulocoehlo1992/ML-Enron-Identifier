import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',"bonus","exercised_stock_options","salary","total_stock_value","wealth","deferred_income"] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
import pandas
df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))
df.set_index(employees, inplace=True)
df.replace('NaN', 0, inplace = True)

### Task 2: Remove outliers
df.drop(df.index[104],inplace=True)

### Task 3: Create new feature(s)
df["wealth"]=df["bonus"] +df["total_stock_value"] +df["salary"] +df["total_payments"] +df["long_term_incentive"]
x=[]
for i in range(0,len(df["from_messages"])):
    if df["from_messages"][i]==0:
        x.append(0)
    else:
        x.append(df["from_poi_to_this_person"][i]/(df["from_messages"][i]))
        
df["fraction_from_poi"]=pandas.Series(x,index=df.index)
y=[]
for i in range(0,len(df["to_messages"])):
    if df["to_messages"][i]==0:
        y.append(0)
    else:
        y.append(df["from_this_person_to_poi"][i]/float(df["to_messages"][i]))
df["fraction_to_poi"]=pandas.Series(y,index=df.index)

data_dict = df.to_dict('index')
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score,recall_score
clf=GaussianNB()
clf.fit(features,labels)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train,Y_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
#Trying Decisiontree Classifier,comment out to run this classifer, before comment above selected GaussianNB classifer

#from sklearn.tree import DecisionTreeClassifier
#params_dt={"criterion":["gini","entropy"],"min_samples_split":[2,3,4,5,6,7,8,10,12,13,15],"min_samples_leaf":[1,2,3,4,5,6]}
#dt=DecisionTreeClassifier()
#clf=GridSearchCV(dt,params_dt)
#clf.fit(X_train,Y_train)
#print(clf.best_params_)
#pred_dt=clf.predict(X_test)
#print("precision score:",precision_score(pred_dt,Y_test))
#print("Recall score:",recall_score(pred_dt,Y_test))

#Trying KNN Classifier

#from sklearn.neighbors import KNeighborsClassifier
#clf=KNeighborsClassifier(n_neighbors=3)
#clf.fit(X_train,Y_train)
#pred_knn=clf.predict(X_test)
#print("precision score:",precision_score(pred_knn,Y_test))
#print("Recall score:",recall_score(pred_knn,Y_test))

#Trying SVC

#from sklearn.svm import SVC
#svc=SVC()
#params_sv={"C":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],"kernel":["linear","poly","rbf","sigmoid"],"gamma":[1,2,3,4,5,6,7,8,9,10]}
#clf=GridSearchCV(svc,params_sv)
#clf.fit(X_train,Y_train)
#print(clf.best_params_)
#pred_svc=clf.predict(X_test)
#print("Precision score:",precision_score(pred_svc,Y_test))
#print("Recall score:",recall_score(pred_svc,Y_test))






### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)