#Aayush Atul Verma 2017A7PS0061P
#Jithin Kalluakalam Sojan 2017A7PS0163P
#Agastya Sampath 2017A3PS0359P

# In[58]:


# All imports
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import svm

from modAL.uncertainty import uncertainty_sampling, classifier_uncertainty, classifier_margin, classifier_entropy
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import vote_entropy, max_disagreement_sampling as kl, entropy


# In[59]:


iris = load_iris()

print("The features of the dataset:", iris.feature_names)
print("The labels of the dataset:" , iris.target_names)
X = iris['data']
Y = iris['target']


# In[60]:


pca = PCA(n_components= 2)
reduced_PCA = pca.fit_transform(X)


# In[61]:


x_axis, y_axis = reduced_PCA[:, 0], reduced_PCA[:, 1]

plt.figure(figsize=(5.5, 3.5), dpi=100)
plt.scatter(x=x_axis, y=y_axis, c=Y, cmap='cividis', s=20, alpha=2)
plt.title('Dataset plot after PCA')
plt.show()


# In[62]:


random.seed(1)
unlabelled_idx = random.sample(range(0, X.shape[0]), 135)

X_unlab = np.asarray(X[unlabelled_idx])
Y_unlab = np.asarray(Y[unlabelled_idx])
X_unlab.reshape(X_unlab.shape[0], -1)
Y_unlab.reshape(Y_unlab.shape[0],)


X_train = np.delete(np.array(X), np.array(unlabelled_idx), axis= 0)
Y_train = np.delete(Y, unlabelled_idx, axis= 0)

X_train.reshape(15,-1)
Y_train.reshape(15,)


# In[63]:


print(unlabelled_idx)


# In[64]:


random.seed(5)
# strategy = vote_entropy
strategy = kl


member1 = ActiveLearner(X_training=X_train, y_training=Y_train, estimator= RandomForestClassifier(n_estimators=8), query_strategy= strategy)
member2 = ActiveLearner(X_training=X_train, y_training=Y_train, estimator= RandomForestClassifier(n_estimators= 1), query_strategy= strategy)
member3 = ActiveLearner(X_training=X_train, y_training=Y_train, estimator= RandomForestClassifier(n_estimators=10), query_strategy= strategy)
member4 = ActiveLearner(X_training=X_train, y_training=Y_train, estimator= KNeighborsClassifier(n_neighbors=8), query_strategy= strategy)
member5 = ActiveLearner(X_training=X_train, y_training=Y_train, estimator= KNeighborsClassifier(n_neighbors= 10), query_strategy= strategy)

committee = Committee(learner_list= [member1, member2, member3, member4, member5])


# In[65]:


import math
unlab_length = X_unlab.shape[0]
disagreement = np.zeros(unlab_length*2).reshape(unlab_length,2)

for i in range(unlab_length):
    index = [i]
    predict = [-1,-1,-1,-1,-1]
    predict[0] = member1.predict(X_unlab[index])[0]
    predict[1] = member2.predict(X_unlab[index])[0]
    predict[2] = member3.predict(X_unlab[index])[0]
    predict[3] = member4.predict(X_unlab[index])[0]
    predict[4] = member5.predict(X_unlab[index])[0]
    
    if not(predict[0]==predict[1]==predict[2]==predict[3]==predict[4]):
        disagreement[i][0] = 1
        count = [0,0,0]
        for j in range(5):
            count[predict[j]]+=1
        for j in range(3):
            if(count[j]):
                disagreement[i][1]-=(count[j]/5)*math.log(count[j]/5)

version_space = disagreement[disagreement[:,0]==1]
version_space_points = X_unlab[disagreement[:,0]==1]
version_space_labels = Y_unlab[disagreement[:,0]==1]
version_space
print('Size of version space: {size:d} points.'.format(size = version_space.shape[0]))


# In[66]:


version_space_length = version_space.shape[0]
ind = np.argsort(version_space[:,1])

print('Order of points to label:')

for i in range(version_space_length):
    print('#{size:2d}'.format(size=i)+' point:'+str(version_space_points[ind[i]])+' label:'+str(version_space_labels[ind[i]]))


# In[67]:


print("Initial accuracy =", committee.score(X, Y))


# In[68]:


x = 40

queries = int((x/100) * 150)
accuracy_list = []
accuracy_list.append(committee.score(X,Y))


# In[69]:


iter = 0
print("Accuracy after", 0, "iterations :", committee.score(X,Y))

for i in range(0,queries):
    
    for x in range(135):
        if(classifier_uncertainty(committee, X_unlab[iter].reshape(1,-1))>= 0.8):
            break
        iter= (iter+1)%(X_unlab.shape[0])
    
    q_id = iter-1

    X_new = X_unlab[q_id].reshape(1,-1)
    Y_new = Y_unlab[q_id].reshape(1,)
    
    X_unlab, Y_unlab = np.asarray(np.delete(X_unlab, q_id, axis= 0)), np.delete(Y_unlab, q_id, axis= 0)
    committee.teach(X_new, Y_new)

    x = committee.score(X, Y)
    accuracy_list.append(x)
    
    if((i+1)%15 == 0):
        print("Accuracy after", i+1, "iterations :", x)

    
print("Number of unlabelled data points left are : ",X_unlab.shape[0])


# In[70]:


plt.figure(figsize=(7,5))
plt.plot(list(range(0,int(queries)+1)), accuracy_list, color= 'green', linewidth= 1 )
plt.xlabel('Query iteration')
plt.ylabel('Classification Accuracy')
plt.title('Graph of iteration-wise accuracy')

plt.ylim(.60,1.00)
plt.xlim(0,queries)

plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




