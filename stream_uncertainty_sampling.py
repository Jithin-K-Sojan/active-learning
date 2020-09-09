#Aayush Atul Verma 2017A7PS0061P
#Jithin Kalluakalam Sojan 2017A7PS0163P
#Agastya Sampath 2017A3PS0359P

# In[25]:


# All imports

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling, classifier_uncertainty, classifier_margin, classifier_entropy

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.markers as mk
import numpy as np
import random
from modAL.models import ActiveLearner


# In[26]:


iris = load_iris()

print("The features of the dataset:", iris.feature_names)
print("The labels of the dataset:" , iris.target_names)
X = iris['data']
Y = iris['target']


# In[27]:


pca = PCA(n_components= 2)
reduced_PCA = pca.fit_transform(X)


# In[28]:


x_axis, y_axis = reduced_PCA[:, 0], reduced_PCA[:, 1]

plt.figure(figsize=(5.5, 3.5), dpi=100)
plt.scatter(x=x_axis, y=y_axis, c=Y, cmap='cividis', s=20, alpha=2)
plt.title('Dataset plot after PCA')
plt.show()


# In[29]:


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


# In[30]:


print("Number of unlabelled data points are : ",X_unlab.shape[0])


# In[31]:


print(np.array(unlabelled_idx))


# In[32]:


random.seed(21)
# strategy = uncertainty_sampling
# xyz = classifier_uncertainty

# strategy = margin_sampling
# xyz = classifier_margin

strategy = entropy_sampling
xyz = classifier_entropy


knn = KNeighborsClassifier(n_neighbors=5
                           , weights='distance')
al_model = ActiveLearner(query_strategy= strategy, X_training=X_train, y_training=Y_train, 
                        estimator= knn )


# In[33]:


print("Initial accuracy =", al_model.score(X, Y))


# In[34]:


x = 40

queries = int((x/100) * 150)

accuracy_list = []
accuracy_list.append(al_model.score(X,Y))


# In[35]:


iter = 0

print("Accuracy after", 0, "iterations :", al_model.score(X,Y))
for i in range(0,queries):
    
    for x in range(135):
        
        if(strategy == margin_sampling):
            decision = xyz(al_model, X_unlab[iter].reshape(1,-1))<= 0.5
        else:
            decision = xyz(al_model, X_unlab[iter].reshape(1,-1))>= 0.5
            
        if(decision):
            break
        iter= (iter+1)%(X_unlab.shape[0])
    
    query_idx = iter - 1
    X_new = X[unlabelled_idx[query_idx]].reshape(1,-1)
    Y_new = Y[unlabelled_idx[query_idx]].reshape(1,)
    
    X_unlab, Y_unlab = np.asarray(np.delete(X_unlab, query_idx, axis= 0)), np.delete(Y_unlab, query_idx, axis= 0)
    al_model.teach(X_new, Y_new)
    x = al_model.score(X, Y)
    accuracy_list.append(x)

    if((i+1)%15 == 0):
        print("Accuracy after", i+1, "iterations :", x)
    
print("Number of unlabelled data points left are :",X_unlab.shape[0])


# In[36]:


plt.figure(figsize=(6,4.5))
markers_on = [0, 15, 30, 45, 60]
plt.plot(list(range(0,int(queries)+1)), accuracy_list, "-.o", markevery=markers_on , color= 'red', linewidth= 1 ,)
plt.xlabel('Query iteration')
plt.ylabel('Classification Accuracy')
plt.title('Graph of iteration-wise accuracy')
# mk.
plt.ylim(.60,1.00)
plt.xlim(0,queries)

plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




