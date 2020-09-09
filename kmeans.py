#Aayush Atul Verma 2017A7PS0061P
#Jithin Kalluakalam Sojan 2017A7PS0163P
#Agastya Sampath 2017A3PS0359P

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris


#Labeling points based on K-Means Clustering

iris_dataset = load_iris()
data = iris_dataset['data']
target = iris_dataset['target']

#Selecting the initial 10% training points.

no_samples = data.shape[0]
test_index = np.random.choice(no_samples,135,replace=False)
train_index = np.delete(np.arange(no_samples), test_index)

train_points = data[train_index]
train_points_labels = target[train_index]
test_points = data[test_index]
test_points_labels = target[test_index]


#Selecting 40% of the unlabeled points for K-Means clustering and running K-Means.


n_kmeans_samples = test_points.shape[0]
clustering_indices = np.random.choice(n_kmeans_samples,54,replace=False)
non_clustering_indices = np.delete(np.arange(n_kmeans_samples),clustering_indices)
clustering_points = test_points[clustering_indices]
clustering_points_labels = test_points_labels[clustering_indices]

kmeans = KMeans(n_clusters = 3, random_state=0).fit(clustering_points)
cluster1 = clustering_points[kmeans.labels_==0]
cluster1_labels = clustering_points_labels[kmeans.labels_==0]
cluster2 = clustering_points[kmeans.labels_==1]
cluster2_labels = clustering_points_labels[kmeans.labels_==1]
cluster3 = clustering_points[kmeans.labels_==2]
cluster3_labels = clustering_points_labels[kmeans.labels_==2]


# Selecting 20% representative points from each cluster


no_labels_cl1 = int(cluster1.shape[0]*0.2)
no_labels_cl2 = int(cluster2.shape[0]*0.2)
no_labels_cl3 = int(cluster3.shape[0]*0.2)
rep1_index = np.random.choice(cluster1.shape[0], no_labels_cl1, replace=False)
rep2_index = np.random.choice(cluster2.shape[0], no_labels_cl2, replace=False)
rep3_index = np.random.choice(cluster3.shape[0], no_labels_cl3, replace=False)


#Printing the representative points from each cluster.

print('\n')
print("Representative points of each cluster:")
print("Cluster 1")
print(cluster1[rep1_index])
print("Cluster 2")
print(cluster2[rep2_index])
print("Cluster 3")
print(cluster3[rep3_index])


# Calculating the labels of entire cluster based on majority voting between representative points of the cluster.


rep1_labels = cluster1_labels[rep1_index]
rep2_labels = cluster2_labels[rep2_index]
rep3_labels = cluster3_labels[rep3_index]

count = [0,0,0]
for i in range(no_labels_cl1):
    count[rep1_labels[i]]+=1
maxLabel = 0
maxCount = 0
for j in range(3):
    if maxCount<count[j]:
        maxCount=count[j]
        maxLabel = j
rep_cluster1_label = maxLabel

count = [0,0,0]
for i in range(no_labels_cl2):
    count[rep2_labels[i]]+=1
maxLabel = 0
maxCount = 0
for j in range(3):
    if maxCount<count[j]:
        maxCount=count[j]
        maxLabel = j
rep_cluster2_label = maxLabel

count = [0,0,0]
for i in range(no_labels_cl3):
    count[rep3_labels[i]]+=1
maxLabel = 0
maxCount = 0
for j in range(3):
    if maxCount<count[j]:
        maxCount=count[j]
        maxLabel = j
rep_cluster3_label = maxLabel


#Printing the labels for the entire clusters.


print('\n')
print("Calculated Labels of the Clusters:")
print("Label for Cluster #1: "+ str(rep_cluster1_label))
print("Label for Cluster #2: "+ str(rep_cluster2_label))
print("Label for Cluster #3: "+ str(rep_cluster3_label))


# Calculating accuracy of the labelling.


cluster_repd_labels = np.arange(clustering_points.shape[0])
cluster_repd_labels[kmeans.labels_==0] = rep_cluster1_label
cluster_repd_labels[kmeans.labels_==1] = rep_cluster2_label
cluster_repd_labels[kmeans.labels_==2] = rep_cluster3_label

cluster_repd_points_correct = clustering_points[cluster_repd_labels==clustering_points_labels]

accuracy = cluster_repd_points_correct.shape[0]/clustering_points.shape[0]
print('\n')
print('The cluster labelling has accuracy of '+str(accuracy))


#Calculating amount and time saved. 


initial_no_training = clustering_points.shape[0]
final_no_training = no_labels_cl1 + no_labels_cl2 + no_labels_cl3
savings = 100*(initial_no_training-final_no_training)
hours = (initial_no_training-final_no_training)

print('\n')
print('The amount saved is: Rs.'+str(savings))
print('The hours saved are: '+str(hours))






