#!/usr/bin/env python
# coding: utf-8

# In[547]:


import numpy as np
import matplotlib.pyplot as plt
X = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])
mu1 = [6.2,3.2]
mu2 = [6.6,3.7]
mu3 = [6.5,3.0]
def compute_Kmeans(X,mu1,mu2,mu3):
    X_array = np.tile(X,3)
    my_dict = {}
    centres_ini = np.array([mu1[0],mu1[1],mu2[0],mu2[1],mu3[0],mu3[1]])
    centres_array = np.tile(centres_ini,10).reshape(10,6)
    Y = np.square(X_array - centres_array)
    distance_array = np.array([[Y[:,0]+Y[:,1],Y[:,2]+Y[:,3],Y[:,4]+Y[:,5]]]).T.reshape(10,3)
    min_distance = np.argmin(distance_array, axis=1)
    req_shape = X.shape[0]
    for i in range(req_shape):
        my_dict.update({(X[i,0],X[i,1]):min_distance[i]})
    return my_dict    
    
first_centers = compute_Kmeans(X,mu1,mu2,mu3)
zero_1_class = []
one_1_class = []
two_1_class = []
zero_1_label = []
one_1_label = []
two_1_label = []
for key,value in first_centers.items():
    if value == 0:
        zero_1_class.append(key)
        zero_1_label.append(value)
    if value == 1:
        one_1_class.append(key)
        one_1_label.append(value)
    if value == 2:
        two_1_class.append(key)
        two_1_label.append(value)
    
def update_mu(class_list):
    x = 0
    y = 0
    for i in range(len(class_list)):
        x = x + class_list[i][0]
        y = y + class_list[i][1]
    return (x/len(class_list),y/(len(class_list)))
updated_1_mu1 = update_mu(zero_1_class)
updated_1_mu2 = update_mu(one_1_class)
updated_1_mu3 = update_mu(two_1_class)

second_centers = compute_Kmeans(X,updated_1_mu1,updated_1_mu2,updated_1_mu3)
zero_2_class = []
one_2_class = []
two_2_class = []
zero_2_label = []
one_2_label = []
two_2_label = []
for key,value in second_centers.items():
    if value == 0:
        zero_2_class.append(key)
        zero_2_label.append(value)
    if value == 1:
        one_2_class.append(key)
        one_2_label.append(value)
    if value == 2:
        two_2_class.append(key)
        two_2_label.append(value)
        
updated_2_mu1 = update_mu(zero_2_class)
updated_2_mu2 = update_mu(one_2_class)
updated_2_mu3 = update_mu(two_2_class)   

plt.figure(figsize=(10,5))
for i in range(len(zero_1_class)):
    plt.scatter(zero_1_class[i][0], zero_1_class[i][1],marker='^', c= 'red' )
for i in range(len(one_1_class)):
    plt.scatter(one_1_class[i][0], one_1_class[i][1],marker='^', c= 'green' )
for i in range(len(two_1_class)):
    plt.scatter(two_1_class[i][0], two_1_class[i][1],marker='^', c= 'blue' )
plt.scatter(mu1[0], mu1[1], c= 'red' )
plt.scatter(mu2[0], mu2[1], c= 'green' )
plt.scatter(mu3[0], mu3[1], c= 'blue' )

for i in range(len(zero_1_class)):
    plt.text(zero_1_class[i][0], zero_1_class[i][1],(zero_1_class[i][0],zero_1_class[i][1]) )
for i in range(len(one_1_class)):
    plt.text(one_1_class[i][0], one_1_class[i][1],(one_1_class[i][0], one_1_class[i][1]) )
for i in range(len(two_1_class)):
    plt.text(two_1_class[i][0], two_1_class[i][1],(two_1_class[i][0], two_1_class[i][1]) )
plt.text(mu1[0], mu1[1], (mu1[0], mu1[1]) )
plt.text(mu2[0], mu2[1], (mu2[0], mu2[1]) )
plt.text(mu3[0], mu3[1], (mu3[0], mu3[1]) )



plt.savefig('task2_iter1_a.jpg')  

plt.figure(figsize=(10,5))
for i in range(len(zero_1_class)):
    plt.scatter(zero_1_class[i][0], zero_1_class[i][1],marker='^',edgecolor = 'black', c = 'None' )
for i in range(len(one_1_class)):
    plt.scatter(one_1_class[i][0], one_1_class[i][1],marker='^',edgecolor = 'black', c = 'None')
for i in range(len(two_1_class)):
    plt.scatter(two_1_class[i][0], two_1_class[i][1],marker='^',edgecolor = 'black', c = 'None' )
plt.scatter(updated_1_mu1[0], updated_1_mu1[1], c= 'red' )
plt.scatter(updated_1_mu2[0], updated_1_mu2[1], c= 'green' )
plt.scatter(updated_1_mu3[0], updated_1_mu3[1], c= 'blue' )

for i in range(len(zero_1_class)):
    plt.text(zero_1_class[i][0], zero_1_class[i][1],(zero_1_class[i][0],zero_1_class[i][1]) )
for i in range(len(one_1_class)):
    plt.text(one_1_class[i][0], one_1_class[i][1],(one_1_class[i][0], one_1_class[i][1]) )
for i in range(len(two_1_class)):
    plt.text(two_1_class[i][0], two_1_class[i][1],(two_1_class[i][0], two_1_class[i][1]) )
plt.text(updated_1_mu1[0], updated_1_mu1[1], (updated_1_mu1[0], updated_1_mu1[1]) )
plt.text(updated_1_mu2[0], updated_1_mu2[1], (updated_1_mu2[0], updated_1_mu2[1]) )
plt.text(updated_1_mu3[0], updated_1_mu3[1], (updated_1_mu3[0], updated_1_mu3[1]) )



plt.savefig('task2_iter1_b.jpg')

plt.figure(figsize=(10,5))
for i in range(len(zero_2_class)):
    plt.scatter(zero_2_class[i][0], zero_2_class[i][1],marker='^', c= 'red' )
for i in range(len(one_2_class)):
    plt.scatter(one_2_class[i][0], one_2_class[i][1],marker='^', c= 'green' )
for i in range(len(two_2_class)):
    plt.scatter(two_2_class[i][0], two_2_class[i][1],marker='^', c= 'blue' )
plt.scatter(updated_1_mu1[0], updated_1_mu1[1], c= 'red' )
plt.scatter(updated_1_mu2[0], updated_1_mu2[1], c= 'green' )
plt.scatter(updated_1_mu3[0], updated_1_mu3[1], c= 'blue' )

for i in range(len(zero_2_class)):
    plt.text(zero_2_class[i][0], zero_2_class[i][1],(zero_2_class[i][0],zero_2_class[i][1]) )
for i in range(len(one_2_class)):
    plt.text(one_2_class[i][0], one_2_class[i][1],(one_2_class[i][0], one_2_class[i][1]) )
for i in range(len(two_2_class)):
    plt.text(two_2_class[i][0], two_2_class[i][1],(two_2_class[i][0], two_2_class[i][1]) )
plt.text(updated_1_mu1[0], updated_1_mu1[1], (updated_1_mu1[0], updated_1_mu1[1]) )
plt.text(updated_1_mu2[0], updated_1_mu2[1], (updated_1_mu2[0], updated_1_mu2[1]) )
plt.text(updated_1_mu3[0], updated_1_mu3[1], (updated_1_mu3[0], updated_1_mu3[1]) )

plt.savefig('task2_iter2_a.jpg')

plt.figure(figsize=(14,5))
for i in range(len(zero_2_class)):
    plt.scatter(zero_2_class[i][0], zero_2_class[i][1],marker='^', edgecolor = 'black', c = 'None' )
for i in range(len(one_2_class)):
    plt.scatter(one_2_class[i][0], one_2_class[i][1],marker='^',edgecolor = 'black', c = 'None' )
for i in range(len(two_2_class)):
    plt.scatter(two_2_class[i][0], two_2_class[i][1],marker='^',edgecolor = 'black', c = 'None' )
plt.scatter(updated_2_mu1[0], updated_2_mu1[1], c= 'red' )
plt.scatter(updated_2_mu2[0], updated_2_mu2[1], c= 'green' )
plt.scatter(updated_2_mu3[0], updated_2_mu3[1], c= 'blue' )

for i in range(len(zero_2_class)):
    plt.text(zero_2_class[i][0], zero_2_class[i][1],(zero_2_class[i][0],zero_2_class[i][1]) )
for i in range(len(one_2_class)):
    plt.text(one_2_class[i][0], one_2_class[i][1],(one_2_class[i][0], one_2_class[i][1]) )
for i in range(len(two_2_class)):
    plt.text(two_2_class[i][0], two_2_class[i][1],(two_2_class[i][0], two_2_class[i][1]) )
plt.text(updated_2_mu1[0], updated_2_mu1[1], (updated_2_mu1[0], updated_2_mu1[1]) )
plt.text(updated_2_mu2[0], updated_2_mu2[1], (updated_2_mu2[0], updated_2_mu2[1]) )
plt.text(updated_2_mu3[0], updated_2_mu3[1], (updated_2_mu3[0], updated_2_mu3[1]) )



plt.savefig('task2_iter2_b.jpg')



import numpy as np
import cv2
import copy 
img = cv2.imread('baboon.png',1)
X = img
######################for K= 3#########################################################
K = 3
print(K)
for x in range(0, K*3):
    d["co_ordinates{0}".format(x)] = np.zeros((X.shape[0],X.shape[1])) + x
initial = True
count = 0
while(initial):
    my_dict = d.copy()
    x_co = []
    y_co = []
    z_co = []
    for i in range(0,K*3):
        if((i%3) == 0):
            dist_x = np.square(X[:,:,0] - d["co_ordinates{0}".format(i)])
            x_co.append(dist_x)
        if((i%3) == 1):
            dist_y = np.square(X[:,:,1] - d["co_ordinates{0}".format(i)])
            y_co.append(dist_y)
        if((i%3) == 2):
            dist_z = np.square(X[:,:,2] - d["co_ordinates{0}".format(i)])
            z_co.append(dist_z)
    distance_from_centers = []
    for i in range(0,K):
        distance_from_centers.append(x_co[i]+y_co[i]+z_co[i])
    temp = np.zeros((1,1200*1200))
    for i in range(len(distance_from_centers)):
        temp = np.concatenate((temp, distance_from_centers[i].flatten().reshape(1,1200*1200)), axis=0)
    class_array = np.argmin(temp[1:, : ], axis=0).reshape(1200,1200)
    unique, counts = np.unique(class_array, return_counts=True)
    count_of_elements = dict(zip(unique, counts))
    index_num = 0
    for i in range(len(distance_from_centers)):
        try:
            x_update = np.sum(np.multiply(X[:,:,0],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+index_num)] = np.zeros((X.shape[0],X.shape[1])) + x_update
            y_update = np.sum(np.multiply(X[:,:,1],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+1+index_num)] = np.zeros((X.shape[0],X.shape[1])) + y_update
            z_update = np.sum(np.multiply(X[:,:,2],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+2+index_num)] = np.zeros((X.shape[0],X.shape[1])) + z_update
            index_num = index_num + 2
        except KeyError:
            pass
    flag=0
    if(str(my_dict)==str(d)):
        break
        
s = np.where(class_array == 0)
p = np.where(class_array == 1)
q = np.where(class_array == 2)
P = np.zeros((1200,1200,3))  
for i in range(s[0].shape[0]):
    P[s[0][i],s[1][i]] = np.array([d['co_ordinates0'][0,0],d['co_ordinates1'][0,0],d['co_ordinates2'][0,0]])
for i in range(p[0].shape[0]):
    P[p[0][i],p[1][i]] = np.array([d['co_ordinates3'][0,0],d['co_ordinates4'][0,0],d['co_ordinates5'][0,0]])
for i in range(q[0].shape[0]):
    P[q[0][i],q[1][i]] = np.array([d['co_ordinates6'][0,0],d['co_ordinates7'][0,0],d['co_ordinates8'][0,0]])
import numpy as np
import cv2
import matplotlib.pyplot as plt

cv2.imwrite('task2_baboon_3.jpg', P)

######################## for k = 5############################################

K = 5
print(K)
for x in range(0, K*3):
    d["co_ordinates{0}".format(x)] = np.zeros((X.shape[0],X.shape[1])) + x
initial = True
count = 0
while(initial):
    my_dict = d.copy()
    x_co = []
    y_co = []
    z_co = []
    for i in range(0,K*3):
        if((i%3) == 0):
            dist_x = np.square(X[:,:,0] - d["co_ordinates{0}".format(i)])
            x_co.append(dist_x)
        if((i%3) == 1):
            dist_y = np.square(X[:,:,1] - d["co_ordinates{0}".format(i)])
            y_co.append(dist_y)
        if((i%3) == 2):
            dist_z = np.square(X[:,:,2] - d["co_ordinates{0}".format(i)])
            z_co.append(dist_z)
    distance_from_centers = []
    for i in range(0,K):
        distance_from_centers.append(x_co[i]+y_co[i]+z_co[i])
    temp = np.zeros((1,1200*1200))
    for i in range(len(distance_from_centers)):
        temp = np.concatenate((temp, distance_from_centers[i].flatten().reshape(1,1200*1200)), axis=0)
    class_array = np.argmin(temp[1:, : ], axis=0).reshape(1200,1200)
    unique, counts = np.unique(class_array, return_counts=True)
    count_of_elements = dict(zip(unique, counts))
    index_num = 0
    for i in range(len(distance_from_centers)):
        try:
            x_update = np.sum(np.multiply(X[:,:,0],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+index_num)] = np.zeros((X.shape[0],X.shape[1])) + x_update
            y_update = np.sum(np.multiply(X[:,:,1],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+1+index_num)] = np.zeros((X.shape[0],X.shape[1])) + y_update
            z_update = np.sum(np.multiply(X[:,:,2],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+2+index_num)] = np.zeros((X.shape[0],X.shape[1])) + z_update
            index_num = index_num + 2
        except KeyError:
            pass
    flag=0
    if(str(my_dict)==str(d)):
        break
        
s1 = np.where(class_array == 0)
s2 = np.where(class_array == 1)
s3 = np.where(class_array == 2)
s4 = np.where(class_array == 3)
s5 = np.where(class_array == 4)
P = np.zeros((1200,1200,3))  
for i in range(s1[0].shape[0]):
    P[s1[0][i],s1[1][i]] = np.array([d['co_ordinates0'][0,0],d['co_ordinates1'][0,0],d['co_ordinates2'][0,0]])
for i in range(s2[0].shape[0]):
    P[s2[0][i],s2[1][i]] = np.array([d['co_ordinates3'][0,0],d['co_ordinates4'][0,0],d['co_ordinates5'][0,0]])
for i in range(s3[0].shape[0]):
    P[s3[0][i],s3[1][i]] = np.array([d['co_ordinates6'][0,0],d['co_ordinates7'][0,0],d['co_ordinates8'][0,0]])
for i in range(s4[0].shape[0]):
    P[s4[0][i],s4[1][i]] = np.array([d['co_ordinates9'][0,0],d['co_ordinates10'][0,0],d['co_ordinates11'][0,0]])
for i in range(s5[0].shape[0]):
    P[s5[0][i],s5[1][i]] = np.array([d['co_ordinates12'][0,0],d['co_ordinates13'][0,0],d['co_ordinates14'][0,0]])
import numpy as np
import cv2
import matplotlib.pyplot as plt

cv2.imwrite('task2_baboon_5.jpg', P)

################for k = 10################################
K = 10
print(K)
for x in range(0, K*3):
    d["co_ordinates{0}".format(x)] = np.zeros((X.shape[0],X.shape[1])) + x
initial = True
count = 0
while(initial):
    my_dict = d.copy()
    x_co = []
    y_co = []
    z_co = []
    for i in range(0,K*3):
        if((i%3) == 0):
            dist_x = np.square(X[:,:,0] - d["co_ordinates{0}".format(i)])
            x_co.append(dist_x)
        if((i%3) == 1):
            dist_y = np.square(X[:,:,1] - d["co_ordinates{0}".format(i)])
            y_co.append(dist_y)
        if((i%3) == 2):
            dist_z = np.square(X[:,:,2] - d["co_ordinates{0}".format(i)])
            z_co.append(dist_z)
    distance_from_centers = []
    for i in range(0,K):
        distance_from_centers.append(x_co[i]+y_co[i]+z_co[i])
    temp = np.zeros((1,1200*1200))
    for i in range(len(distance_from_centers)):
        temp = np.concatenate((temp, distance_from_centers[i].flatten().reshape(1,1200*1200)), axis=0)
    class_array = np.argmin(temp[1:, : ], axis=0).reshape(1200,1200)
    unique, counts = np.unique(class_array, return_counts=True)
    count_of_elements = dict(zip(unique, counts))
    index_num = 0
    for i in range(len(distance_from_centers)):
        try:
            x_update = np.sum(np.multiply(X[:,:,0],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+index_num)] = np.zeros((X.shape[0],X.shape[1])) + x_update
            y_update = np.sum(np.multiply(X[:,:,1],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+1+index_num)] = np.zeros((X.shape[0],X.shape[1])) + y_update
            z_update = np.sum(np.multiply(X[:,:,2],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+2+index_num)] = np.zeros((X.shape[0],X.shape[1])) + z_update
            index_num = index_num + 2
        except KeyError:
            pass
    flag=0
    if(str(my_dict)==str(d)):
        break
        
s1 = np.where(class_array == 0)
s2 = np.where(class_array == 1)
s3 = np.where(class_array == 2)
s4 = np.where(class_array == 3)
s5 = np.where(class_array == 4)
s6 = np.where(class_array == 5)
s7 = np.where(class_array == 6)
s8 = np.where(class_array == 7)
s9 = np.where(class_array == 8)
s10 = np.where(class_array == 9)
P = np.zeros((1200,1200,3))  
for i in range(s1[0].shape[0]):
    P[s1[0][i],s1[1][i]] = np.array([d['co_ordinates0'][0,0],d['co_ordinates1'][0,0],d['co_ordinates2'][0,0]])
for i in range(s2[0].shape[0]):
    P[s2[0][i],s2[1][i]] = np.array([d['co_ordinates3'][0,0],d['co_ordinates4'][0,0],d['co_ordinates5'][0,0]])
for i in range(s3[0].shape[0]):
    P[s3[0][i],s3[1][i]] = np.array([d['co_ordinates6'][0,0],d['co_ordinates7'][0,0],d['co_ordinates8'][0,0]])
for i in range(s4[0].shape[0]):
    P[s4[0][i],s4[1][i]] = np.array([d['co_ordinates9'][0,0],d['co_ordinates10'][0,0],d['co_ordinates11'][0,0]])
for i in range(s5[0].shape[0]):
    P[s5[0][i],s5[1][i]] = np.array([d['co_ordinates12'][0,0],d['co_ordinates13'][0,0],d['co_ordinates14'][0,0]])
for i in range(s6[0].shape[0]):
    P[s6[0][i],s6[1][i]] = np.array([d['co_ordinates15'][0,0],d['co_ordinates16'][0,0],d['co_ordinates17'][0,0]])
for i in range(s7[0].shape[0]):
    P[s7[0][i],s7[1][i]] = np.array([d['co_ordinates18'][0,0],d['co_ordinates19'][0,0],d['co_ordinates20'][0,0]])
for i in range(s8[0].shape[0]):
    P[s8[0][i],s8[1][i]] = np.array([d['co_ordinates21'][0,0],d['co_ordinates22'][0,0],d['co_ordinates23'][0,0]])
for i in range(s9[0].shape[0]):
    P[s9[0][i],s9[1][i]] = np.array([d['co_ordinates24'][0,0],d['co_ordinates25'][0,0],d['co_ordinates26'][0,0]])
for i in range(s10[0].shape[0]):
    P[s10[0][i],s10[1][i]] = np.array([d['co_ordinates27'][0,0],d['co_ordinates28'][0,0],d['co_ordinates29'][0,0]])
import numpy as np
import cv2
import matplotlib.pyplot as plt

cv2.imwrite('task2_baboon_10.jpg', P)

################fo k = 20##############################################

ee_x = []
ee_y = []
ee_z = []
tt = []
count = 0

for i in range(36,230,8):
    count = count + 1
    ee_x.append(i)
    if(count == 20):
        break
for i in range(42,200,7):
    count = count + 1
    ee_y.append(i)
    if(count == 20):
        break
for i in range(45,240,9):
    count = count + 1
    ee_z.append(i)
    if(count == 20):
        break
for i in range(len(ee_x)):
    tt.append(ee_x[i])
    tt.append(ee_y[i])
    tt.append(ee_z[i])
    
K = 20
print(K)
for x in range(0, K*3):
    d["co_ordinates{0}".format(x)] = np.zeros((X.shape[0],X.shape[1])) + tt[x]
initial = True
count = 0
while(initial):
    my_dict = d.copy()
    x_co = []
    y_co = []
    z_co = []
    for i in range(0,K*3):
        if((i%3) == 0):
            dist_x = np.square(X[:,:,0] - d["co_ordinates{0}".format(i)])
            x_co.append(dist_x)
        if((i%3) == 1):
            dist_y = np.square(X[:,:,1] - d["co_ordinates{0}".format(i)])
            y_co.append(dist_y)
        if((i%3) == 2):
            dist_z = np.square(X[:,:,2] - d["co_ordinates{0}".format(i)])
            z_co.append(dist_z)
    distance_from_centers = []
    for i in range(0,K):
        distance_from_centers.append(x_co[i]+y_co[i]+z_co[i])
    temp = np.zeros((1,1200*1200))
    for i in range(len(distance_from_centers)):
        temp = np.concatenate((temp, distance_from_centers[i].flatten().reshape(1,1200*1200)), axis=0)
    class_array = np.argmin(temp[1:, : ], axis=0).reshape(1200,1200)
    unique, counts = np.unique(class_array, return_counts=True)
    count_of_elements = dict(zip(unique, counts))
    index_num = 0
    for i in range(len(distance_from_centers)):
        try:
            x_update = np.sum(np.multiply(X[:,:,0],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+index_num)] = np.zeros((X.shape[0],X.shape[1])) + x_update
            y_update = np.sum(np.multiply(X[:,:,1],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+1+index_num)] = np.zeros((X.shape[0],X.shape[1])) + y_update
            z_update = np.sum(np.multiply(X[:,:,2],np.where(class_array == i,1,0)))/count_of_elements[i]
            d["co_ordinates{0}".format(i+2+index_num)] = np.zeros((X.shape[0],X.shape[1])) + z_update
            index_num = index_num + 2
        except KeyError:
            pass
    flag=0
    if(str(my_dict)==str(d)):
        break
s1 = np.where(class_array == 0)
s2 = np.where(class_array == 1)
s3 = np.where(class_array == 2)
s4 = np.where(class_array == 3)
s5 = np.where(class_array == 4)
s6 = np.where(class_array == 5)
s7 = np.where(class_array == 6)
s8 = np.where(class_array == 7)
s9 = np.where(class_array == 8)
s10 = np.where(class_array == 9)
s11 = np.where(class_array == 10)
s12 = np.where(class_array == 11)
s13 = np.where(class_array == 12)
s14 = np.where(class_array == 13)
s15 = np.where(class_array == 14)
s16 = np.where(class_array == 15)
s17 = np.where(class_array == 16)
s18 = np.where(class_array == 17)
s19 = np.where(class_array == 18)
s20 = np.where(class_array == 19)
P = np.zeros((1200,1200,3))  
for i in range(s1[0].shape[0]):
    P[s1[0][i],s1[1][i]] = np.array([d['co_ordinates0'][0,0],d['co_ordinates1'][0,0],d['co_ordinates2'][0,0]])
for i in range(s2[0].shape[0]):
    P[s2[0][i],s2[1][i]] = np.array([d['co_ordinates3'][0,0],d['co_ordinates4'][0,0],d['co_ordinates5'][0,0]])
for i in range(s3[0].shape[0]):
    P[s3[0][i],s3[1][i]] = np.array([d['co_ordinates6'][0,0],d['co_ordinates7'][0,0],d['co_ordinates8'][0,0]])
for i in range(s4[0].shape[0]):
    P[s4[0][i],s4[1][i]] = np.array([d['co_ordinates9'][0,0],d['co_ordinates10'][0,0],d['co_ordinates11'][0,0]])
for i in range(s5[0].shape[0]):
    P[s5[0][i],s5[1][i]] = np.array([d['co_ordinates12'][0,0],d['co_ordinates13'][0,0],d['co_ordinates14'][0,0]])
for i in range(s6[0].shape[0]):
    P[s6[0][i],s6[1][i]] = np.array([d['co_ordinates15'][0,0],d['co_ordinates16'][0,0],d['co_ordinates17'][0,0]])
for i in range(s7[0].shape[0]):
    P[s7[0][i],s7[1][i]] = np.array([d['co_ordinates18'][0,0],d['co_ordinates19'][0,0],d['co_ordinates20'][0,0]])
for i in range(s8[0].shape[0]):
    P[s8[0][i],s8[1][i]] = np.array([d['co_ordinates21'][0,0],d['co_ordinates22'][0,0],d['co_ordinates23'][0,0]])
for i in range(s9[0].shape[0]):
    P[s9[0][i],s9[1][i]] = np.array([d['co_ordinates24'][0,0],d['co_ordinates25'][0,0],d['co_ordinates26'][0,0]])
for i in range(s10[0].shape[0]):
    P[s10[0][i],s10[1][i]] = np.array([d['co_ordinates27'][0,0],d['co_ordinates28'][0,0],d['co_ordinates29'][0,0]])
for i in range(s11[0].shape[0]):
    P[s11[0][i],s11[1][i]] = np.array([d['co_ordinates30'][0,0],d['co_ordinates31'][0,0],d['co_ordinates32'][0,0]])
for i in range(s12[0].shape[0]):
    P[s12[0][i],s12[1][i]] = np.array([d['co_ordinates33'][0,0],d['co_ordinates34'][0,0],d['co_ordinates35'][0,0]])
for i in range(s13[0].shape[0]):
    P[s13[0][i],s13[1][i]] = np.array([d['co_ordinates36'][0,0],d['co_ordinates37'][0,0],d['co_ordinates38'][0,0]])
for i in range(s14[0].shape[0]):
    P[s14[0][i],s14[1][i]] = np.array([d['co_ordinates39'][0,0],d['co_ordinates40'][0,0],d['co_ordinates41'][0,0]])
for i in range(s15[0].shape[0]):
    P[s15[0][i],s15[1][i]] = np.array([d['co_ordinates42'][0,0],d['co_ordinates43'][0,0],d['co_ordinates44'][0,0]])
for i in range(s16[0].shape[0]):
    P[s16[0][i],s16[1][i]] = np.array([d['co_ordinates45'][0,0],d['co_ordinates46'][0,0],d['co_ordinates47'][0,0]])
for i in range(s17[0].shape[0]):
    P[s17[0][i],s17[1][i]] = np.array([d['co_ordinates48'][0,0],d['co_ordinates49'][0,0],d['co_ordinates50'][0,0]])
for i in range(s18[0].shape[0]):
    P[s18[0][i],s18[1][i]] = np.array([d['co_ordinates51'][0,0],d['co_ordinates52'][0,0],d['co_ordinates53'][0,0]])
for i in range(s19[0].shape[0]):
    P[s19[0][i],s19[1][i]] = np.array([d['co_ordinates54'][0,0],d['co_ordinates55'][0,0],d['co_ordinates56'][0,0]])
for i in range(s20[0].shape[0]):
    P[s20[0][i],s20[1][i]] = np.array([d['co_ordinates57'][0,0],d['co_ordinates58'][0,0],d['co_ordinates59'][0,0]])
import numpy as np
import cv2
import matplotlib.pyplot as plt

cv2.imwrite('task2_baboon_20.jpg', P)

