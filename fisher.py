# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:08:52 2020

@author: 30790
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def compute_si(n,A):
    A=A[:,:n]
    A_mean=np.mean(A,axis=0)
    length,n=np.shape(A) #length为数组样本数，n为数组纬度数
    si=np.zeros([n,n])
    for i in range(0,length,1):
        a1=A[i,:]-A_mean
        g=np.array(a1*np.array([a1]).T)
        si=si+g
    return si

def compute_d(a,b):
    mu1=np.array(a)
    mu2=np.array(b)
    d=np.array([mu1-mu2])
    return d

def compute_w(sw,d):
    w=(np.dot(np.linalg.inv(sw),d.T)).T
    return w

def compute_y0(a,b,w):
    na,s=np.shape(a)
    nb,s=np.shape(b)
    a_y0=np.zeros([na,1]) #y0
    b_y0=np.zeros([nb,1])
#计算μ1 μ2 得到分类点y0
    for i in range(0,na,1):
        a_y0[i]=np.dot(w,a[i,:])
      
        y0_mu1=np.mean(a_y0)
        
    for i in range(0,nb,1):
        b_y0[i]=np.dot(w,b[i,:])
        y0_mu2=np.mean(b_y0)
    y0=(y0_mu1+y0_mu2)/2
    return y0
#def compute_acc():
    
if __name__ == '__main__':
    iris_df = datasets.load_iris()
    iris_2=iris_df.data[50:100,0:4]
    iris_3=iris_df.data[100:150,0:4]
    target2=np.ones([50,1])
    target3=np.ones([50,1])*2
    iris2_train, iris2_test, target2_train, target2_test = train_test_split(iris_2, target2, test_size=0.4, random_state=6)
    iris3_train, iris3_test, target3_train, target3_test = train_test_split(iris_3, target3, test_size=0.4, random_state=6)
    
    iris2_mean=np.mean(iris2_train,axis=0)
    iris3_mean=np.mean(iris3_train,axis=0)
    s2=np.zeros([4,4])
    s3=np.zeros([4,4])
    s2=compute_si(4,iris2_train) ######???????
    s3=compute_si(4,iris3_train)
    sw0=s2+s3
    d0=compute_d(iris2_mean,iris3_mean)
    w0=compute_w(sw0,d0)
    y_0=compute_y0(iris2_train,iris3_train,w0)
    correct2=0
    correct3=0
    iris2_y0_target=np.zeros([20,1])
    iris3_y0_target=np.zeros([20,1])
    iris2_y0_test=np.zeros([20,1])
    iris3_y0_test=np.zeros([20,1])

    for i in range(0,20,1):
        iris2_y0_test[i]=np.dot(w0,np.array(iris2_test[i,:]))
        if iris2_y0_test[i]>=y_0:
            iris2_y0_target[i]=1
        else:
            iris2_y0_target[i]=2
            #判断是否正确
        if iris2_y0_target[i]==target2_test[i,:]:
            correct2=correct2+1
        else:
            continue 
    
        iris3_y0_test[i]=np.dot(w0,np.array(iris3_test[i,:]))
        if iris3_y0_test[i]<y_0:
            iris3_y0_target[i]=2
        else:
            iris3_y0_target[i]=1
        if iris3_y0_target[i]==target3_test[i]:
            correct3=correct3+1
        else:
            continue
    accuracy=(correct3+correct2)/40
    print("1 2类鸢尾花的分类准确率为：{}".format(accuracy))
    q=[]
    for i in range(20):
        q.append(0)
    plt.scatter(iris2_y0_test,q,color='r',label='class2')
    plt.scatter(iris3_y0_test,q,color='b',label='class3')
    plt.legend(loc="upper left")
#    plt.savefig('C:\\Users\\30790\\Desktop\\Fisher\\iris12.png')
    plt.show()