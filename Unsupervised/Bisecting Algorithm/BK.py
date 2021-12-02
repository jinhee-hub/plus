import argparse 
import numpy as np
from numpy import array
from numpy import linalg as lg
import random
import math

#No.1
####### make user interface

def Parser():
    parser= argparse.ArgumentParser()
    parser.add_argument('-data')   # argument for data.txt
    parser.add_argument('-k')    # total number of desired clusters
    parser.add_argument('-s')   # minimum number of data points by one cluster
    parser.add_argument('-d')   # maximum intra_cluster distance by one cluster`
    parser.add_argument('-output') # output file
    

    return parser



class Node:    
    
    def __init__(self,value,d):        
        self.key = value       # value = number of cluster
        self.d = d             # d= all data in this cluster
        self.left = None       # Left Child        
        self.right = None      # Right Child
        self.lv=None

class Tree:

    def __init__(self):   
        self.root=None
        self.leaf=[]
        self.leafkey=[]   
        
    def addRoot(self, value, d):     # add Root for tree
        if self.root is None:
            self.root = Node(value,d)
            self.root.lv=0            # root node's level
            self.leaf.append(self.root)      # currently, this is only leaf node
            self.leafkey.append(self.root.key)  # get its value

    def addNode(self, value1, value2, d1, d2, m):        
        self._addNode(self.root, value1, value2, d1, d2, m)
              
    def _addNode(self, node, value1, value2, d1 ,d2, m):        

        if node.right is None:           # no right child means it also have no left child 
            self.leaf.remove(node)           # this node has children, so it is not a leaf node anymore
            self.leafkey.remove(node.key)    # so, remove it
            node.right=Node(value2, d2)      # make nodes for right and left children
            node.left=Node(value1, d1)
            node.right.lv=node.lv+1          # child level = this node level + 1
            node.left.lv=node.lv+1            
            self.leaf.append(node.right)     # before make children, these left, right child node are leaf nodes
            self.leaf.append(node.left)
            self.leafkey.append(node.right.key)
            self.leafkey.append(node.left.key)
           
        else:                       
            maxidx=self.leafkey.index(max(self.leafkey))   # get node who has maximum value
            newnode=self.leaf[maxidx]            
            self._addNode(newnode, value1, value2, d1, d2, m)      # do again
    
             

    def lvorder(self):       # get node information level by level
        o=[]             # o has values
        oo=[]            # oo has level data
        def _lvorder(root):
            q=[root]            
            while q:
                root=q.pop(0)   
                #print(root.key, end=" ")
                oo.append(root.lv)
                o.append(root.key)
                if root.left:
                    q.append(root.left)
                if root.right:
                    q.append(root.right)
        _lvorder(self.root)
       
        return o, oo




def initcent(X):
    #initial 2 centroids
    N=len(X)
    index= [i for i in range(N)]
    randidx=random.sample(index,2) # randomly select two different data as centroids
    
    cent1 = X[randidx[0]]
    cent2 = X[randidx[1]]

    return cent1, cent2

# create new centroid as the mean of clustered datapoints
def newcent(X):

    nd=len(X)
    sumd=sum(X)

    newcentroid=sumd/nd

    return newcentroid


# input is a cluster and a centroid of this cluster
def icd(c,cent):   # this function is defined by HW6
    nc=len(c)
  
    centroid=cent    
    
    dist=[]
    for dp in c:
        d=np.linalg.norm(dp-centroid, ord=2)
        dist.append(d)       
            
    sumd=sum(dist)
    Dc=sumd/nc    
    
    return Dc


# input data
def kmeans(X):  # k=2, therefore, 2means

    init=initcent(X)  #initial 2 centroids
    cent1=init[0]    
    cent2=init[1]
    old1=cent1
    old2=cent2
    count=0
        
    while True:      
        count+=1

        if count == 100:  # if iteration is over 100, then stop kmeans
            break

        c1=[]
        c2=[]
        for dp in X:  # dp is each data point in data
        
            # using Euclidean distance
            dis1=np.linalg.norm(dp-cent1, ord=2)  # L2 norm of data-centroid1
            dis2=np.linalg.norm(dp-cent2, ord=2)  # L2 norm of data-centroid2

            if dis1 < dis2:
                c1.append(dp)  # if dp is closed to centroid1, then dp is classifeid as c1 
            else: 
                c2.append(dp)  # else, classified as c2   
        
        
        delta_oldnew1 = np.linalg.norm(old1- cent1, ord=2) # compare old centroid and new centroid.   
        delta_oldnew2 = np.linalg.norm(old2- cent2, ord=2) # compare old centroid and new centroid.      
               

        if delta_oldnew1 < 0.0001 or delta_oldnew2 < 0.0001:  # if there are no changes between old and new centroids, then stop
            break            

        old1=cent1
        old2=cent2
        
        
    return c1, c2, cent1, cent2


# input b is last leaf nodes' datapoints
# input data is data from data.txt
def print_clusters(data,b):
    # bb shows cluster numbering and each datapoints
    #print(b)
    bb=dict()
    cc=dict()   
    for i in range(len(b)):
        bb[i]=b[i]
        cc[i]=len(b[i])
    
    out=[]
    for i in data:
        #print(i)
        for key, value in bb.items():                       
            for d in range(len(value)):    
                              
                if set(value[d]) == set(i):  # if this datapoint is in this cluster                     
                    out.append(key)
                       
    return out
    

def bisecting(X):  
   
    # all datapoints are in one cluster c    
    c=X

    # candidate and num and a are used to get output
    candidate=[]
    num=[]    
    candidate.append(c)
    num.append(len(c))   

    # tree is used for draw dendrogram
    tree=Tree()
    tree.addRoot(len(c),c)
    node=tree.root      
    
    a=[len(c)]   
    b=[c]
    #print(a)
    while True:             
        #print(a)                 # you can see each clusters by each kmeans
        idx=num.index(max(num))   # find a cluster which has the maximum number of data points  
        id=a.index(max(a))           
                                     
        current=candidate[idx]    # This stores datapoints in this cluster 
        m=max(a)   
        
        # criteria for stop: the number of datapoints in one cluster                    I am confused the expression in HW6
        # Even if a cluster has less datapoints than s, it will do further clustering          80
        if len(current) < s:                                        #                         /  ＼
            break                                                   #                        3     77       go further or not? Because left cluster 3 is less than s
                                                                    #                              /  ＼
                                                                    #                             33    44         
        candidate.pop(idx)
        num.pop(idx)       
        a.pop(id)
        b.pop(id)

        km=kmeans(current) # do kmeans for current data       

        # left is the smaller number of datapoints
        if len(km[0]) < len(km[1]):
            left=km[0]
            right=km[1]           
        else:
            left=km[1]
            right=km[0]
        
        candidate.append(left)
        candidate.append(right)
        num.append(len(left))
        num.append(len(right))
        
       
        # each data in a is current leaf node (it means current clusters and each cluster's number of datapoints)        
        a.insert(id,len(left))
        a.insert(id,len(right))                       
        
        # b sotres datapoints
        b.insert(id, left)
        b.insert(id, right)   
        
           
        
        tree.addNode(len(left), len(right), left, right, m)

        # criteria for stop: the number of datapoints in one cluster           it could be end like this example
        # Even if other cluster has many datapoints, it will stop                             80
        # if len(left) < s or len(right) < s:                         #                      /  ＼
        #    break                                                   #                      3     77      <- no more clustering  because 3 < s     
       

        # critera for stop: total number of clusters
        if len(num) >= k:            
            break

        #print(a)
        # criteria: intra-cluster distance 
        Dc1=icd(km[0], km[2])
        Dc2=icd(km[1], km[3])
        if Dc1 < d or Dc2 < d:   # if this cluster's intra-cluster distance is smaller than d   
            continue             #  stop all the bisecting algorithm? or just stop this cluster's clustering
            #break               #  if 'continue'  it will continue do clustering for other larger clusters,
                                 #  if 'break',   even if other clusters have many datapoints, it could be stop if one cluster does not satisfy this criteria. ex) 27 23 29 1  
    
    
    dendrogram=tree.lvorder()[0]   # this list has level order node values
    dd=tree.lvorder()[1]           # and each level
       
    # print dendrogram
    # this dendrogram is expressed by level by level.       For, example,   if  s=6                                           My dendrogram
                                                            #        O                               80                            
    for j in range(max(dd)+1):                              #      /   ＼                         /      ＼                            80
        for i in range(len(dd)):                            #     o       o                     29          51                         29 51
            if dd[i] == j:                                  #   /  ＼     /  ＼     ->        /   ＼        /  ＼           ->         10 19 19 32
                print(dendrogram[i],end=' ')                #  o    o    o     o            10     19      19    32                    2 17 11 21
        print()                                             #      /  ＼      /  ＼              /   ＼         /  ＼                  10 11
                                                            #     o     o    o     o            2     17       11    21
                                                            #                     /  ＼                             /  ＼
                                                            #                    o    o                            10    11
                                                            # only shows level 
    return b


#data=np.genfromtxt('C:/Users/이진희/Desktop/ML3/data1.txt', delimiter=' ')


#k=50   # total number of clusters
#s=6    # minimum number of data points in one cluster
#d=0.001  # intra cluster distance
#b=bisecting(data)


# print cluster of each datapoint

#out=print_clusters(data,b)
#for i in out:
#    print(i)
        



if __name__ == '__main__':    

    parser = Parser()
    args = parser.parse_args()

    data_file=str(args.data)  # data file     

    k=float(args.k)   # total number of clusters
    s=float(args.s)   # minimum number of data points in one cluster
    d=float(args.d)   # intra cluster distance
    output=str(args.output)
        
    data=np.genfromtxt(data_file, delimiter=' ').astype(float)  # read data.txt    
    

    b=bisecting(data)
    
    # print cluster of each datapoint

    f=open("%s" %output,'w')

    pc=print_clusters(data,b)
    #for i in pc:
    #    print(i)
    np.savetxt(f, pc, fmt='%i')

    f.close()




#python3 BK.py -data data1.txt -k cluster_num -s cluster_size -d intra_dist -output output_file.txt

