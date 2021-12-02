######## Kmeans for Cluster Analysis
######## Jinhee Lee lee.8945 
######## python3 kmeans.py -database_file=data1.txt -k=30 -max_iters=20 -eps=0.1 -output_file=output.txt
######## python3 kmeans.py -database_file=data2.txt -k=30 -max_iters=20 -eps=0.1 -output_file=output.txt

import argparse
import random
import math

##### No.1
def Parser():   # make Parser for execute
    parser=argparse.ArgumentParser()
    parser.add_argument('-database_file')
    parser.add_argument('-k')
    parser.add_argument('-max_iters')
    parser.add_argument('-eps')
    parser.add_argument('-output_file')

    return parser


##### No.2
def read_data(databaseName):       # read database

    dataName=databaseName    

    databaseMatrix=[]
    numRow=0

    f1=open(dataName,'r')

    datab=[]
    count=0
    while True:
        line=f1.readline()         # get data
        line=line.strip('\n')
        line=line.split(' ')
        a=line
        if (len(line) < 2):        # get rid of unintentional blank data
           break

        c=[]
        for i in range(0, len(a), 1):
            #print(a[i])
            b=float(a[i])
            c.append(b)
            #print(b)
        if [] in c:
           c.delete([])
        #print(c)
        if not line:
           break
        datab.append(c)
        count+=1
        
    databaseMatrix=datab       

    # one row = one data point  if n
    # one col = dimension,      if m
    # there are n datapoints and each datapoint has m dimensions like 3D, 4D,....

    return databaseMatrix


##### No.3
## make k number of centroids. Each centroid has D dimensions : number of column data in database
def initCentroids(database, k):

   databaseMatrix=database

   centroids=[]
   
   for cl in range(0, k, 1):
       randDp=random.choice(databaseMatrix)   # Choice k datapoints randomly in database
       pp=databaseMatrix.index(randDp)
       centroids.append(randDp)               # These k datapoints are initially k centroids
   #print(centroids[0][0])
   
   return centroids   # make k centroids, each centroid has 950 dimensions in the case of data1


## calculate Euclidean distance using centroids and datapoints dimension vector
def EuclideanDistance(database, centroids):

    databaseMatrix=database
    numRow=len(databaseMatrix)
    numCol=len(databaseMatrix[0])
    
    EuDist=[]                   # Euclidean distance of all datapoints by all centroids
    for dp in range(0, numRow, 1):   # to the all datapoints in database
        

        distByCent=[]           # Euclidean distance of one datapoint by all centroids
        for cl in range(0, k, 1): # to the all k centroids
            
            rr=[]
            for dim in range(0, numCol, 1):  # to the all dimensions in datapoints
                
               r=(centroids[cl][dim]-databaseMatrix[dp][dim]) ** 2

               rr.append(r)

            distByDim=math.sqrt(sum(rr))  # Euclidean distance of one datapoint by one centroid
                
            distByCent.append(distByDim)
                
        EuDist.append(distByCent)

    # EuDist[0]: datapoint's Euclidean distances by all centroids : [ a, b, c, d ]
    # Eudist[0][0]: datapoint's Euclidean distance by one centroid  [ a ]

    #print("Calculate Euclidean Distance")   # for check

    return EuDist


## Assign each datapoints to each clusters    
def AssignCluster(database, EuDist):

    databaseMatrix=database

    numRow=len(databaseMatrix)
    numCol=len(databaseMatrix[0])

    clusterDp=[]
    for dp in range(0, numRow, 1):  # to the all datapoints in database
        minDist=min(EuDist[dp])        # find minimum Euclidean distance between distance to the k centroids
        
        minDistIndex= EuDist[dp].index(minDist) # find minimum distance's Index and that indicate k cluster
                    
        clusterDp.append(minDistIndex)         # assign each datapoints to the each clusters

    cluster=clusterDp

    #print("Assign datapoints to each clusters")   # for check 

    return cluster


## find new centroids using datapoints which are classified as each clusters
def MakeNewCentroids(database, cluster, oldCentroids):

    databaseMatrix=database

    numRow=len(databaseMatrix)
    numCol=len(databaseMatrix[0])
           
    # update the cluster centroids as new centroids            
    NewCentroids=[]    
    for cl in range(0, k, 1):              # to the all clusters

        clustDp=[]
        for dp in range(0, numRow, 1):     # to the all datapoints           
                
            if cluster[dp] == cl:           # if datapoint is classified as specific cluster between k clusters
                dpoint=databaseMatrix[dp]   # get this datapoint
                clustDp.append(dpoint)      # assign it at specific cluster between k clusters
            
        transClustdp=zip(*clustDp)          # transform cluster-Matrix to calculate mean of each dimension
        numnewdp=len(clustDp)               # number of datapoints of specific cluster    
            

        newcent=[]
        for x in transClustdp:              # x are points of specific dimension 
            sumdim=sum(x)                   # sum these points
            newpoint=sumdim/numnewdp        # calculate mean of these points        
            newcent.append(newpoint)        # new point of all dimensions
            
        NewCentroids.append(newcent)        # new centroid are consist of D dimension points
                                            # NewCentroids are consist of new centroid by each k clusters
        if not NewCentroids[cl] :           # to avoid empty clustering
            NewCentroids=oldCentroids
            #print("sometimes a cluster cannot have any data points ")

    return NewCentroids


## calculate differences of new and old centroids to limit iterations
def NewOldDifference(database, oldCentroids, newCentroids):

    databaseMatrix=database

    numRow=len(databaseMatrix)
    numCol=len(databaseMatrix[0])

    DiffOldNew=[]
    for cl in range(0, k, 1):               # to the all clusters        

        rr1=[]
        
        for dim in range(0, numCol, 1):     # to the all dimensions(attributes) of each centroid

        

            old=oldCentroids[cl][dim]       # old centroids
            new=newCentroids[cl][dim]       # new centroids
            
                
            r1=(old-new) ** 2
            rr1.append(r1)
                
        dist2ByDim=math.sqrt(sum(rr1))        # calculate Euclidean distance of old and new's one centroid
            
        Rdist2ByDim=dist2ByDim                 

        DiffOldNew.append(Rdist2ByDim)        # to the all k centroids

        
    return DiffOldNew

## make cluster datapoints as output format
def ClusterFormOutput(database, cluster):

    databaseMatrix=database
    numRow=len(databaseMatrix)
    numCol=len(databaseMatrix[0])
    
    DpCluster=[]
    for cl in range(0, k, 1):              # to the all clusters
        clusterDpIndex=[]
        for dp in range(0, numRow, 1):     # to the all datapoints
            if cluster[dp] == cl:        # if datapoints assigned to the k cluster
                clusterDpIndex.append(dp)   # store datapoint's index
        DpCluster.append(clusterDpIndex)    # store each cluster's assigned datapoints

    return DpCluster


### Output Format and make output File
def outputKClusters(output_file, DpCluster):
    f=open(output_file, 'w+')

    num=[]
    eachnum=[]
    for cl in range(len(DpCluster)):
        eachnum.append(len(DpCluster[cl]))
        f.write(str(cl) + ': ' + str(' '.join([str(i) for i in DpCluster[cl]]) + '\n'))
        print(cl,':', ' '.join([str(i) for i in DpCluster[cl]]),' ')   # get output file as format of output
    num.append(eachnum)
    #print("number of datapoints in each clusters", num) 
    

## generate Kmeans to classified datapoints as k clusters
def genKmeans(database, k, n, e, output_file):

    database=database    

    # randomly initialize k centroids
    centroids=initCentroids(database,k)      # k centroids [ [Dim], [Dim], [Dim], [Dim] ]
    

    oldCentroids=centroids      # make old centroids to compare to e    

    itr=True        # it used to limit iterations using e

    OutKclusters=[]
    num=[]
    for iteration in range(0, n, 1) or itr==True:  # make n iterations        

        # Euclidean Distance
        EuDist=EuclideanDistance(database, centroids)   # calculate Euclidean Distance

        # assign each datapoint to each of the k clusters based on Euclidean distance
        cluster=AssignCluster(database, EuDist)        # assign each datapoint to each cluster

        # update the cluster centroids
        NewCentroids=MakeNewCentroids(database, cluster, oldCentroids)  # create new centroids by average of each dimension vector points    


        oldCentroids=centroids              # store centroids at oldcentroids
        newCentroids=NewCentroids           # new centroids is updated at NewCentroids
        centroids=newCentroids              # assign centroids as new centroids


        # calculate differences between old and new centroids
        DiffOldNew=NewOldDifference(database, oldCentroids, newCentroids)

        ## clustering datapoints as output format
        DpCluster=ClusterFormOutput(database, cluster) 
        

        # compare to the e for iterations until difference of new and old centroids are less than e
        count=0

        #print("iteration", iteration)
        
        for cl in range(0, k, 1):            
            if (DiffOldNew[cl] < e ):   # compare differences of old and new centroids to e
                                       # must all differences less than e to end iteration               
                count+=1
                if count == k:    # all k centroids are less than e                     
                    itr=False                    
        if(itr==False):
            break               

    # output File     
    outputKClusters(output_file, DpCluster)              


if __name__ == '__main__':

    #Argument parser to read
    parser = Parser()
    args = parser.parse_args()
    databaseName=str(args.database_file)    # database file name
    k=int(args.k)                           # number of k clusters
    n=int(args.max_iters)                   # number of iterations
    e=float(args.eps)                       # e-difference for limits of difference of old and new centroids
    outputFile=str(args.output_file)
    
    database=read_data(databaseName)        # read database

    genKmeans(database, k, n, e, outputFile)   # apply kmeans function

