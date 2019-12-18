#Author - Ashish Kumar Jayant (16730), M.Tech Research, CSA
#Title - Linear Algebra Assignent -2 

import sys
import networkx as nx
import matplotlib.pyplot as plt




problem = sys.argv[1]
path = sys.argv[2]



def give_zero_matrix(m,n):
    c=[]
    for i in range(m):
        c.append([])
        for j in range(n):
            c[i].append(0)
        
    return c

def give_me_identity(n):
    I=[]
    for i in range(n):
        l=[]
        for j in range(n):
            if i!=j:
                l.append(0)
            else:
                l.append(1)
        I.append(l)
    return I
def norm(a,flag):
    if flag==0:
        s=0
        s=[i**2 for i in a]
        s1=sum(s)
        return s1**(1/2)    
    else:
        s=0
        s=[i[0]**2 for i in a]
        s1=sum(s)
        return s1**(1/2)

def matrix_mul(a,b):
    c=give_zero_matrix(len(a),len(b[0]))
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                c[i][j] = c[i][j] + a[i][k]*b[k][j]
    return c
def transpose(a):
    c=give_zero_matrix(len(a[0]),len(a))
    for i in range(len(c)):
        for j in range(len(c[0])):
            c[i][j] = a[j][i] 
    return c
def get_column_k(a,k):
    c = [[a[i][k]] for i in range(len(a))]
    return c
def set_column_k(a,v,k):
    for i in range(len(a)):
        a[i][k] = v[k][0]
    return a
def scalar_mul(a,k):
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j]=k*a[i][j]
    return(a)

def scalar_div(a,k):
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j]=a[i][j]/k
    return a
def matrix_diff(a,b):
    c = give_zero_matrix(len(a),len(a[0]))
    for i in range(len(a)):
        for j in range(len(a[0])):
            c[i][j] = a[i][j] - b[i][j]
    return(c)    
def add_column(Y,v):
    for i in range(len(Y)):
        Y[i].extend(v[i])
    return Y

def gs_cofficient(v1, v2):
    x = matrix_mul(transpose(v2), v1) 
    x = scalar_div(x,norm(v1,1)**2)
    return x[0][0]


def proj(v1, v2):
    return scalar_mul(v1,gs_cofficient(v1, v2))

def grahm_sch(X):
    Y = []
    for i in range(len(X[0])):
        temp_vec = get_column_k(X,i)
        if len(Y)!=0:
            for j in range(len(Y[0])) :
                inY = get_column_k(Y,j)
                x_i = get_column_k(X,i)
                proj_vec = proj(inY, x_i)
                temp_vec = matrix_diff(temp_vec,proj_vec)
                
        if i==0:
            Y.extend(temp_vec)
        else:
            Y = add_column(Y,temp_vec)
    norm_vec =[]    
    for i in range(len(Y[0])):
        x = get_column_k(Y,i)
        norm_vec.append(norm(x,1))
    for i in range(len(Y)):
        for j in range(len(Y[0])):
            Y[i][j] = Y[i][j]/norm_vec[j]
    Q=Y
    R = matrix_mul(transpose(Q),X)
    #R = round_mat(R)
    #Q = round_mat(Q)
    return Q,R
def round_mat(a):
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j]=round(a[i][j],8)
    return a

#----------IDEAL FOR SYMMETRIC MATRICES-------------------------------------------
def qr_algorithm(A,n):
    a=A
    U = give_me_identity(len(A))
    for i in range(n):
        q,r = grahm_sch(a)#q-r factorization
        U = matrix_mul(U,q)
        a_1 = matrix_mul(r,q)
        #print(a_1)
        a=a_1
    return a,U


def eigvals_scratch_qr(a):
    a,e_vec=qr_algorithm(a,100)           #Running QR Algorithm for 400 iterations
    #print("Approximated triangular matrix using QR Algorithm- ")
    #print(a)
    eig=[a[i][i] for i in range(len(a))]
    return eig,e_vec 

#All pair shortest path aux function - Floyd Warshall
def floyd_marshall_ap_shortest_paths(a):
    V = range(len(a))
    for k in V:
        for u in V:
            for v in V:
                a[u][v] = min(a[u][v],a[u][k] + a[k][v])
    for v in V:
        if a[v][v] < 0:      
            return True
    return False

import copy

#-------------------K-means algorithm scratch implementation for k=2 and 1D data (acc to requirement of assignment)----------------------------------
def distance(x1,x2):
    return abs(x1-x2)
def mean(v):
    return sum(v)/len(v)
def update_centroid(labels,v):
    cluster ={}
    for key,value in labels.items():
        if value not in cluster.keys():
            cluster[value] = [v[key]]
        else:
            cluster[value].append(v[key])
    centroid_1 = mean(cluster[1])
    centroid_2 = mean(cluster[2])
    return centroid_1,centroid_2
        
#for two cluster 
def kmeans_scratch(v,seed):
    centroid_1 = seed
    centroid_2 = -seed
    
    labels = {}
    delta_centroid=999999                  #assigning high value
    while (delta_centroid>0.5):           #till centroid don't change that much 
        c=0
        for i in v:
            d_centroid_1 = distance(i,centroid_1)
            d_centroid_2 = distance(i,centroid_2)
            if d_centroid_1>d_centroid_2:
                labels[c]=1
            else:
                labels[c]=2
            c+=1
        
        old_centroid_1 = copy.deepcopy(centroid_1)
        centroid_1,centroid_2 = update_centroid(labels,v)
        delta_centroid = abs(old_centroid_1 - centroid_1)
    return labels


def argsort(a): 
    return sorted(range(len(a)), key=a.__getitem__)


#--------------SVD----------(NUMPY ALLOWED!)----------------------


def svd_scratch(A,d):
    import numpy as np
    At_A = np.matmul(A.T,A)
    eigvals_At_A = np.real(np.linalg.eig(At_A)[0])
    eigvecs_At_A = np.linalg.eig(At_A)[1]
    sorted_eigenvecs = eigvecs_At_A[:,np.argsort(-eigvals_At_A)]
    
    sorted_eigenvals = eigvals_At_A[np.argsort(-eigvals_At_A)]
    sorted_eigenvals = np.round(sorted_eigenvals,5)
    singular_values_sorted = np.sqrt(np.real(sorted_eigenvals))
    V = np.real(sorted_eigenvecs)
    e_diag = np.diag(singular_values_sorted)
    S=e_diag
    U=np.zeros([A.shape[0],len(singular_values_sorted)])
    I=np.eye(A.shape[0])
    for i in range(len(S)):
        if singular_values_sorted[i]==0:
            U[:,i] = I[:,i]
        else:
            U[:,i] = np.matmul(A,V[:,i])/singular_values_sorted[i]
   
        
    
    return U,S,V
def reconst(U,S,V):
    import numpy as np
    return np.matmul(U,np.matmul(S,V.T))
    
def rmse(x,x_svd):
    import numpy as np
    
    diff = np.square(x-x_svd)
    diff = np.sum(diff)
    return np.sqrt(diff)
#--------------------------------------------------

#-----------------------PROBLEM DRIVER FUNCTIONS------------------------------------------------------

def problem_1(path):
    f = open(path,'r')
    f_lines = f.readlines()
    A = []
    for i in f_lines:
        x=i.split(" ")
        x = [float(i) for i in x]
        A.append(x)
    print("Input matrix is - ")
    print(A)
    f.close()
    result = eigvals_scratch_qr(A)
    if A==transpose(A):
        for i in range(len(result[0])):	    
            print("Eigen-value = "+str(result[0][i]))
            print("Correspondig eigen-vector = "+str(get_column_k(result[1],i)))
    else:
        print("non-symmetric matrix - eigen vectors may or may not perfectly converge (Demerit of primitive numerical method used -QR Algo)")
        for i in range(len(result[0])):     
            print("Eigen-value = "+str(result[0][i]))
            print("Correspondig eigen-vector = "+str(get_column_k(result[1],i)))


def problem_2(path):
    data = nx.read_gml(path)
    #preserving labels for future use
    labels = {}
    labels_list = list(data.nodes())
    n=len(labels_list)
    for i in range(len(list(data.nodes()))):
        labels[i] = labels_list[i]   
    #extracting adjacency matrix out of graph
    el_classico_matrix = nx.to_numpy_matrix(data)  #NOT USING NUMPY !! ITS THE LIBRARY FUNCTION OF NETWORKX TO EXTRACT ADJACENCY MATRIX
    el_classico_matrix_list = el_classico_matrix.tolist()
    degree_el_classico_matrix = give_zero_matrix(n,n)
    ones = [[1] for i in range(n)]
    degree_el_classico_list = matrix_mul(el_classico_matrix_list,ones)
    degree_el_classico_list = [i[0] for i in degree_el_classico_list]
    for i in range(n):
        degree_el_classico_matrix[i][i] = sum(el_classico_matrix_list[i])
        
    #-----------------1) Degree centrality-------------------------------------
    print("--------------1) Degree Centrality-----------------------------------")
    for key,value in labels.items(): 
        print(str(value)+"="+str(degree_el_classico_list[key])) #check
    
    #-----------------2) Closeness Centrality----------------------------------
    
    #All pair shortest path aux function - Floyd Marshall
    
    em = copy.deepcopy(el_classico_matrix_list)
    INF=999999999999
    for i in range(len(em)):
        for j in range(len(em)):
            if em[i][j]==0:
                em[i][j]=INF
            if i==j:
                em[i][j]=0
    x = floyd_marshall_ap_shortest_paths(em)
    if x:
        print("negative cycles found")
    print("--------------2) Closeness Centrality (lesser the value, closer to others and hence more central)-----------------")
    c=0
    for i in em:
        print(str(labels[c])+"="+str(sum(i)))
        c+=1
    #----------------3)Betweenness Centrality
    
    #----------------4)Eigenvector centrality (may take time since its using scratch eigv algo----------------------------------
    print("--------------4) Eigenvector Centrality (influential neighbhours)-----------------")
    eigenvals = eigvals_scratch_qr(el_classico_matrix_list)[0]
    eigenvecs = eigvals_scratch_qr(el_classico_matrix_list)[1]
    ind = argsort(eigenvals)
    sorted_eigenvecs = []
    for i in ind:
        sorted_eigenvecs.append(get_column_k(eigenvecs,i))
    #eigenvector corresponding to largest eigenvalue
    v = sorted_eigenvecs[len(el_classico_matrix_list)-1]
    for i in range(len(el_classico_matrix_list)):
        print(str(labels[i])+"="+str(v[i][0]))
 
    
    


def problem_3(path):
    data = nx.read_gml(path)
    labels = {}
    labels_list = list(data.nodes())
    n=len(labels_list)
    for i in range(len(list(data.nodes()))):
        labels[i] = labels_list[i]   
    #extracting adjacency matrix out of graph
    el_classico_matrix = nx.to_numpy_matrix(data)  #NOT USING NUMPY !! ITS THE LIBRARY FUNCTION OF NETWORKX TO EXTRACT ADJACENCY MATRIX
    el_classico_matrix_list = el_classico_matrix.tolist()
    degree_el_classico_matrix = give_zero_matrix(n,n)
    ones = [[1] for i in range(n)]
    degree_el_classico_list = matrix_mul(el_classico_matrix_list,ones)
    degree_el_classico_list = [i[0] for i in degree_el_classico_list]
    for i in range(n):
        degree_el_classico_matrix[i][i] = sum(el_classico_matrix_list[i])
    
        #making normalized dolphin laplacian matrix
    identity_matrix = give_me_identity(n)
    for i in range(n):
        degree_el_classico_matrix[i][i] = 1/((degree_el_classico_matrix[i][i])**(1/2))
    
    temp = matrix_mul(degree_el_classico_matrix,el_classico_matrix_list)
    norm_laplacian = matrix_diff(identity_matrix,matrix_mul(temp,degree_el_classico_matrix))
    
    #------EIGEN VALUES, EIGEN VECTORS OF NORMALIZED LAPLACIAN & CALCULATION OF FIEDLER VECTOR
    
    #-------------Eigen vals and eigen vecs calculation using QR algorithm built in part 1--------
    eigenvals = eigvals_scratch_qr(norm_laplacian)[0]
    eigenvecs = eigvals_scratch_qr(norm_laplacian)[1]
    ind = argsort(eigenvals)
    sorted_eigenvecs = []
    for i in ind:
        sorted_eigenvecs.append(get_column_k(eigenvecs,i))
    
    v = sorted_eigenvecs[1]
    v = [v[i][0] for i in range(len(v))]
    labels_1 = kmeans_scratch(v,mean(v))
    colors = [0]*len(labels_1)
    for key,value in labels_1.items():
        colors[key]=value
    colors = ['yellow' if i==1 else 'pink' for i in colors]
    g = nx.from_numpy_matrix(el_classico_matrix)
    plt.figure(1,figsize=(15,7))
    #print(labels_list)
    #plt.title("Barca - Real Madrid clustered (#ValverdeOut) \nTeam 1 = "+str(colors.count('yellow'))+" players\nTeam 2 = "+str(colors.count('pink'))+" players")
    nx.draw(g,with_labels=True,font_weight='bold',font_color='black',node_color=colors,labels=labels,node_size=1190)
    plt.show()
    
def problem_4a(path):
    #NUMPY FOR THIS PROBLEM ONLY
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    data = np.genfromtxt(path,delimiter=',')
    d_list=[2,5,10,20,50,100,150,200,400,500]
    rmse_list=[]
    r = random.randint(50,100)
    x = data[:,1:]
    d=784
    U,S,V = svd_scratch(x,d)  
    for d in d_list:
        U_ = U[:,:d]
        S_ = S[:d,:d]
        V_ = V[:d,:d]
        print("------------------------------------------------------------------------")
        print('for d='+str(d))
        print("Original matrix =")
        print(x.shape)
        print("U=")
        print(U.shape)
        print("S=")
        print(S.shape)
        print("V=")
        print(V.shape)
        x_svd=reconst(U_,S_,V_)
        print(x_svd.shape)
        x_svd_r = np.zeros(x.shape)
        print("Reconstructed matrix=")
        x_svd_r[:x_svd.shape[0],:x_svd.shape[1]]=x_svd
        rmse_list.append(rmse(x,x_svd_r))
    print("------------------------------------------------------------------------------")
    print("rmse for d=2,5,10,20,50,100,150,200,400,500 - ")
    print(rmse_list)
    plt.title("  X-axis: Values of d and Y-axis :RMSEs")
    #plt.ylim(top=1e-7)
    plt.scatter(d_list,rmse_list)
    plt.show()

    

def problem_4b(path):
    import numpy as np
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    data = np.genfromtxt(path,delimiter=',')
    print("fitting tsne (for 3000 samples) please wait...........")
    #plotting 3000 samples
    X=data[:3000,1:]
    y = data[:3000,0]
    X_2d = tsne.fit_transform(X)
    target_ids = range(0,10)
    target_names = np.array([0,1,2,3,4,5,6,7,8,9])
    plt.figure(figsize=(6, 5))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'darkred']
    for i, c, label in zip(target_ids, colors,target_names ):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()

#-----------------------------------------------------------------------------------


if problem=='problem1':
    problem_1(path)
elif problem=='problem2':
    problem_2(path)
elif problem=='problem3':
    problem_3(path)
elif problem=='problem4a':
    problem_4a(path)
elif problem=='problem4b':
    problem_4b(path)
else:
    print('check ypur sys arguements')
    
    
