#!/usr/bin/env python
# coding: utf-8
#Author - Ashish Kumar Jayant
#Title - Assignment 1 : Linear System of Equations



import copy   # for copying array by value not by reference
import sys


#-------------------INPUT---------------------------------------------------
problem = sys.argv[1]
path = sys.argv[2]

f = open(path,'r')
f_lines = f.readlines()
a = []
for i in f_lines:
    x=i.split(" ")
    x = [float(i) for i in x]
    a.append(x)
print("Input matrix is - ")
print(a)
f.close()
#--------------------------------------------------------------------------


#-----------AUXILLARY FUNCTIONS--------------------------------------------
def check_sanity(a,i):
    if a[i][0]!=0:
        return 0,True
    else:
        l = [abs(c[i]) for c in a]
        return l.index(max(l)),l.index(max(l))==i
    
def row_exchange(a,i,j):
    t = a[i]
    a[i] = a[j]
    a[j] = t
    return a

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

def gauss_jordan(a,m,n,E):
    pivot = 0
    i=0
    for j in range(n):
        l = [bool(k) for k in a[i]]
        try:
            pivot = l.index(True)
            for i in range(m):
                if i>j and a[i][j]!=0:  
                    if a[pivot][j]==0:
                        a=row_exchange(a,pivot,i)
                        I=give_me_identity(n)
                        I=row_exchange(I,pivot,i)
                        E.append(I)

                    else:
                        c = a[i][j]/a[pivot][j]    
                        for p in range(n):
                            a[i][p] = a[i][p] - c*a[pivot][p]
                        I=give_me_identity(n)
                        I[i][pivot]=-c
                        I[i][i]=1
                        E.append(I)



        except:
            pass
    return a,E
        
    
def rank(a,m,n):
    c=0;
    for i in range(m):
        for j in range(n):
            if a[i][j]!=0:
                c+=1
                break
    return c
def print_matrix(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            print(str(x[i][j]),end=" ")
        print("\n")
def driver_part_1_task_1(a,part):
    E=[] 
    m = len(a)
    n = len(a[0])
    l,f=check_sanity(a,0)
    if f==False:
        a=row_exchange(a,0,l)
        I=give_me_identity(n)
        I=row_exchange(I,0,l)
        E.append(I)
    a,E = gauss_jordan(a,m,n,E)
    if part==1:
        print("Rank of matrix = "+str(rank(a,m,n)))
        print("----------------------------------------------")
        print("Row Echeleon Form = ")
        print_matrix(a)
        print("----------------------------------------------")
        for i in range(len(E)):
            print("Elementary matrix for row transformation no. "+str(i+1))
            print("\n")
            print_matrix(E[i])
    if part==2:
        return rank(a,m,n),a
def driver_part_2_task_1(a):
    n_unknowns = len(a[0])-1
    n_equations = len(a)
    augmented = copy.deepcopy(a)
    b = copy.deepcopy(a)
    for i in range(n_equations):
        b[i].pop();
    rank_augmented,echeleon_augmented=driver_part_1_task_1(augmented,2)
    rank_b,echeleon_b = driver_part_1_task_1(b,2)
    if rank_b<rank_augmented:
        print("NO SOLUTION EXISTS")
    elif rank_b==rank_augmented and rank_b==n_unknowns:
        print("UNIQUE SOLUTION EXISTS")
        x={}
        for i in range(n_equations-1,-1,-1):
            s=0
            for j in range(n_unknowns,-1,-1):
                if i<j:
                    if j==n_unknowns:
                        s=s+echeleon_augmented[i][j]
                    else:
                        s=s-x[j]*echeleon_augmented[i][j]

            x[i]=s/echeleon_augmented[i][i]
            
                    
        for i in x.keys():
            print("x_"+str(i)+"="+str(x[i]))
    else:
        print("MANY SOLUTION EXISTS")
        
        if len(echeleon_augmented)+1 == len(echeleon_augmented[0]):
            
            print("Taking values of free variables = 1")
            free=[]
            x={}
            for i in range(n_unknowns-1,-1,-1):
                s=0
                for j in range(n_unknowns,-1,-1):
                    if i<j:
                        if echeleon_augmented[i][j]==0:
                            x[i]=1
                            free.append(i)
                        if j==n_unknowns:
                            s=s+echeleon_augmented[i][j]
                        else:
                            s=s-x[j]*echeleon_augmented[i][j]
                if i not in x.keys():
                    x[i]=s/echeleon_augmented[i][i]
            free = set(free)
            print("free variables are - ")
            for i in free:    
                print("x_"+str(i)+",",end=" ")
            print("\n")
            print("one of the solution - ")
            for i in x.keys():
                print("x_"+str(i)+"="+str(x[i]))
        else:
            print("Works for n*(n+1) case only")
            

#-------------------RUN------------------------------------

if problem=='problem1':
	driver_part_1_task_1(a,1)
elif problem=='problem2':
	driver_part_2_task_1(a)
else:
    print("please check your command line arguements - don't give path or filename in quotes")



