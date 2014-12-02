import numpy as np
from numpy.linalg import inv

def train():
    print("enter m and n")
    m=int(input())
    n=int(input())

    x=np.zeros((m,n))
    i=0
    for i in range(m):#taking the x matrix input
        j=0
        for j in range(n):
            k=int(input())
            x[i,j]=k

    y=np.zeros((m,1))
    i=0
    for i in range(m):
        k=int(input())
        y[i,0]=k
    temp=(np.transpose(x)).dot(y)
    temp2=np.matrix(np.transpose(x))*np.matrix(x)
    temp2=inv(np.matrix(temp2))
    theta=np.matrix(temp2)*np.matrix(temp)
    #print(temp)
    print(theta)
    return theta
while True:
      print("1.Train")
      print("2.Output")
      choice=int(input())
      if choice==1 :
         thta=train()
      else :
          n=int(input("Enter n\n"))
          x1=np.zeros((n,1))
          i=0
          for i in range(n):
              k=int(input())
              x1[i,0]=k
          hypo=(np.transpose(thta)).dot(x1)
          print("For given parameters hypothesis-")
          print(hypo)

