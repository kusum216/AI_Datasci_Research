####matrices with Numpy

import numpy as np
A = [[1, 4, 5, 12], ######python doesn't have a built-in type for matrices. However, we can treat a list of a list as a matrix.###Py
    [-5, 8, 9, 0],
    [-6, 7, 11, 19]]
print("A =", A) 
print("A[1] =", A[1])      # 2nd row
print("A[1][2] =", A[1][2])   # 3rd element of 2nd row
print("A[0][-1] =", A[0][-1])   # Last element of 1st Row
column = [];        # empty list
for row in A:
  column.append(row[2])   
print("3rd column =", column)

#### MATRIX OPERATIONS

A = np.array([[2, 4], [5, -6]])   #####additon of matrix 
B = np.array([[9, -3], [3, 6]])
C = A + B      
print(C)

A = np.array([[2, 4], [5, -6]])   #####subtraction of matrix 
B = np.array([[9, -3], [3, 6]])
C = A- B
print(C)

A = np.array([[3, 6, 7], [5, -3, 0]])   #####multiplication of matrix 
B = np.array([[1, 1], [2, 1], [3, -3]])
C = A.dot(B)
print(C)

A = np.array([[1, 1], [2, 1], [3, -3]]) ###transpose of matrix
print(A.transpose())

#####Access matrix elements, rows and columns

A = np.array([2, 4, 6, 8, 10])

print("A[0] =", A[0])     # First element     
print("A[2] =", A[2])     # Third element 
print("A[-1] =", A[-1])   # Last element     

####access elements from 2-D

A = np.array([[1, 4, 5, 12],
    [-5, 8, 9, 0],
    [-6, 7, 11, 19]])
print("A[0][0] =", A[0][0])  #  First element of first row
print("A[1][2] =", A[1][2])  #### Third element of second row
print("A[-1][-1] =", A[-1][-1])    ###### Last element of last row 



A = np.array([[1, 4, 5, 12],   #########access rows
    [-5, 8, 9, 0],
    [-6, 7, 11, 19]])
print("A[0] =", A[0]) # First Row
print("A[2] =", A[2]) # Third Row
print("A[-1] =", A[-1]) # Last Row 


A = np.array([[1, 4, 5, 12],   ######acccesss coulm
    [-5, 8, 9, 0],
    [-6, 7, 11, 19]])
print("A[:,0] =",A[:,0]) # First Column
print("A[:,3] =", A[:,3]) # Fourth Column
print("A[:,-1] =", A[:,-1]) # Last Column


###########Slicing of a Matrix

A = np.array([[1, 4, 5, 12, 14], 
    [-5, 8, 9, 0, 17],
    [-6, 7, 11, 19, 21]])

print(A[:2, :4])  # two rows, four columns
print(A[:1,])  # first row, all columns
print(A[:,2])  # all rows, second column
print(A[:, 2:5])  # all rows, third to the fifth column


#####rank of matrix

a = np.arange(1, 10)
a.shape = (3, 3)
print("a = ")    ####output comes in new row
print(a)       
rank = np.linalg.matrix_rank(a)
print(rank)

###### determinnat of matrix

a = np.array([[2, 2, 1],
               [1, 3, 1],
               [1, 2, 2]])
print("a = ")
print(a)
det = np.linalg.det(a)
print( np.round(det))

##### true inverse

a = np.array([[2, 2, 1],
               [1, 3, 1],
               [1, 2, 2]])
print("a = ")
print(a)
det = np.linalg.det(a)
print(np.round(det))
inv = np.linalg.inv(a)
print("Inverse of a = ")
print(inv)


#####pseudo inverse

a = np.array([[2, 8],
               [1, 4]])
print("a = ")
print(a)
det = np.linalg.det(a)
print(np.round(det))
pinv = np.linalg.pinv(a)
print("Pseudo Inverse of a = ")
print(pinv)

#####Flatten is a simple method to transform a matrix into a one-dimensional numpy array. For this, we can use the flatten() method of an ndarray object.

a = np.arange(1, 10)
a.shape = (3, 3)
print("a = ")
print(a)
print("After flattening")
print("------------------")
print(a.flatten())



####3To create an empty matrix of 5 columns and 0 row

A = np.empty((0, 5))
print('The value is :', A)  ##### by default the matrix type is float64
print('The type of matrix is :', A.dtype)

###shape of mtarix

A = np.empty((0, 5))
print('The shape of matrix is :', A.shape)

#### create empty matrix

a=np.matrix(12)
print('The matrix with 12 random values:', a)


####numpy zeros  It returns a new array of given shape and type, filled with zeros

import numpy as np
a = np.zeros((7, 5))
print('The matrix is : \n', a)   # print the matrix
a.dtype   # check the type of matrix










