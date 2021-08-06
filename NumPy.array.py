##installing numpy
##### pip install numpy

### import numpy

import numpy
arr = numpy.array([1, 2, 3, 4, 5])
print(arr)

###NumPy as np

import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)

##NumPy is used to work with arrays. The array object in NumPy is called ndarray,
####We can create a NumPy ndarray object by using the array() function.

arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))

###0-D Arrays

arr = np.array(21)
print(arr)

###1-D Arrays

arr = np.array([1, 2, 3, 4, 5])
print(arr)

###2-D Arrays

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)

##3-D arrays

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr)

###Check Number of Dimensions

a = np.array(21)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

###Higher Dimensional Arrays

arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print('number of dimensions :', arr.ndim)

###NumPy Array Indexing


arr = np.array([1, 2, 3, 4])  
print(arr[0])   ###NumPy arrays start with 0, meaning that the first element has index 0, and the second has index 1

arr = np.array([1, 2, 3, 4])  ###Get the first element from the following array
print(arr[0])

###Access 2-D Arrays

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])  ###Access the 2nd element on 1st dim
print('2nd element on 1st dim: ', arr[0, 1])

###Access 3-D Arrays

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr[0, 1, 2])

###Negative Indexing

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('Last element from 2nd dim: ', arr[1, -1])

###Slicing arrays


arr = np.array([1, 2, 3, 4, 5, 6, 7])  ##Slice elements from index 1 to index 5 from the following array
print(arr[1:5])

arr = np.array([1, 2, 3, 4, 5, 6, 7])  ###Slice elements from the beginning to index 4
print(arr[:4])

###Negative Slicing

arr = np.array([1, 2, 3, 4, 5, 6, 7])   ##Slice from the index 3 from the end to index 1 from the end
print(arr[-3:-1])

arr = np.array([1, 2, 3, 4, 5, 6, 7])   ###Return every other element from the entire array
print(arr[::2])

####Slicing 2-D Arrays

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[1, 1:4])      ###From the second element, slice elements from index 1 to index 4

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[0:2, 2])    ####From both elements, return index 2

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[0:2, 1:4])    ###From both elements, slice index 1 to index 4 this will return a 2-D array

###NumPy Data Types
###NumPy has some extra data types, and refer to data types with one character, like i for integers, u for unsigned integers etc

arr = np.array([1, 2, 3, 4])
print(arr.dtype)   ##data type of an array object

arr = np.array([1, 2, 3, 4], dtype='S')   ##Create an array with data type string
print(arr)
print(arr.dtype)

arr = np.array([1.1, 2.1, 3.1])  ###Change data type from float to integer by using 'i' as parameter value
newarr = arr.astype('i')  ## astype() function creates a copy of the array, and allows you to specify the data type as a parameter
print(newarr)
print(newarr.dtype)

arr = np.array([1, 0, 3])  ##Change data type from integer to boolean
newarr = arr.astype(bool)
print(newarr)
print(newarr.dtype)

###COPY

arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42
print(arr)
print(x)

#####Make a view, change the original array, and display both arrays

arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42
print(arr)
print(x)

arr = np.array([1, 2, 3, 4, 5])  ###Make a view, change the view, and display both array
x = arr.view()
x[0] = 31
print(arr)
print(x)


arr = np.array([1, 2, 3, 4, 5])  ###Print the value of the base attribute to check if an array owns it's data or not
x = arr.copy()
y = arr.view()
print(x.base)
print(y.base)

####NumPy Array Shape

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  ###Print the shape of a 2-D array
print(arr.shape)


arr = np.array([1, 2, 3, 4], ndmin=5)  ###Create an array with 5 dimensions using ndmin
print(arr)
print('shape of array :', arr.shape)

####NumPy Array Reshaping

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  ###Reshape From 1-D to 2-D
newarr = arr.reshape(4, 3)
print(newarr)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  ###Reshape From 1-D to 3-D
newarr = arr.reshape(2, 3, 2)
print(newarr)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])  ###Convert 1D array with 8 elements to 3D array with 2x2 elements
newarr = arr.reshape(2, 2, -1)
print(newarr)

###Flattening the arrays

arr = np.array([[1, 2, 3], [4, 5, 6]])  ###Convert the array into a 1D array
newarr = arr.reshape(-1)
print(newarr)

###NumPy Array Iterating

arr = np.array([1, 2, 3])  ###Iterating means going through elements one by one
for x in arr:
    print(x)

arr = np.array([[1, 2, 3], [4, 5, 6]]) ###Iterating 2-D Arrays
for x in arr:
  print(x)

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]) ###3-D array
for x in arr:
  print(x)

#####NumPy Joining Array

arr1 = np.array([1, 2, 3])   ###joining two array
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)

arr1 = np.array([[1, 2], [3, 4]])  ####join 2-D array
arr2 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr1, arr2), axis=1)
print(arr)

####Joining Arrays Using Stack Functions

arr1 = np.array([1, 2, 3])  ###Stacking is same as concatenation, the only difference is that stacking is done along a new axis
arr2 = np.array([4, 5, 6])
arr = np.stack((arr1, arr2), axis=1)
print(arr)

arr1 = np.array([1, 2, 3])  ###stacking along rows
arr2 = np.array([4, 5, 6])
arr = np.hstack((arr1, arr2))
print(arr)

arr1 = np.array([1, 2, 3])  ###stacking along columns
arr2 = np.array([4, 5, 6])
arr = np.vstack((arr1, arr2))
print(arr)


arr1 = np.array([1, 2, 3])  ###Stacking Along Height (depth)
arr2 = np.array([4, 5, 6])
arr = np.dstack((arr1, arr2))
print(arr)

#### NumPy Splitting Array

arr = np.array([1, 2, 3, 4, 5, 6]) ###spliting array in 3 parts
newarr = np.array_split(arr, 3)
print(newarr)

arr = np.array([1, 2, 3, 4, 5, 6])   ###Split Into Arrays
newarr = np.array_split(arr, 3)
print(newarr[0])
print(newarr[1])
print(newarr[2])

####Split the 2-D array into three 2-D arrays along rows

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3, axis=1)
print(newarr)

####Use the hsplit() method to split the 2-D array into three 2-D arrays along rows

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.hsplit(arr, 3)
print(newarr)


####Searching Arrays

arr = np.array([1, 2, 3, 4, 5, 4, 4])   ####Find the indexes where the value is 4
x = np.where(arr == 4)
print(x)

####Search Sorted

arr = np.array([6, 7, 8, 9])   ###Find the indexes where the value 7 should be inserted
x = np.searchsorted(arr, 7)
print(x)

####Search From the Right Side

arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 7, side='right')
print(x)


arr = np.array([1, 3, 5, 7])   #####Find the indexes where the values 2, 4, and 6 should be inserted
x = np.searchsorted(arr, [2, 4, 6])
print(x)

#####NumPy Sorting Arrays

arr = np.array([2, 3, 0, 1])
print(np.sort(arr))

arr = np.array(['kusum', 'shruti', 'saurav'])  ###sort alphabettically
print(np.sort(arr))

arr = np.array([True, False, True])  ##### sort boolean array
print(np.sort(arr))

####Sorting a 2-D Array

arr = np.array([[4, 2, 3], [5, 0, 1]])
print(np.sort(arr))

#######NumPy Filter Array

arr = np.array([41, 42, 43, 44])    ### Create a filter array that will return only values higher than 41

# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
  # if the element is higher than 41, set the value to True, otherwise False:
  if element > 41:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)

####Random Numbers in NumPy

from numpy import random
x = random.randint(10)   ###Generate a random integer from 0 to 10
print(x)

####Generate Random Array

x=random.randint(10, size=(5))
print(x)

x = random.randint(10, size=(3, 5))  ####Generate a 2-D array with 3 rows, each row containing 5 random integers from 0 to 10
print(x)










