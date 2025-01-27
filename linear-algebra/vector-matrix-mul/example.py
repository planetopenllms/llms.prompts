def matrix_vector_multiplication(A, v):
    # Check if the matrix and vector dimensions are compatible
    if len(A[0]) != len(v):
        raise ValueError("Number of columns in A must equal number of rows in v")
    
    result = []
    for i in range(len(A)):  # iterate over each row of the matrix
        row_result = 0
        for j in range(len(A[0])):  # iterate over columns of the matrix
            row_result += A[i][j] * v[j]  # sum of products
        result.append(row_result)
    
    return result

def vector_matrix_multiplication(v, A):
    # Check if the vector and matrix dimensions are compatible
    if len(v) != len(A):
        raise ValueError("Number of elements in v must equal number of rows in A")
    
    result = []
    for j in range(len(A[0])):  # iterate over each column of the matrix
        col_result = 0
        for i in range(len(v)):  # iterate over the vector and rows of the matrix
            col_result += v[i] * A[i][j]  # sum of products
        result.append(col_result)
    
    return result


### Example Usage:

# Example matrix and vectors
A = [[1, 2], 
     [3, 4], 
     [5, 6]]  # 3x2 matrix
v = [1, 
     2]  # 2x1 vector (column)

# Matrix-Vector multiplication (result will be a 3x1 vector)
result_matrix_vector = matrix_vector_multiplication(A, v)
print("Matrix-Vector Result:", result_matrix_vector)

# Vector-Matrix multiplication (result will be a 1x3 vector)
v_row = [1, 2]  # Row vector (1x2)
A_matrix = [[1, 3, 5], 
            [2, 4, 6]]  # 2x3 matrix
result_vector_matrix = vector_matrix_multiplication(v_row, A_matrix)
print("Vector-Matrix Result:", result_vector_matrix)


## => Matrix-Vector Result: [5, 11, 17]
## => Vector-Matrix Result: [5, 11, 17]

