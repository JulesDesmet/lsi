#!/usr/bin/env python

from term_doc_matrix import TfIdf
from math import sqrt
import numpy as np
from scipy import sparse


class SVD:
    tfidf: TfIdf

    def __init__(self, tfidf_matrix: "TfIdf" , k):
        """
        Initialize the class SVD
        :param tfidf_matrix: the TF-IDF matrix with tf idf score values
        :param k: K is the rank we want to reduce our dimensionality too
        """
        self.tfidf = tfidf_matrix
        # Stores the eigen values for M*MT
        self.eigenvaluesMMT = []
        # Stores the eigen values for MT * M
        self.eigenvaluesMTM = []
        # Stores the eigenvectors of M*MT which equals the matrix U in the SVD decomposition
        self.eigenvectorsU = []
        #  Stores the eigenvectors of MT*M which equals the matrix V in the SVD decomposition
        self.eigenvectorsV = []
        # Used to store the numpy matrix version of U and V, used for querying
        self.u = np.zeros(shape=(len(self.tfidf.terms),k))
        # holds v transpose not v
        self.v = np.zeros(shape=(k, self.tfidf.nr_documents))
        self.k: int = k

    @property
    def create_numpy_matrices(self):
        a = sparse.lil_matrix((len(self.tfidf.terms), self.tfidf.nr_documents))
        """
        Generates numpy matrices from the TF-idf object
        :return: Returns the generated matrix
        """
        for i in range(len(self.tfidf.terms)):
            for j in self.tfidf.docs_per_term[i]:
                a[i,j] = self.tfidf(j,i)
        return a

    def turn_sparse(self, matrix):
        """
        function to turn dense matrices into sparse matrices
        :param matrix: Takes a numpy array(matrix)
        :return: Sparse matrix
        """
        return sparse.csr_matrix(matrix)

    """
    # This function was an attempt at an efficient way to do matrix multiplication without libraries
    def matrix_multiplication(self):
        # This is a M * M_transpose multiplication
        result: list[dict[int, float]] =[]
        # rows of matrix M
        for i in range(self.tfidf.nr_documents):
            temp: dict = {}
            # columns of matrix M_T (which will be rows of matrix M)
            for j in range(self.tfidf.nr_documents):
                sum = self.efficient_dot(j,i)

                # rows of matrix M_T
                              for k in range(self.tfidf.nr_documents):
                    print("I:"+str(i) +" K:" +str(k) + "J: " +str(j))
                    sum += self.tfidf(k,i) * self.tfidf(k,j)
                  

                temp[j] = sum
            result.append(temp)
        print(result)
    """

    """
    # Attempt at a dot multiplication used in matrix multiplication without numpy libraries and by skipping 0 values.
    def efficient_dot(self, index: int, index2: int):
        sum = 0
        #print(len(self.tfidf.tfidf_scores[index]))
        if len(self.tfidf.tfidf_scores[index]) >= len(self.tfidf.tfidf_scores[index2]):
            for key in self.tfidf.tfidf_scores[index]:
                if key in self.tfidf.tfidf_scores[index2]:
                    sum += self.tfidf.tfidf_scores[index].get(key)*self.tfidf.tfidf_scores[index2].get(key)
        elif len(self.tfidf.tfidf_scores[index]) < len(self.tfidf.tfidf_scores[index2]):
            for key in self.tfidf.tfidf_scores[index2]:
                if key in self.tfidf.tfidf_scores[index]:
                    sum += self.tfidf.tfidf_scores[index].get(key)*self.tfidf.tfidf_scores[index2].get(key)
        return sum
    """

    """Matrix multiplication in numpy, should not be used"""
    def matrix_multiplication_numpy(self, matrixA, matrixB):
        ra, ca = matrixA.shape
        rb, cb = matrixB.shape
        assert ca == rb, f"{ca} != {rb}"
        output = np.zeros(shape=(ra,cb))
        for i in range(ra):
            output[i] = np.dot(matrixA[i],matrixB)
        return output

    """Multiplication of two sparse matrices"""
    def matrix_sparse_multiplication(self, matrixA, matrixB):
        """
        Takes two sparse matrices and multiplies them using the scipy sparse matrix dot multiplication
        :param matrixA: sparse matrix
        :param matrixB: sparse matrix
        :return: sparse matrix
        """
        return sparse.csr_matrix.dot(matrixA, matrixB)

    """Power transformation of sparse matrix"""
    def power_transformation_sparse(self, matrix, tolerable_error, version):
        """

        :param matrix: Matrix M you want to power inverse by
        :param tolerable_error: The tolerable error
        :param version: MMT (M*MT) or MTM (MT*M)
        :return:
        """
        row, col = matrix.shape
        # Initial guess vector, fill it (col, 1)
        x = np.ones(col)
        # When max iteration is reached convergence ends, value can be adjusted
        max_iteration = 50
        # Starting value
        lambda_old = 1.0
        cond = True
        step = 1
        eigenvalues = []
        eigenvectors = []
        # Depending on it being a M * MT or MT * M multiplication we fill in different arrays
        if version == "MTM":
            eigenvalues = self.eigenvaluesMTM
            eigenvectors = self.eigenvectorsV
        elif version == "MMT":
            eigenvalues = self.eigenvaluesMMT
            eigenvectors = self.eigenvectorsU
        while cond:
            x_guess = x
            # Multiply M * x
            x = self.sparse_array_multiplication(matrix, x)
            # Multiply MT * M * x
            x = sparse.csr_matrix.dot(matrix.transpose(), x)
            # Iterate over all eigenvalues and subtract lambda*x*xt from the original matrix for each eigenvalue
            for i in range(len(eigenvalues)):
                prev_x_t = np.reshape(eigenvectors[i], (1, col))
                prev_x = np.reshape(eigenvectors[i], (col, 1))
                right1 = eigenvalues[i]*prev_x
                right2 = prev_x_t@x_guess
                x = x - right1@right2
            lambda_new = np.linalg.norm(x)
            # Get new x divided by eigenvalue
            x = x/lambda_new # change to frobenius norm
            step = step + 1
            if step > max_iteration:
                break
            error = abs(lambda_new-lambda_old)
            lambda_old=lambda_new
            # Check if error is met, if so then we stop convergence
            cond = (error > tolerable_error)
        if version == "MTM":
            self.eigenvectorsV.append(x)
            self.eigenvaluesMTM = np.append(self.eigenvaluesMTM,lambda_old)
        elif version == "MMT":
            self.eigenvectorsU.append(x)
            self.eigenvaluesMMT= np.append(self.eigenvaluesMMT, lambda_old)

    def sparse_array_multiplication(self, sparse, guess):
        """
        Matrix * vector multiplication using Python3.5 matrix multiplication @
        :param sparse: sparse matrix
        :param guess: a vector
        :return:
        """
        return sparse @ guess


    """Test class to check new power iteration making use of associative method"""
    def calculate_eigenvalues(self, matrix):
        """

        :param matrix: the starting matrix of which we calculate the eigenvalues
        :param k: the amount of singular values we want, the rank
        :return: Fills in the eigenvectors and eigenvalues
        """
        matrix_t = matrix.transpose()
        for i in range(self.k):
            self.power_transformation_sparse(matrix, 0.0001, "MTM")
            self.power_transformation_sparse(matrix_t, 0.0001, "MMT")

    def calculate_sigma(self):
        """
        Eigenvalues computed in calculate_eigenvalues need to be square rooted to get sigma for our SVD, we also
        immediately put this in a diagonal matrix for query purposes
        :return:
        """
        for i in range(len(self.eigenvaluesMTM)):
            self.eigenvaluesMTM[i] = sqrt(self.eigenvaluesMTM[i])
        self.eigenvaluesMTM = np.diag(self.eigenvaluesMTM)

    def convert_uv_numpy(self):
        """
        Converting the eigenvector arrays to numpy arrays for query purposes
        :return:
        """
        u = np.zeros(shape=(len(self.eigenvectorsU[0]),len(self.eigenvectorsU)))
        v = np.zeros(shape=(len(self.eigenvectorsV), len(self.eigenvectorsV[0])))
        for i in range(len(self.eigenvectorsU[0])):
            for j in range(len(self.eigenvectorsU)):
                u[i,j] = self.eigenvectorsU[j][i]
        for i in range(len(self.eigenvectorsV)):
            for j in range(len(self.eigenvectorsV[0])):
                v[i,j] = self.eigenvectorsV[i][j]
        self.u = u
        self.v = v

    def return_query(self, query):
        """
        Gives the result for a query multiplied by the matrix, we use the components of the SVD decomposition to do this more efficiently
        :param query:
        :return: result of M*query
        """
        # Calculate V_T * x (our query)
        first = self.v @ query
        # Calculate U * SIGMA
        second = self.u @ self.eigenvaluesMTM
        # Computes M * x
        result = first @ second
        return result

    def return_query_document(self, query):
        """
        Maps representation of query in document space
        :param query:
        :return:
        """
        # Returns the representation of the query in document space
        v_trans = self.v.transpose()
        result = query @ self.v
        result = result @ v_trans
        return result



