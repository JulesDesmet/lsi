#!/usr/bin/env python

from term_doc_matrix import TfIdf
from math import sqrt
import numpy as np
from scipy import sparse
from threading import Thread

import threading
class SVD:
    tfidf: TfIdf
    def __init__(self, tfidf_matrix: "TfIdf" ):
        self.term_matrix: list[dict] = []
        self.document_matrix: list[dict] = []
        self.singular_values: list[dict] = []
        self.tfidf = tfidf_matrix
        self.eigenvalues = []
        self.eigenvectors = []

    """
    Function that turns the tfidf object into a numpy matrix
    """
    def create_numpy_matrices(self):
        a = np.zeros(shape=(len(self.tfidf.terms),self.tfidf.nr_documents))
        for i in range(len(self.tfidf.terms)):
            for j in range(self.tfidf.nr_documents):
                a[i,j] = self.tfidf(j,i)
        return a

    """
    Turns matrix in sparse matrix
    """
    def turn_sparse(self, matrix):
        return sparse.csr_matrix(matrix)

    """Matrix multiplication with no numpy libraries"""
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
                """                for k in range(self.tfidf.nr_documents):
                    print("I:"+str(i) +" K:" +str(k) + "J: " +str(j))
                    sum += self.tfidf(k,i) * self.tfidf(k,j)
                    """

                temp[j] = sum
            result.append(temp)
        print(result)


    """Dot multiplication with no numpy libraries and with skipping 0 values"""
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


    """Matrix multiplication in numpy, should not be used"""
    def matrix_multiplication_numpy(self, matrixA, matrixB):
        ra, ca = matrixA.shape
        rb, cb = matrixB.shape
        assert ca == rb, f"{ca} != {rb}"
        output = np.zeros(shape=(ra,cb))
        for i in range(ra):
            output[i] = np.dot(matrixA[i],matrixB)
        return output
        #print(np.matmul(matrixA, matrixB))


    """Multiplication of two sparse matrices"""
    def matrix_sparse_multiplication(self, matrixA, matrixB):
        return sparse.csr_matrix.dot(matrixA, matrixB)

    """Power transformation of sparse matrix"""
    def power_transformation_sparse(self, matrix, tolerable_error):
        row, col = matrix.shape
        # Initial guess vector, fill it (col, 1)
        x = np.ones(col)
        # When max iteration is reached convergence ends, value can be adjusted
        max_iteration = 50
        # Starting value
        lambda_old = 1.0
        cond = True
        step = 1
        while cond:
            # Multiply M * x
            x_guess = x
            x = self.sparse_array_multiplication(matrix, x)
            # Multiply MT * M * x
            #x = matrix.transpose() @ x
            x = sparse.csr_matrix.dot(matrix.transpose(), x)
            """
            for i in eigenvalues
                x - (lambda * eigenvec[i]) * (eigenvec[i].transpose*x_guess
            """
            for i in range(len(self.eigenvalues)):
                prev_x_t = np.reshape(self.eigenvectors[i], (1, col))
                prev_x = np.reshape(self.eigenvectors[i], (col, 1))
                right1 = self.eigenvalues[i]*prev_x
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

        self.eigenvalues.append(lambda_old)
        self.eigenvectors.append(x)
        #print("CURRENT LENGTH {}".format(len(self.eigenvalues)))



    """Method to get all eigenvalues from the power iteration, still needs to be worked on"""
    def get_eigenvalues(self, matrix):
        row, col = matrix.shape
        new_matrix = matrix
        for i in range(row):
            eigval, eigvec = self.power_transformation_sparse(new_matrix, 0.0001)
            self.eigenvalues.append(eigval)
            self.eigenvectors.append(eigvec)
            xt = np.reshape(eigvec, (1, row))
            x = np.reshape(eigvec, (row, 1))
            xxt = x @ xt
            print(xxt)
            new_matrix = new_matrix - (xxt*(eigval/abs(x)**2))
            print(eigval)
            print("THIS IS THE MAX VALUE")
            print(np.amax(new_matrix))
            print("THIS IS THE MAX VALUE")
            new_matrix = self.turn_sparse(new_matrix)
        print(self.eigenvalues)

    """Matrix * vector using Python3.5+ matrix multiplication"""
    def sparse_array_multiplication(self, sparse, guess):
        return sparse @ guess


    """Test class to check new power iteration making use of associative method"""
    def new_idea(self, matrix):
        row, col = matrix.shape
        matrix_t = matrix.transpose()
        print("WE SHOULD RECEIVE AN EIGENVECTOR OF {} rows".format(row))
        for i in range(row):
            self.power_transformation_sparse(matrix_t, 0.0001)

    """ 
    def eigenval_MMT(self, matrix):
        
        row, col = matrix.shape
        matrix_t = matrix.transpose()
        for i in range(row):
            self.power_transformation_sparse(matrix_t, 0.0001)
            self.eigenvalues.append(eigval)
            self.eigenvectors.append(eigvec)
            
            xt = np.reshape(eigvec, (1, row))
            x = np.reshape(eigvec, (row, 1))
            
            xxt = x @ xt
            print(xxt.shape)
            print(matrix.shape)
           # new_matrix = matrix - (xxt * (eigval / abs(x) ** 2))
            next_iter = matrix - (eigval*xxt)
            print(next_iter.shape)
            break"""


