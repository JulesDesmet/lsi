#!/usr/bin/env python

from term_doc_matrix import TfIdf
from math import sqrt
import numpy as np
import string
from scipy import sparse
from threading import Thread

import threading
class SVD:
    tfidf: TfIdf
    def __init__(self, tfidf_matrix: "TfIdf" , k):
        self.term_matrix: list[dict] = []
        self.document_matrix: list[dict] = []
        self.singular_values: list[dict] = []
        self.tfidf = tfidf_matrix
        self.eigenvaluesMMT = []
        self.eigenvaluesMTM = []
        self.eigenvectorsU = []
        self.eigenvectorsV = []
        self.k: int = k
    """
    Function that turns the tfidf object into a numpy matrix
    """
    def create_numpy_matrices(self):
        #a = np.zeros(shape=(len(self.tfidf.terms),self.tfidf.nr_documents))
        a = sparse.lil_matrix((len(self.tfidf.terms), self.tfidf.nr_documents))
        for i in range(len(self.tfidf.terms)):
            for j in self.tfidf.docs_per_term[i]:
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
        if version == "MTM":
            eigenvalues = self.eigenvaluesMTM
            eigenvectors = self.eigenvectorsV
        elif version == "MMT":
            eigenvalues = self.eigenvaluesMMT
            eigenvectors = self.eigenvectorsU
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
            self.eigenvaluesMTM.append(lambda_old)
        elif version == "MMT":
            self.eigenvectorsU.append(x)
            self.eigenvaluesMMT.append(lambda_old)



    """Matrix * vector using Python3.5+ matrix multiplication"""
    def sparse_array_multiplication(self, sparse, guess):
        return sparse @ guess


    """Test class to check new power iteration making use of associative method"""
    def calculate_eigenvalues(self, matrix):
        """
        :param matrix: the starting matrix of which we calculate the eigenvalues
        :param k: the amount of singular values we want
        :return:
        """
        row, col = matrix.shape
        matrix_t = matrix.transpose()
        for i in range(self.k):
            self.power_transformation_sparse(matrix, 0.0001, "MTM")
            self.power_transformation_sparse(matrix_t, 0.0001, "MMT")
            print(f"{i:3} {self.eigenvalues[-1]} {self.eigenvectors[-1]}"

    def calculate_sigma(self):
        for i in range(len(self.eigenvaluesMTM)):
            self.eigenvaluesMTM[i] = sqrt(self.eigenvaluesMTM[i])
                


