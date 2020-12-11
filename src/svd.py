#!/usr/bin/env python

from term_doc_matrix import TfIdf
from math import sqrt
from threading import Thread

import threading
class SVD:
    tfidf: TfIdf
    def __init__(self, tfidf_matrix: "TfIdf" ):
        self.term_matrix: list[dict] = []
        self.document_matrix: list[dict] = []
        self.singular_values: list[dict] = []
        self.tfidf = tfidf_matrix

    def calculate_eigen_values(self):
        print(self.tfidf(1,10))
    def power_transformation(self, max_error: float):

        x0 = dict.fromkeys(range(len(self.tfidf.terms)),1)
        #print(x0)
        result: list[dict] = []
        temp: dict = {}
        for i in range(len(self.tfidf.terms)):
            for j in range(1):
                sum = 0
                for k in range(self.tfidf.nr_documents):
                    sum = sum + self.tfidf(i, k) * x0.get(k)
                temp[i] = sum
        frob = self.frobenius_norm(temp)
        for i in range(len(temp)):
            temp[i] = temp[i]/frob
        result.append(temp)
        print(result)
    def frobenius_norm(self, x1:dict):
        result = 0.0
        for i in range(len(x1)):
            result += x1.get(i) ** 2
        result = sqrt(result)
        return result

    def multiply_M_MT(self):
        result: list[dict] = []
        print(len(self.tfidf.terms))
        for i in range(len(self.tfidf.terms)):
            temp: dict = {}
            for j in range(len(self.tfidf.terms)):
                sum = 0
                for k in range(self.tfidf.nr_documents):
                    sum = sum + self.tfidf(i,k) * self.tfidf(j,k)
                temp[j] = sum
            result.append(temp)
        print(result)


    def matrix_multiplication(self):
        self.tfidf.optimise()
        # This is a M * M_transpose multiplication
        result: list[dict[int, float]] =[]
        # rows of matrix M
        for i in range(len(self.tfidf.terms)):
            # columns of matrix M_T (which will be rows of matrix M)
            for j in range(len(self.tfidf.terms)):
                # rows of matrix M_T
                for k in range(self.tfidf.nr_documents):
                    result[i][j] += self.tfidf(i,k) * self.tfidf(j,k)
        return result
