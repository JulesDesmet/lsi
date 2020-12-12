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



