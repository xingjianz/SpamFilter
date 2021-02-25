import os
import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer


def create_test_set(ham, spam, ratio_test):
    ham_num_test = math.floor(len(ham) * ratio_test)
    spam_num_test = math.floor(len(spam) * ratio_test)
    test_set = []
    for j in range(ham_num_test):
        test_set.append(ham.pop(random.randint(0, len(ham) - 1)))

    for j in range(spam_num_test):
        test_set.append(spam.pop(random.randint(0, len(spam) - 1)))

    return test_set


def bayes(ham, spam, test):
    total = len(ham) + len(spam)
    p_ham = len(ham) / total
    p_spam = 1 - p_ham
    v = len(ham[0][0])

    ham_words = np.zeros(v, dtype=float)
    for h in ham:
        ham_words = ham_words + h[0]
    ham_sum = np.sum(ham_words)
    for k in range(len(ham_words)):
        ham_words[k] = (ham_words[k] + 1) / (ham_sum + v)

    spam_words = np.zeros(v, dtype=float)
    for s in spam:
        spam_words = spam_words + s[0]
    spam_sum = np.sum(spam_words)
    for k in range(len(spam_words)):
        spam_words[k] = (spam_words[k] + 1) / (spam_sum + v)

    # begin testing
    corrects = 0
    for d in test:
        p_h = p_ham
        p_s = p_spam
        for k in range(v):
            count = d[0][k]
            if count == 0:
                continue
            p_h = p_h * (pow(ham_words[k], count))
            p_s = p_s * (pow(spam_words[k], count))

        r = 1
        if p_h >= p_s:
            r = 0
        if r == d[1]:
            corrects = corrects + 1
    rate = corrects / len(test)
    return rate


def knn(ham, spam, test, k):
    corrects = 0
    for d in test:
        nn = []
        for h in ham:
            dis = np.linalg.norm(d[0] - h[0])
            nn.append((dis, 0))
        for s in spam:
            dis = np.linalg.norm(d[0] - s[0])
            nn.append((dis, 1))
        nn.sort()
        k_nn = nn[:k]
        r = 0
        for n in k_nn:
            r = r + n[1]
        r = r / k
        if r >= 0.5:
            r = 1
        else:
            r = 0
        if r == d[1]:
            corrects = corrects + 1
    rate = corrects / len(test)
    return rate


porter = PorterStemmer()
dictionary = {}
i = 0
ham_count = 0
spam_count = 0

p = os.getcwd() + '/enron1'
p1 = p + '/ham'
for filename in os.listdir(p1):
    if '.txt' not in filename:
        continue
    with open(os.path.join(p1, filename), 'r', errors='ignore') as f:
        ham_count = ham_count + 1
        for line in f:
            for word in line.split():
                if word.isalpha():
                    w = porter.stem(word)
                    if w not in dictionary:
                        dictionary[w] = i
                        i = i + 1

p1 = p + '/spam'

for filename in os.listdir(p1):
    if '.txt' not in filename:
        continue
    with open(os.path.join(p1, filename), 'r', errors='ignore') as f:
        spam_count = spam_count + 1
        for line in f:
            for word in line.split():
                if word.isalpha():
                    w = porter.stem(word)
                    if w not in dictionary:
                        dictionary[w] = i
                        i = i + 1

ham_data = []
p1 = p + '/ham'
for filename in os.listdir(p1):
    with open(os.path.join(p1, filename), 'r', errors='ignore') as f:
        d = np.zeros(i, dtype=int)
        for line in f:
            for word in line.split():
                if word.isalpha():
                    w = porter.stem(word)
                    index = dictionary[w]
                    d[index] = d[index] + 1
        ham_data.append((d, 0))

spam_data = []
p1 = p + '/spam'
for filename in os.listdir(p1):
    with open(os.path.join(p1, filename), 'r', errors='ignore') as f:
        d = np.zeros(i, dtype=int)
        for line in f:
            for word in line.split():
                if word.isalpha():
                    w = porter.stem(word)
                    index = dictionary[w]
                    d[index] = d[index] + 1
        spam_data.append((d, 1))
# preprocessing finished

bayes_rate = 0
knn_rate = 0
bayes_rates = []
knn_rates = []
trials = 1
for j in range(trials):
    h_ = copy.copy(ham_data)
    s_ = copy.copy(spam_data)
    k = 1
    tests = create_test_set(h_, s_, 0.01)
    b = bayes(h_, s_, tests)
    n = knn(h_, s_, tests, k)
    bayes_rates.append(b)
    knn_rates.append(n)
    bayes_rate = bayes_rate + b
    knn_rate = knn_rate + n


print(bayes_rate / trials)
print(knn_rate / trials)

# with open("results.txt", "w") as output:
#     output.write(str(bayes_rates))
#     output.write(str(knn_rates))

x = range(trials)

# plt.plot(x, bayes_rates)
# plt.plot(x, knn_rates)
# plt.legend(["Bayes'", "k-NN"])
# plt.show()
