import numpy as np
import pandas as pd
import heapq

path = '/content/recolab-data/collaborative/'

def user_user_collaborative_filtering(P_inv,R):
    recommendation_matrix = np.dot(P_inv,np.dot(R,np.dot(R.T,np.dot(P_inv,R))))

    return recommendation_matrix

def item_item_collaborative_filtering(Q_inv,R):
    recommendation_matrix = np.dot(R,np.dot(Q_inv,np.dot(R.T,np.dot(R,Q_inv))))

    return recommendation_matrix


with open(path+'items.txt') as f:
    items = f.read()
    items = items.split('\n')
    # print(items)
items = items[:-1]
items = np.array(items)
items = np.reshape(items,(items.shape[0],1))

R = []

with open(path+'ratings.txt') as f:
    ratings = f.read().split('\n')
    for row in ratings[:-1]:
        row = row.rstrip().split(" ")
        row = [int(x) for x in row]
        
        R.append(row)

ratings = np.array(R)


print(items.shape)
print(ratings.shape)

row_sum = np.reshape(np.sum(ratings,axis = 1),(ratings.shape[0],1))
col_sum = np.reshape(np.sum(ratings,axis = 0),(ratings.shape[1],1))

P = np.zeros((ratings.shape[0],ratings.shape[0]),dtype = 'int32')
Q = np.zeros((ratings.shape[1],ratings.shape[1]),dtype = 'int32')

for i in range(ratings.shape[0]):
    P[i][i] = int(row_sum[i][0])

for i in range(ratings.shape[1]):
    Q[i][i] = int(col_sum[i][0])

print(P[:5])
print(Q[:5])

P_inv = np.zeros((P.shape[0],P.shape[1]))
Q_inv = np.zeros((Q.shape[0],Q.shape[1]))
for i in range(P.shape[0]):
    if P[i,i] == 0:
        P_inv[i,i] = 0
    else:
        P_inv[i,i] = 1/P[i,i]

for i in range(Q.shape[0]):
    if Q[i,i] == 0:
        Q_inv[i,i] = 0
    else:
        Q_inv[i,i] = 1/Q[i,i]

P_inv = P_inv ** (0.5)
Q_inv = Q_inv ** (0.5)
print(P_inv[:5])
print(Q_inv[:5])


user_recommendation = user_user_collaborative_filtering(P_inv,ratings)

print(user_recommendation[:5])

user_500 = user_recommendation[499,:]
top_100 = heapq.nlargest(100,range(len(user_500)),user_500.take)
# print(top_100)
print('\n\nTop 100 recommendations : \n')
for x in top_100:
    print(items[x])

print('\n\nTop 5 recommendations : \n')
top_5 = heapq.nlargest(5,range(len(user_500)),user_500.take)
for x in top_5:
    print(items[x])


with open(path+'orig.txt') as f:
    original = f.read().split(' ')
    original = [int(x) for x in original]

# print(original)

count = 0
total_count = sum(original)
for x in top_100:
    if(original[x] == 1):
        count += 1

print('Correct predictions in top 100 : ',count)

count = 0
for x in top_5:
    if(original[x] == 1):
        count += 1

print('Correct predictions in top 5 : ',count)


item_recommendation = item_item_collaborative_filtering(Q_inv,ratings)

print(item_recommendation[:5])

user_500 = item_recommendation[499,:]
top_100 = heapq.nlargest(100,range(len(user_500)),user_500.take)
# print(top_100)
print('\n\nTop 100 recommendations : \n')
for x in top_100:
    print(items[x])

print('\n\nTop 5 recommendations : \n')
top_5 = heapq.nlargest(5,range(len(user_500)),user_500.take)
for x in top_5:
    print(items[x])


count = 0
total_count = sum(original)
for x in top_100:
    if(original[x] == 1):
        count += 1


print('Correct predictions in top 100 : ',count)

count = 0
for x in top_5:
    if(original[x] == 1):
        count += 1

print('Correct predictions in top 5 : ',count)






