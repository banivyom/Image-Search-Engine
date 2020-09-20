from scipy.spatial.distance import hamming, cosine, euclidean
import numpy as np
import cv2

def topIDs(model, train_vectors, img_path,top_n=20):

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32), cv2.INTER_CUBIC)

    image = image.reshape(-1,32,32,3)
    query_vector = model.predict(image)

    distances = []
    
    for i in range(len(train_vectors)):
        distances.append(cosine(train_vectors[i], query_vector))
        
    return np.argsort(distances)[:top_n]