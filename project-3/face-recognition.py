# https://github.com/ngoahamos/eigenfaces
# https://www.youtube.com/watch?v=8BTv-KZ2Bh8
# https://github.com/alaminanik/PCA-Based-Face-Recognition
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_DIR = 'dataset/LFW_Test' # 'training-images'
TEST_MAGES_DIR = 'dataset/CelebA_Train'
DEFAULT_SIZE = [250, 250] 

def read_images(image_path=IMAGE_DIR, default_size=DEFAULT_SIZE):
    images = []
    images_names = []
    image_dirs = [image for image in os.listdir(image_path) if not image.startswith('.')]
    for image_dir in image_dirs:
        dir_path = os.path.join(image_path, image_dir)
        image_names = [image for image in os.listdir(dir_path) if not image.startswith('.')]
        for image_name in image_names:
            image = Image.open (os.path.join(dir_path, image_name))
            image = image.convert ("L")
            # resize to given size (if given )
            if (default_size is not None ):
                image = image.resize (default_size , Image.LANCZOS )
            images.append(np.asarray (image , dtype =np. uint8 ))
            images_names.append(image_dir)
    return [images,images_names]


def as_row_matrix (X):
    if len (X) == 0:
        return np. array ([])
    mat = np. empty ((0 , X [0].size ), dtype =X [0]. dtype )
    for row in X:
        mat = np.vstack(( mat , np.asarray( row ).reshape(1 , -1))) # 1 x r*c 
    return mat

def get_number_of_components_to_preserve_variance(eigenvalues, variance=.95):
    for ii, eigen_value_cumsum in enumerate(np.cumsum(eigenvalues) / np.sum(eigenvalues)):
        if eigen_value_cumsum > variance:
            return ii
def pca (X, y, num_components =0):
    [n,d] = X.shape
    if ( num_components <= 0) or ( num_components >n):
        num_components = n
        mu = X.mean( axis =0)
        X = X - mu
    if n>d:
        C = np.dot(X.T,X) # Covariance Matrix
        [ eigenvalues , eigenvectors ] = np.linalg.eigh(C)
    else :
        C = np.dot (X,X.T) # Covariance Matrix
        [ eigenvalues , eigenvectors ] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors )
        for i in range (n):
            eigenvectors [:,i] = eigenvectors [:,i]/ np.linalg.norm( eigenvectors [:,i])
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort (- eigenvalues )
    eigenvalues = eigenvalues [idx ]
    eigenvectors = eigenvectors [:, idx ]
    num_components = get_number_of_components_to_preserve_variance(eigenvalues)
    # select only num_components
    eigenvalues = eigenvalues [0: num_components ].copy ()
    eigenvectors = eigenvectors [: ,0: num_components ].copy ()
    return [ eigenvalues , eigenvectors , mu]  

def subplot ( title , images , rows , cols , sptitle ="", sptitles =[] , colormap = plt.cm.gray, filename = None, figsize = (10, 10) ):
    fig = plt.figure(figsize = figsize)
    # main title
    fig.text (.5 , .95 , title , horizontalalignment ="center")
    for i in range ( len ( images )):
        ax0 = fig.add_subplot( rows , cols ,( i +1))
        plt.setp ( ax0.get_xticklabels() , visible = False )
        plt.setp ( ax0.get_yticklabels() , visible = False )
        if len ( sptitles ) == len ( images ):
            plt.title("%s #%s" % ( sptitle , str ( sptitles [i ]) )  )
        else:
            plt.title("%s #%d" % ( sptitle , (i +1) )  )
        plt.imshow(np.asarray(images[i]) , cmap = colormap )
    if filename is None :
        plt.show()
    else:
        fig.savefig( filename )

def get_eigen_value_distribution(eigenvectors):
    return np.cumsum(eigenvectors) / np.sum(eigenvectors)

def plot_eigen_value_distribution(eigenvectors, interval):
    plt.scatter(interval, get_eigen_value_distribution(eigenvectors)[interval])

def project (W , X , mu):
    return np.dot (X - mu , W)
def reconstruct (W , Y , mu) :
    return np.dot (Y , W.T) + mu


def dist_metric(p,q):
    p = np.asarray(p).flatten()
    q = np.asarray (q).flatten()
    return np.sqrt (np.sum (np. power ((p-q) ,2)))

def predict (W, mu , projections, y, X):
    minDist = float("inf")
    minClass = -1
    Q = project (W, X.reshape (1 , -1) , mu)
    for i in range (len(projections)):
        dist = dist_metric( projections[i], Q)
        if dist < minDist:
            minDist = dist
            minClass = i
    return minClass

def save_predictions(eigenvectors,projections, mean, image_names, test_images_dir):
    images_list = [image for image in os.listdir(test_images_dir) if not image.startswith('.')]
    counter = 0;
    for image_path in images_list:
        print(os.path.join(test_images_dir, image_path));
        image = Image.open(os.path.join(test_images_dir, image_path))
        image = image.convert ("L")
        if (DEFAULT_SIZE is not None ):
            image = image.resize (DEFAULT_SIZE , Image.LANCZOS )

        test_image = np. asarray (image , dtype =np. uint8 )
        predicted = predict(eigenvectors, mean , projections, image_names, test_image)
        print(image_names[predicted])
        counter+=1
        if counter == 10:
            exit(0)


[X, y] = read_images()      
average_weight_matrix = np.reshape(as_row_matrix(X).mean( axis =0), X[0].shape)
plt.imshow(average_weight_matrix, cmap=plt.cm.gray)
plt.title("Mean Face")
plt.show()

[eigenvalues, eigenvectors, mean] = pca (as_row_matrix(X), y)

E = []
number = eigenvectors.shape[1]
for i in range (min(number, 16)):
    e = eigenvectors[:,i].reshape(X[0].shape )
    E.append(np.asarray(e))
# plot them and store the plot to " python_eigenfaces.png"
subplot ( title ="Eigenfaces", images=E, rows =4, cols =4, colormap =plt.cm.gray ) #filename ="python_pca_eigenfaces.png"


[X_small, y_small] = read_images(image_path="training-images-small") 
[eigenvalues_small, eigenvectors_small, mean_small] = pca (as_row_matrix(X_small), y_small)

steps =[i for i in range (eigenvectors_small.shape[1])]
E = []
for i in range (len(steps)):
    numEvs = steps[i]
    P = project(eigenvectors_small[: ,0: numEvs ], X_small[0].reshape (1 , -1) , mean_small)
    R = reconstruct(eigenvectors_small[: ,0: numEvs ], P, mean_small)
    # reshape and append to plots
    R = R.reshape(X_small[0].shape )
    E.append(np.asarray(R))
# subplot ( title ="Reconstruction", images =E, rows =4, cols =4, 
#          sptitle ="Eigenvectors ", sptitles =steps , colormap =plt.cm.gray ) #filename ="python_pca_reconstruction.png"
 


projections = []
for xi in X:
    projections.append(project (eigenvectors, xi.reshape(1 , -1) , mean))

save_predictions(eigenvectors=eigenvectors, projections=projections,mean=mean,image_names=y, test_images_dir=TEST_MAGES_DIR)

# image = Image.open("test.jpg")
# image = image.convert ("L")
# if (DEFAULT_SIZE is not None ):
#     image = image.resize (DEFAULT_SIZE , Image.LANCZOS )
# test_image = np. asarray (image , dtype =np. uint8 )
# predicted = predict(eigenvectors, mean , projections, y, test_image)

# subplot ( title ="Prediction", images =[test_image, X[predicted]], rows =1, cols =2, 
#          sptitles = ["Unknown image", "Prediction :{0}".format(y[predicted])] , figsize = (5,5)) # , colormap =plt.cm.gray ,  filename ="prediction_test.png"
