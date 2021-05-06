#!/usr/bin/env python
# coding: utf-8

# 
# # **NOTEBOOK WANTED**
# ---
# 
# 

# # Importations

# In[1]:


from platform import python_version
python_version()


# In[2]:


import numpy as np
#import sklearn as sk
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import time


# In[3]:


from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score


# In[4]:


import cv2
from PIL import Image
from matplotlib import pyplot as plt
from os import listdir
from os import mkdir
from os import chdir
from os import getcwd
from os import path as pa
import shutil
import scipy
import scipy.stats
from deepface import DeepFace


# In[5]:


import tensorflow as tf
print(tf.__version__)


# In[6]:


import mtcnn
# print version
print(mtcnn.__version__)
import matplotlib.patches as pat


# In[7]:


model_rf = tf.keras.models.load_model('facenet_keras2.h5')
# summarize input and output shape
print(model_rf.inputs)
print(model_rf.outputs)


# ## Seuils et listes

# In[8]:


l_seuil=np.array([10.95177734, 11.20710986, 11.19705008, 11.22642203, 11.25440334, 11.37028115,
 11.44615134, 11.45221935, 11.45384116, 11.47330318, 11.46870628])

l_sigma=np.array([1.5107747, 1.5097685, 1.4701916, 1.4331723, 1.4052494, 1.3568811, 1.3224955,
 1.3083414, 1.3023157, 1.2942157, 1.2904698])

l_reso=np.array([20,25,30,35,40,50,60,70,80,90,100])

l_seuil_bas=l_seuil-l_sigma

l_seuil_std = [0.98705189, 0.94402418, 0.82337106, 0.70955529, 0.73845168, 0.64627971,
 0.60074408, 0.57992375, 0.56640448, 0.56008308, 0.55475069]



# # Functions

# In[9]:


# extract a single face from a given photograph
def extract_face(filename, size_input,size_output):
    """ load image from file """
    image = Image.open(filename)
    # convert to RGB
    image = image.convert('RGB')
    # convert to array
    pixels = np.array(image)
    # detect faces in the image
    results = detector.detect_faces(pixels)
    if len(results)== 0:
        return 0
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    #resolution voulue pour le test
    image = image.resize(size_input)
    #resolution à l'entrée du réseau
    image = image.resize(size_output)
    face_array = np.array(image)
    return face_array


# ##### Other functions

# In[10]:



def draw_faces(filename, result_list):
    # load the image
    """ draw each face separately"""
    data = plt.imread(filename)
    # plot each face as a subplot
    for i in range(len(result_list)):
        # get coordinates
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        # define subplot
        plt.subplot(1, len(result_list), i+1)
        plt.axis('off')
        # plot face
        plt.imshow(data[y1:y2, x1:x2])
    # show the plot
    plt.show()
    
def draw_image_with_boxes(filename, result_list):
    # load the image
    data = plt.imread(filename)
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = pat.Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = pat.Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    # show the plot
    plt.show()

    
# same as extract_face but for several faces in an image
def extract(filename, size_input,size_output):
  # load the image
  # load image from file
    image = Image.open(filename)
  # convert to RGB
    image = image.convert('RGB')
  # convert to array
    pixels = np.array(image)
  # detect faces in the image
    result_list = detector.detect_faces(pixels)
  # extract the bounding box from the first face
    liste_faces=[]
    for i in range(len(result_list)):
    # get coordinates
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        try :
            face = pixels[y1:y2, x1:x2]
            x = Image.fromarray(face)

            #resolution voulue
            x = x.resize(size_input)
            #resolution à l'entrée du réseau
            x = x.resize(size_output)
            face_array = np.array(x)
            liste_faces.append(face_array)
        except : 
            print('impossible de charger image')
    liste_faces=np.asarray(liste_faces)
    return liste_faces



def super_extract(filename, portrait,size_input,size_output):
    """function that combines draw_boxes and extract"""
  # load the image
  # load image from file
    image = Image.open(filename)
  # convert to RGB
    image = image.convert('RGB')
    if portrait==True:
        image = image.rotate(-90, expand=True)
  # convert to array
    pixels = np.array(image)
  # detect faces in the image
    result_list = detector.detect_faces(pixels)
  # extract the bounding box from the first face
    liste_faces=[]
    liste_dim=[]
#partie graphique
    plt.imshow(image)
    ax = plt.gca()
    for i in range(len(result_list)):
    # get coordinates
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        rect = pat.Rectangle((x1, y1), width, height, fill=False, color='red')
        liste_dim.append([width, height])
        # draw the box
        ax.add_patch(rect)
        try :
            face = pixels[y1:y2, x1:x2]
            x = Image.fromarray(face)

            #resolution voulue
            x = x.resize(size_input)
            #resolution à l'entrée du réseau
            x = x.resize(size_output)
            face_array = np.array(x)
            liste_faces.append(face_array)
        except : 
            print('impossible de charger image')
    liste_faces=np.asarray(liste_faces)
    liste_dim=np.asarray(liste_dim)
    plt.show()
    return liste_faces,liste_dim


# ## Loading data

# In[11]:


# 
def load_faces(directory, size_input,size_output):
    """load images and extract faces for all images in a directory"""
    faces = []
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        if type(extract_face(path, size_input,size_output))== np.ndarray:
            face = extract_face(path, size_input,size_output)
        # store
            faces.append(face)
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory, size_input,size_output):
    X, y = [], []
    # enumerate folders, on per class
    for subdir in listdir(directory):
        print(subdir)
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not pa.isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path, size_input,size_output)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        #print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

# load a dataset that contains one subdir for each class that in turn contains images
def nb_classes(directory):
    k=0
    for subdir in listdir(directory):
        k+=1
    return k
def id_possibles(directory):
    l=[]
    for subdir in listdir(directory):
        l.append(subdir)
    return l

# load images and extract faces for all images in a directory
def load_faces_2(directory, size_input,size_output):
    faces = []
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        if type(extract(path, size_input,size_output))== np.ndarray:
            l_face = extract(path, size_input,size_output)
        # store
            for face in l_face:
                faces.append(face)
    return np.asarray(faces)


# ## Embedding 

# In[12]:


def get_embedding(model, face_pixels):
    """returns embedding from the pixels of a face"""
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


# ## Unknown Detection

# In[13]:


#
def distance_embd(embed,prediction,newTrainX, y_train):
    """function that gives the mean distance between an embedding and it's predicted class"""
    l_p=np.where(y_train == prediction)
    #embeding réel des photos de la personne prédite dans le set de données
    #real embeding of photos in the predicted class
    l_emb=newTrainX[l_p]
    if len(l_emb)>1:
        ctr_emb=np.zeros((len(l_emb[0])))
        for k in range (len(l_emb[0])):
            ctr_emb[k]=np.mean(l_emb[:,k])
        # calcul de la distance
        distance=np.linalg.norm(ctr_emb-embed)
    else:
        p=np.where(y_train == prediction)[0]
        ctr_emb=newTrainX[p]
    distance=np.linalg.norm(ctr_emb-embed)
    return distance


# In[14]:


#
def calcul_embedding_reso(path_image,portrait):
    """gives the embedding of an image and it's size"""
    result=super_extract(path_image,portrait,(160,160),(160,160))
    #images
    x_test_mult = result[0]
    #dimensions
    liste_dim=result[1]
    newTestX_mult = list()
    for face_pixels in x_test_mult:
        embedding = get_embedding(model_rf, face_pixels)
        newTestX_mult.append(embedding)
    newTestX_mult = np.asarray(newTestX_mult)
    print("Nombre de visages détectés : %f" % np.shape(newTestX_mult)[0])
    if len(newTestX_mult)==0:
        return("pas de visage sur cette image")
    return newTestX_mult,liste_dim



def detection_unknown(embed,resolution,newKnownX,l_reso=np.array([20,25,30,35,40,50,60,70,80,90,100]),l_seuil=l_seuil_bas):
    """returns True if the embed corresponds to an unknown face (is not close enough to another embedding) and False if it is considered as known"""
    l_dist=[]
    for emb in newKnownX:
        l_dist.append(np.linalg.norm(emb-embed))
    l_dist=np.asarray(l_dist)
    seuil=l_seuil[np.argmin(np.abs(l_reso-resolution))]
    distance = np.min(l_dist)
    print("distance l2 = %f" % distance)
    print("seuil = %f" %seuil)
    if distance > seuil:
        #print("individu inconnu")
        return True
    else:
        #print("individu connu")
        return False


# ## Detection of known people

# In[15]:


#
def detection_wanted(list_emb,list_name_wanted,path_image,portrait,l_reso,l_seuil):
    """detects if an image contains a known face by comparing to a list of embd of a known person"""
    # preprocess    
    newX=list_emb
    newY=list_name_wanted
    model.fit(newX,newY)
    
    # ANALYSIS of the image
    result=super_extract(path_image,portrait,(160,160),(160,160))
    #images
    x_test_mult = result[0]
    #dimensions
    liste_dim=result[1]
    
    # calcul des emb
    newTestX_mult = list()
    for face_pixels in x_test_mult:
        embedding = get_embedding(model_rf, face_pixels)
        newTestX_mult.append(embedding)
    newTestX_mult = np.asarray(newTestX_mult)
    if len(newTestX_mult) == 0:
        return("pas de visage sur cette image")
    y_pred = model.predict(newTestX_mult)
  
    #comparison with the former data : deciding if wanted people is already there
    
    for k in range (len(y_pred)):
        dim=liste_dim[k]
        print("dimension image = %f" % min(dim) + "px")
        if min(dim) <= 30:
            print("visage trop petit pour bonne fiabiité")
            return ("visage trop petit pour bonne fiabiité")
        
        print(y_pred[k])
        if detection_unknown(newTestX_mult[k],min(dim),newX,l_reso) == False:
            print("individu connu")
            if y_pred[k] in list_name_wanted:
                print(y_pred[k] + " identifié")
                face=x_test_mult[k]
                plt.axis('off')
                plt.imshow(face)
                plt.show()
        else:
            print("individu inconnu")

            
def detection_wanted_dataset(path_set,list_emb,list_name_wanted,portrait,l_reso,l_seuil):
    """Uses detection_xanted for a dataset"""
    for filename in listdir(path_set):
        path_image = directory + filename
        detection_wanted(list_emb,list_name_wanted,path_image,portrait,l_reso,l_seuil)

def convert_embd(path_wanted):
    list_emb=[]
    list_name_wanted=[]
    l=listdir(path_wanted)
    for name in l:
        path1=path_wanted + "/" + name
        for filename in listdir(path1):
            path2=path1+"/"+filename
            face=extract_face(path2, (160,160),(160,160))
            x_test_m=[]
            x_test_m.append(face)
            for face_pixels in x_test_m:
                embedding = get_embedding(model_rf, face_pixels)
                list_emb.append(embedding)
                list_name_wanted.append(name)
    list_emb=np.asarray(list_emb)
    list_name_wanted=np.asarray(list_name_wanted)
    return list_emb,list_name_wanted


#Même fonction qu'avant mais qui prend en entrée un dossier avec des
#sous dossier nommés contenant des images d'une personne wanted stockée 
#et plus des embd

#
def detection_wanted_img(path_wanted,path_image,portrait,l_reso,l_seuil):
    """same function as detection_wanted but with a fold with imges as input and no more just embd"""
    print("calcul des embeddings à rechercher")
    res=convert_embd(path_wanted)
    list_emb,list_name_wanted=res[0],res[1]
    print("calcul terminé")
    list_emb=np.asarray(list_emb)
    list_name_wanted=np.asarray(list_name_wanted)
    detection_wanted(list_emb,list_name_wanted,path_image,portrait,l_reso,l_seuil)




def detection_wanted_dataset_img(path_set,path_wanted,portrait,l_reso,l_seuil):
    print("calcul des embeddings à rechercher")
    res=convert_embd(path_wanted)
    list_emb,list_name_wanted=res[0],res[1]
    print("calcul terminé")
    print(list_name_wanted)
    for filename in listdir(path_set):
        path_image = path_set + filename
        print("analyse de l'image : " + filename)
        detection_wanted(path_wanted,path_image,portrait,l_reso,l_seuil)


# #### Maraudage

# In[16]:



def analyse_new_image(newKnownX,y_known,path_image,name_image,pathknown,portrait,l_seuil,l_reso):
    """Analyses images, founds faces and compares them to already loaded people"""
    #chargement des personnes connues
    liste_known=listdir(pathknown)
    t=time.clock()
    result=super_extract(path_image,portrait,(160,160),(160,160))
    #images
    x_test_mult = result[0]
    #dimensions
    liste_dim=result[1]
    newTestX_mult = list()
    for face_pixels in x_test_mult:
        embedding = get_embedding(model_rf, face_pixels)
        newTestX_mult.append(embedding)
    newTestX_mult = np.asarray(newTestX_mult)
    tt=time.clock()
    print("temps pour extraction : %f " % np.abs(tt-t))
    print("Nombre de visages détectés : %f" % np.shape(newTestX_mult)[0])
    if len(newTestX_mult)==0:
        return("pas de visage sur cette image")
    if len(newKnownX)>1:
    #fiting with known people and trained people
        model.fit(newKnownX,y_known)
    #identifying people on the new image
        y_pred = model.predict(newTestX_mult)
        print(y_pred)
        probability = model.predict_proba(newTestX_mult)
        print(np.shape(probability))
  #comparison with the former data : deciding if identified people were already there
        for k in range (len(y_pred)):
            dim=liste_dim[k]
            print("dimension image = %f" % min(dim) + "px")
            if min(dim)<=30:
                print("visage trop petit pour bonne fiabiité")
                return ("visage trop petit pour bonne fiabiité")
            list_dist=liste_distance(newTestX_mult[k],newKnownX)
            #affichage(list_dist)
            if detection_unknown(newTestX_mult[k],min(dim),newKnownX,l_reso) == False and y_pred[k] in liste_known:
        
                print('individu connu repéré')
                face=x_test_mult[k]
                l=k+1
                img=Image.fromarray(face)
                path2 = pathknown + y_pred[k] + "/" + name_image + "ind_%f" % l + ".jpg"
                img.save(path2)
                newKnownX.append(newTestX_mult[k])
                y_known.append(y_pred[k])
                print(y_pred[k])
            else:
                face=x_test_mult[k]
                img=Image.fromarray(face)
                l=k+1
                path_savek=pathknown + '/' + name_image + "ind_%f" % l
                try:
                    mkdir(path_savek)
                except:
                    print('dossier existe')
                img.save(path_savek + '/' + "ind_%f" % l + '.jpg')
                print("individu non encore renontré")
                newKnownX.append(newTestX_mult[k])
                y_known.append(name_image + "ind_%f" % l)
    else:
        for k in range (len(x_test_mult)):
            face=x_test_mult[k]
            img=Image.fromarray(face)
            l=k+1
            path_savek=pathknown + '/' + name_image + "ind_%f" % l
            try:
                mkdir(path_savek)
            except:
                print('dossier existe')
            img.save(path_savek + '/' + "ind_%f" % l + '.jpg')
            print("individu non encore renontré")
            newKnownX.append(newTestX_mult[k])
            y_known.append(name_image + "ind_%f" % l)
        
        


# In[17]:


def maraudage(path_images,pathknown,portrait,l_seuil,l_reso):
    """uses analyse_new_image for a complete dataset and shows potential suspects"""
    liste=listdir(path_images)
    nb_image=len(liste)
    liste_wanted=[]
    newKnownX = []
    y_known=[]
    for k in range (nb_image):
        t1 = time.clock()
        image_name= liste[k]
        image=Image.open(path_images+"/"+image_name)
        #image = image.rotate(90, expand=True)
        path_image = path_images+"/"+image_name
        print("analyse de l'image : "+ image_name)
        analyse_new_image(newKnownX,y_known,path_image,image_name,pathknown,portrait,l_seuil,l_reso)
        t2 = time.clock()
        dt=np.abs(t2-t1)
        print("temps pour analyse : %f " % dt)
        l=listdir(pathknown)
        #print(len(newKnownX))
        for subdir in l:
            if len(listdir(pathknown+subdir)) > 3:
                print("individu suspect repéré")
                liste_wanted.append(subdir)
    for subdir in liste_wanted:
        print("photos du suspect : ")
        pathw=pathknown+subdir+"/"
        lw=listdir(pathw)
        for nameimg in lw:
            img=Image.open(pathw+nameimg)
            img=np.asarray(img)
            plt.imshow(img)
        


# In[18]:


def affichage(prob):
    """displays a distribution and its gaussian modelisation"""
    plt.plot(prob)
    bins=50
    ### Histogramme des données
    y, x = np.histogram(prob2, bins=200, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    
    
    #estimation de la loi
    dist = getattr(scipy.stats, "norm")
    # Modéliser la loi
    param = dist.fit(prob)
    #print(param)
    

    count, bins2= np.histogram(prob, bins=bins, density=True)
    #plt.show()
    mu, sigma = param[0], param[1]
    plt.figure()
    plt.plot(bins2, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins2 - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r',label='distribution prédite')
    plt.hist(prob, bins=bins, density=True,label='histogramme')
    plt.show()


# ### Partie video

# In[19]:


from moviepy.editor import VideoFileClip

# prendre une capture à un instant donné
def capture_video(path_video,path_save,name_save,instant):
    """takes a picture from a video at a definite moment"""
    vidcap = cv2.VideoCapture(path_video)
    vidcap.set(cv2.CAP_PROP_POS_MSEC,1000*instant)      # just cue to 20 sec. position
    success,image = vidcap.read()
    if success:
        print("save")
        cv2.imwrite(path_save+"/"+name_save+".jpg", image)     # save frame as JPEG file
        #cv2.imshow("20sec",image)
        cv2.waitKey()
    else:
        print("echec")
#prendre une capture avec la webcam
def capture_video_webcam(path_save,name_save):
    """takes a picture with the webcam"""
    vidcap = cv2.VideoCapture(0)
    success,image = vidcap.read()
    if success:
        print("save")
        cv2.imwrite(path_save+"/"+name_save+".jpg", image)     # save frame as JPEG file
        #cv2.imshow("20sec",image)
        cv2.waitKey()
    else:
        print("echec")


# In[20]:


#detecter le maraudage sur une video

def analyse_video(path_video,name_video,path_save,pathknown,portrait,l_seuil,l_reso):
    """detects marauding on a video"""
    liste_wanted=[]
    newKnownX = []
    y_known=[]
    clip = VideoFileClip(path_video)
    T=clip.duration
    t1=time.clock()
    t2=0
    while t2<T :
        t2 = np.abs(time.clock()-t1)
        name_save=name_video+str(t2)
        capture_video(path_video,path_save,name_save,t2)
        path_image = path_save + "/" + name_save + ".jpg"
        print("analyse de l'image : "+ name_save)
        t3 = time.clock()
        analyse_new_image(newKnownX,y_known,path_image,name_save,pathknown,portrait,l_seuil,l_reso)
        t4 = time.clock()
        dt=np.abs(t4-t3)
        print("temps pour analyse : %f " % dt)
        l=listdir(pathknown)
        #print(len(newKnownX))
        for subdir in l:
            if len(listdir(pathknown+subdir)) > 3:
                print("individu suspect repéré")
                liste_wanted.append(subdir)
        t2 = np.abs(time.clock()-t1)
    for subdir in liste_wanted:
        print("photos du suspect : ")
        pathw=pathknown+subdir+"/"
        lw=listdir(pathw)
        for nameimg in lw:
            img=Image.open(pathw+nameimg)
            img=np.asarray(img)
            plt.imshow(img)

#detecter le maraudage sur la webcam
            
def analyse_video_webcam(duration,name_video,path_save,pathknown,portrait,l_seuil,l_reso):
    """detects marauding in real time with the webcam as cctv"""
    liste_wanted=[]
    newKnownX = []
    y_known=[]
    t1=time.clock()
    t2=0
    while t2 < duration :
        t2 = np.abs(time.clock()-t1)
        name_save=name_video+str(t2)
        capture_video_webcam(path_save,name_save)
        path_image = path_save + "/" + name_save + ".jpg"
        print("analyse de l'image : "+ name_save)
        t3 = time.clock()
        analyse_new_image(newKnownX,y_known,path_image,name_save,pathknown,portrait,l_seuil,l_reso)
        t4 = time.clock()
        dt=np.abs(t4-t3)
        print("temps pour analyse : %f " % dt)
        l=listdir(pathknown)
        #print(len(newKnownX))
        for subdir in l:
            if len(listdir(pathknown+subdir)) > 3:
                print("individu suspect repéré")
                liste_wanted.append(subdir)
        t2 = np.abs(time.clock()-t1)
    for subdir in liste_wanted:
        print("photos du suspect : ")
        pathw=pathknown+subdir+"/"
        lw=listdir(pathw)
        for nameimg in lw:
            img=Image.open(pathw+nameimg)
            img=np.asarray(img)
            plt.imshow(img)


# In[26]:


def detection_video(path_video,name_video,path_wanted,path_save,portrait,l_reso,l_seuil):
    """identify pepole in real time with the webcam"""
    liste_wanted=[]
    newKnownX = []
    clip = VideoFileClip(path_video)
    y_known=[]
    t1=time.clock()
    t2=0
    T=clip.duration
    res=convert_embd(path_wanted)
    list_emb=res[0]
    list_name_wanted=res[1]
    while t2 < T :
        t2 = np.abs(time.clock()-t1)
        name_save=name_video+str(t2)
        capture_video(path_video,path_save,name_save,t2)
        path_image = path_save + "/" + name_save + ".jpg"
        print("analyse de l'image : "+ name_save)
        t3 = time.clock()
        detection_wanted(list_emb,list_name_wanted,path_image,portrait,l_reso,l_seuil)
        t4 = time.clock()
        dt=np.abs(t4-t3)
        print("temps pour analyse : %f " % dt)


# In[21]:



def detection_video_webcam(duration,name_video,path_wanted,path_save,portrait,l_reso,l_seuil):
    """identify pepole in real time with the webcam"""
    liste_wanted=[]
    newKnownX = []
    y_known=[]
    t1=time.clock()
    t2=0
    res=convert_embd(path_wanted)
    list_emb=res[0]
    list_name_wanted=res[1]
    while t2 < duration :
        t2 = np.abs(time.clock()-t1)
        name_save=name_video+str(t2)
        capture_video_webcam(path_save,name_save)
        path_image = path_save + "/" + name_save + ".jpg"
        print("analyse de l'image : "+ name_save)
        t3 = time.clock()
        detection_wanted(list_emb,list_name_wanted,path_image,portrait,l_reso,l_seuil)
        t4 = time.clock()
        dt=np.abs(t4-t3)
        print("temps pour analyse : %f " % dt)


# # Face detector

# In[22]:


detector = mtcnn.MTCNN()


# In[23]:


backends = ['opencv', 'ssd', 'dlib']


# # Face recognition

# ## Inception pré-entrainé

# In[24]:


#define svc classifier model 
model = SVC(kernel='linear', probability=True)

