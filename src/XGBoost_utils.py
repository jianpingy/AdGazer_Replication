import re
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import sys
import math
import time
from imutils.object_detection import non_max_suppression
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from PIL import Image
from io import BytesIO
from pytesseract import pytesseract
from gensim import corpora, models, similarities
import nltk
from nltk.corpus import stopwords
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

General_Category = {
    'Potatoes / Vegetables / Fruit': ['Potatoes / Vegetables / Fruit'],
    'Chemical products': ['Chemical products'],
    'Photo / Film / Optical items': ['Photo / Film / Optical items'],
    'Catering industry': ['Catering industry'],
    'Industrial products other': ['Industrial products other'],
    'Media': ['Media'],
    'Real estate': ['Real estate'],
    'Government': ['Government'],
    'Personnel advertisements': ['Personnel advertisements'],
    'Cars / Commercial vehicles': ['Cars / Commercial vehicles'],
    'Cleaning products': ['Cleaning products'],
    'Retail': ['Retail'],
    'Fragrances': ['Fragrances'],
    'Footwear / Leather goods': ['Footwear / Leather goods'],
    'Software / Automation': ['Software / Automation'],
    'Telecommunication equipment': ['Telecommunication equipment'],
    'Tourism': ['Tourism'],
    'Transport/Communication companies': ['Transport/Communication companies'],
    'Transport services': ['Transport services'],
    'Insurances': ['Insurances'],
    'Meat / Fish / Poultry': ['Meat / Fish / Poultry'],
    'Detergents': ['Detergents'],
    'Foods General': ['Foods general', 'Bread / Banquet', 'Chocolate / Confectionery', 'Soup / Soup products', 'Edible fats', 'Sugar / Herbs / Spices', 'Dairy'],
    'Other services': ['Education', 'Other services'], 
    'Banks and Financial Services': ['Banks / Financing', 'Financial services other'],
    'Office Products': ['Office equipment', 'Office automation hardware', 'Office products'],
    'Household Items': ['Household items', 'Small household equipment'],
    'Non-alcoholic beverages': ['Non-alcoholic beverages', 'Coffee/Tea'],
    'Hair, Oral and Personal Care': ['Skin care', 'Hair care', 'Oral care', 'Personal care electric'],
    'Fashion and Clothing': ['Outerwear', 'Underwear / Sleepwear'],
    'Other products and Services': ['Pet foods', 'Other products and services', 'Other advertisements'],
    'Paper products': ['Paper products', 'Paper products body care'],
    'Alcohol and Other Stimulants': ['Weak alcoholic drinks', 'Strong alcoholic drinks', 'Tobacco'],
    'Medicines': ['Medicines', 'Bandages'],
    'Recreation and Leisure': ['Recreation', 'Leisure items / Hobby items'],
    'Electronics': ['Kitchen appliances', 'Brown goods (Sound and video Electronics)'],
    'Home Furnishings': ['Home furnishings', 'Home upholstery', 'Home textiles'],
    'Products for Business Use': ['Products for business use', 'Other business services']}

#Saliency Map: Itti-Koch
def Itti_Saliency(img, scale_final=4):

    r = copy.copy(img[:,:,0].astype('float64'))
    g = copy.copy(img[:,:,1].astype('float64'))
    b = copy.copy(img[:,:,2].astype('float64'))

    #Intensity
    I = (r+g+b)/3
    dim1_img, dim2_img = np.shape(I)

    #Normalization of r,g,b
    mask1 = I >= 0.1*np.max(I)
    mask2 = I < 0.1*np.max(I)
    r[mask1] = r[mask1]/I[mask1]
    r[mask2] = 0
    g[mask1] = g[mask1]/I[mask1]
    g[mask2] = 0
    b[mask1] = b[mask1]/I[mask1]
    b[mask2] = 0

    #Fine-tuned Color Channels
    R = r-(g+b)/2
    G = g-(r+b)/2
    B = b-(r+g)/2
    Y = (r+g)/2-np.abs(r-g)/2-b

    #Intensity Feature Maps
    I_pyr = [I]
    R_pyr = [R]
    G_pyr = [G]
    B_pyr = [B]
    Y_pyr = [Y]

    I_maps = []
    RG_maps = []
    BY_maps = []

    for i in range(8):
        I_pyr.append(cv.pyrDown(I_pyr[i]))
        R_pyr.append(cv.pyrDown(R_pyr[i]))
        G_pyr.append(cv.pyrDown(G_pyr[i]))
        B_pyr.append(cv.pyrDown(B_pyr[i]))
        Y_pyr.append(cv.pyrDown(Y_pyr[i]))

    for c in (2,3,4):
        for d in (3,4):
            shape = (np.shape(I_pyr[c])[1],np.shape(I_pyr[c])[0])
            temp = cv.resize(I_pyr[c+d], shape, interpolation=cv.INTER_LINEAR)
            temp_G = cv.resize(G_pyr[c+d], shape, interpolation=cv.INTER_LINEAR)
            temp_R = cv.resize(R_pyr[c+d], shape, interpolation=cv.INTER_LINEAR)
            temp_B = cv.resize(B_pyr[c+d], shape, interpolation=cv.INTER_LINEAR)
            temp_Y = cv.resize(Y_pyr[c+d], shape, interpolation=cv.INTER_LINEAR)

            I_maps.append(np.abs(I_pyr[c]-temp))
            RG_maps.append(np.abs((R_pyr[c]-G_pyr[c])-(temp_G-temp_R)))
            BY_maps.append(np.abs((B_pyr[c]-Y_pyr[c])-(temp_Y-temp_B)))

    g_kernel = cv.getGaborKernel((5, 5), 2.0, np.pi/4, 10.0, 0.5, 0)
    O_maps = []
    for theta in (0, np.pi/4, np.pi/2, 3*np.pi/4):
        O_pyr = [I]
        for i in range(8):
            filtered = cv.filter2D(I_pyr[i], ddepth=-1, kernel=g_kernel)
            dim1,dim2 = np.shape(filtered)
            O_pyr.append(cv.resize(filtered, (dim1//2,dim2//2), interpolation=cv.INTER_LINEAR))
        for c in (2,3,4):
            for d in (3,4):
                shape = (np.shape(O_pyr[c])[1],np.shape(O_pyr[c])[0])
                temp = cv.resize(O_pyr[c+d], shape, interpolation=cv.INTER_LINEAR)

                O_maps.append(np.abs(O_pyr[c]-temp))
                
    S = 0
    M = 10
    scaling = 2**scale_final

    for I_map in I_maps:
        temp = normalization(I_map,M)
        temp = cv.resize(temp, (dim1_img//scaling, dim2_img//scaling), interpolation=cv.INTER_LINEAR)
        S += temp

    for i in range(len(RG_maps)):
        temp = normalization(RG_maps[i],M)+normalization(BY_maps[i],M)
        temp = cv.resize(temp, (dim1_img//scaling, dim2_img//scaling), interpolation=cv.INTER_LINEAR)
        S += temp

    for O_map in O_maps:
        temp = normalization(O_map,M)
        temp = cv.resize(temp, (dim1_img//scaling, dim2_img//scaling), interpolation=cv.INTER_LINEAR)
        S += temp

    S = 1/3*S
    return S
            
#Saliency map helper
def normalization(X, M):
    max_val = np.max(X)
    
    #first step
    X = X*M/(max_val+1e-5)
    
    #second step
    mask = X < M
    m_bar = np.mean(X[mask])
    
    #third step
    return (M-m_bar)**2*X
  
#For K-Means and Saliency Features
def salience_matrix_conv(sal_mat,threshold,num_clusters,enhance_rate=2):
    norm_sal = sal_mat**enhance_rate/np.max(sal_mat**enhance_rate)
    mask = norm_sal < threshold
    norm_sal[mask] = 0
    [dim1,dim2] = np.shape(sal_mat)
    mask = norm_sal >= threshold
    vecs = []
    for i in range(dim1):
        for j in range(dim2):
            if norm_sal[i,j] == 0:
                continue
            else:
                vecs.append([i/dim1,j/dim2,norm_sal[i,j]])
    vecs = np.array(vecs)
    km = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(vecs)
    return (vecs, km)

def Center(X):
    ws = X[:,2].reshape(len(X[:,2]),1)
    ws = ws/np.sum(ws)
    loc = X[:,0:2]
    return np.sum(loc*ws,axis=0)

def Cov_est(X, center):
    n = X.shape[0]
    loc = X[:,0:2]
    return np.matmul((loc-center).T,(loc-center))/n

def Renyi_Entropy(P):
    n = len(P)
    Q = np.ones_like(P)/n
    return np.sum(P*np.log(P/Q))#-2*np.log(np.sum(np.sqrt(P*Q)))

def img_clusters(num_clusters, img, kmeans_labels, k_means_centers, vecs, show_cluster=False):
    clusters = []
    labels = []
    scores = []
    widths = []
    pred = kmeans_labels

    for i in range(num_clusters):
        clusters.append(np.zeros_like(img))
        labels.append(pred == i)

    for k in range(num_clusters):
        for item in vecs[labels[k]]:
            i,j,val = item
            i = int(i); j = int(j)
            clusters[k][i,j] = val
        scores.append(np.sum(clusters[k]))
        widths.append(np.linalg.det(Cov_est(vecs[labels[k]],Center(vecs[labels[k]]))))
    ind = np.argsort(-1*np.array(scores))
    scores = np.array(scores)[ind]
    widths = np.array(widths)[ind]
    perc_S = np.array(scores)/sum(scores)
    D = 1/(Renyi_Entropy(perc_S)+0.001)

    if show_cluster:
        fig,ax = plt.subplots(1,num_clusters)
        for i in range(num_clusters):
            ax[i].imshow(clusters[i])
            ax[i].axis('off')
        plt.savefig("clusters.png", bbox_inches='tight')
        plt.show()

    return (clusters,perc_S,widths,D)

def weights_pages(ad_size, num_clusters):
    if ad_size == '1g':
        return np.concatenate((np.ones(num_clusters),0.2*np.ones(num_clusters)))
    elif ad_size == '1w':
        return np.ones(2*num_clusters)
    elif ad_size == 'hw':
        return np.concatenate((0.2*np.ones(num_clusters),np.ones(num_clusters)))
    else:
        return 0.5*np.ones(2*num_clusters)

def ad_loc_indicator(ad_size):
    if ad_size == '1g':
        return 0
    elif ad_size == '1w':
        return 2
    elif ad_size == 'hw':
        return 1
    else:
        return 3

def full_weights(ad_sizes, num_clusters):
    out = []
    for ad_size in ad_sizes:
        out.append(weights_pages(ad_size,num_clusters))
    return np.array(out)

def ad_loc_indicator_full(ad_sizes):
    out = []
    for ad_size in ad_sizes:
        out.append(ad_loc_indicator(ad_size))
    return np.array(out)

def filesize_individual(img_path):
    if type(img_path) == str:
        out = os.path.getsize(img_path)/1000
    else:
        img = Image.fromarray(img_path)
        img_file = BytesIO()
        img.save(img_file, 'jpeg')
        out = img_file.tell()

    return out/1e6

def KL_dist(P,Q):
    return np.sum(P*(np.log(P)-np.log(Q)),axis=1)

def KL_score(y_pred,y_test):
    kde_pred = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(y_pred.reshape(-1, 1))
    log_dens_pred = kde_pred.score_samples(y_pred.reshape(-1, 1))
    x = np.linspace(-10,10,num=100)
    kde_true = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(y_test.reshape(-1, 1))
    log_dens_true = kde_true.score_samples(y_test.reshape(-1, 1))
    plt.plot(x,np.exp(kde_pred.score_samples(x.reshape(-1, 1))),label='pred')
    plt.plot(x,np.exp(kde_true.score_samples(x.reshape(-1, 1))),label='true')
    plt.legend()
    plt.show()
    return np.sum(np.exp(log_dens_pred)*(log_dens_pred-log_dens_true))

def medoid(X_in, d):
    temp = []
    temp_d = []
    for item in X_in:
        temp.append(np.sum(d(X_in-item)))
        temp_d.append(np.sum(d(X_in-item), axis=1))
    return (np.argmin(temp), temp_d[np.argmin(temp)])

def sqr_dist(x):
    return np.multiply(x,x)

def data_normalize(X_train,X_test):
    num_train = X_train.shape[0]
    m_train = np.sum(X_train,axis=0)/num_train
    s_train = np.sqrt(np.sum((X_train-m_train)**2,axis=0))
    X_train_transf = (X_train-m_train)/s_train
    X_test_transf = (X_test-m_train)/s_train
    return X_train_transf, X_test_transf

def typ_cat(medoids, X_test, category, d):
    ind_temp = np.arange(len(category))
    ind_interest = ind_temp[np.array(category)==1][0]
    typ = np.sum(d(X_test-medoids[ind_interest]), axis=1)
    return typ
  
  
#EAST Text Detection
def text_detection_east(image, text_detection_model_path):
    orig = image.copy()
    (H, W) = image.shape[:2]
    (newW, newH) = (W, H)
    rW = W / float(newW)
    rH = H / float(newH)
    # resize the image and grab the new image dimensions
    image = cv.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    # print("[INFO] loading EAST text detector...")
    net = cv.dnn.readNet(text_detection_model_path)
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    # show timing information on text prediction
    # print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    count = 0

    for (x1, y1, x2, y2) in boxes:
        count += 1

    return count
  
  #Texts and Objects
def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def ad_object_detection(model, image, crop_dim=600):
    final_output = []
    results = model(np.moveaxis(center_crop(image,(crop_dim,crop_dim)),-1,0))
    results_all = model(np.moveaxis(image,-1,0))
    table = results.pandas().xyxy[0]
    table_all = results_all.pandas().xyxy[0]
    ad_objs = {}
    #      xmin    ymin    xmax   ymax  confidence  class    name
    for i, obj in enumerate(list(table['name'])):
        coords = [list(table['xmax'])[i], list(table['xmin'])[i], list(table['ymax'])[i], list(table['ymin'])[i]]
        area = (list(table['xmax'])[i] - list(table['xmin'])[i])*(list(table['ymax'])[i] - list(table['ymin'])[i])
        if obj in ad_objs:
            ad_objs[obj][0] += list(table['confidence'])[i]
            ad_objs[obj][1] += area
            ad_objs[obj][2].append(coords)
        else:
            ad_objs[obj] = [list(table['confidence'])[i], area, [coords]]
    for i, obj in enumerate(list(table_all['name'])):
        coords_all = [list(table_all['xmax'])[i], list(table_all['xmin'])[i], list(table_all['ymax'])[i], list(table_all['ymin'])[i]]
        area_all = (list(table_all['xmax'])[i] - list(table_all['xmin'])[i])*(list(table_all['ymax'])[i] - list(table_all['ymin'])[i])
        if obj in ad_objs:
            ad_objs[obj][0] += list(table_all['confidence'])[i]
            ad_objs[obj][1] += area_all
            ad_objs[obj][2].append(coords_all)
        else:
            ad_objs[obj] = [list(table_all['confidence'])[i], area_all, [coords_all]]

    count = 0
    for obj in list(ad_objs.keys()):
        count += len(ad_objs[obj][2])

    return ad_objs, count

def ad_word_classes(image, dutch_preposition, stop_words):
    ad_text_dic = {}
    text = pytesseract.image_to_string(image)
    result = re.sub(r'\W+','*',re.sub(r'\d+', '*', text.lower())).split('*')[:-1]
    for item in result:
        if item != '':
            if len(item) > 3:
                if item not in dutch_preposition and item not in stop_words:
                    if item in ad_text_dic:
                        ad_text_dic[item] += 1
                    else:
                        ad_text_dic[item] = 1
    
    return ad_text_dic

def topic_features(ad_objs, ad_text_dic, dictionary, model, num_topics=20):
    ad_topic_weights = np.zeros(num_topics)
    ad_topic_weights[0] = 1

    return ad_topic_weights

def object_and_topic_variables(ad_image, ctpg_image, has_ctpg, dictionary, dutch_preposition, language, model_obj, model_lda, num_topic=20):
    # nltk.download('stopwords')
    # stop_words = stopwords.words(language)

    #Ad
    ad_objs, ad_num_objs = ad_object_detection(model_obj, ad_image, crop_dim=600)
    ad_text_dic = None #ad_word_classes(ad_image, dutch_preposition, stop_words)
    ad_topic_weights = topic_features(ad_objs, ad_text_dic, dictionary, model_lda, num_topic)

    #Counterpage
    if has_ctpg:
        ctpg_objs, ctpg_num_objs = ad_object_detection(model_obj, ctpg_image, crop_dim=600)
        ctpg_text_dic = None #ad_word_classes(ctpg_image, dutch_preposition, stop_words)
        ctpg_topic_weights = topic_features(ctpg_objs, ctpg_text_dic, dictionary, model_lda, num_topic)
    else:
        ctpg_num_objs = 0
        ctpg_topic_weights = np.ones(num_topic)/num_topic

    #Topic Difference
    Diff = KL_dist(ad_topic_weights.reshape(1,num_topic), ctpg_topic_weights.reshape(1,num_topic))

    return ad_num_objs, ctpg_num_objs, ad_topic_weights, Diff
  
def product_category():
    global General_Category
    categories = np.array(list(General_Category.keys()))
    five_categories = []
    for i in range(len(categories)//5):
        five_categories.append(list(categories[(5*i):(5*i+5)]))
    if len(categories)%5 > 0:
        i = i+1
        five_categories.append(list(categories[(5*i):(5*i+5)]))
    
    #Create dictionary
    Name_to_Index_dict = {}
    for i in range(len(categories)):
        Name_to_Index_dict[categories[i]] = i

    #User Questions
    flag = True
    while flag:
        for i, item in enumerate(five_categories):
            print("list "+str(i+1)+" out of "+str(len(five_categories)))
            for j, cat in enumerate(item):
                print(str(j)+": "+cat)
            choice = input("Please choose the general category. If no good fit, please type N; otherwise, type the numbering: ")
            if choice == "N":
                print()
                continue
            else:
                choice = item[int(choice)]
                break
        confirm = input("If you have chosen successfully, please type Y or y; otherwise, please type any other key: ")
        if confirm == 'Y' or confirm == 'y':
            flag = False
        else:
            print('Please choose again.')

    #Output
    out = np.zeros(38)
    out[Name_to_Index_dict[choice]] = 1

    return out


def Region_Selection(img):
    As = []
    while True:
        _, _, w, h = cv.selectROI("Select ROI", img, fromCenter=False, showCrosshair=False)
        As.append(w*h)
        ans = input("Continue? (y/n) ")
        if ans == 'n':
            break
    A = sum(As)
    return A

def RMSRPD(y_pred, y_true):
    diff = y_pred - y_true
    den = 0.5*(np.abs(y_pred) + np.abs(y_true))
    return np.sqrt(np.mean((diff/den)**2))

def Caption_Generation(image): #image is a PIL Image object
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    with torch.no_grad():
        model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True, torch_dtype=torch.float16)
        model = model.to(device=device, dtype=torch.float16).eval()
        
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True) #'openbmb/MiniCPM-Llama3-V-2_5'


        question = 'Describe the image in a paragraph. Please include details.'
        msgs = [{'role': 'user', 'content': question}]

        res = model.chat(
            image=image,
            context=None,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True, # if sampling=False, beam_search will be used by default
            temperature=0.7,
            # system_prompt='' # pass system_prompt if needed
        )

    return res[0]

def Topic_emb(caption):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    with torch.no_grad():
        model = SentenceTransformer("src/Magazine_Topic_Embedding_sample_size15").to(device).eval()
        embeddings = model.encode(caption).reshape(1,768)
    return embeddings

def VLAD_Encoding_SIFT(image):
    if image is None:
        return np.zeros((1,7)), np.zeros((1,1)), np.zeros((1,5))
    h,w,_ = image.shape
    sift = cv.SIFT_create()
    kmeans = torch.load('src/SIFT/kmeans.pt', weights_only=False)
    pca_vlad = torch.load('src/SIFT/pca.pt', weights_only=False)
    num_clusters = 64

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    labels = kmeans.predict(descriptors)
    centroids = kmeans.cluster_centers_
    vlad_vector = np.zeros((num_clusters, descriptors.shape[1]))
    for j in range(descriptors.shape[0]):
        cluster_id = labels[j]
        vlad_vector[cluster_id] += descriptors[j] - centroids[cluster_id]
    kps_np = np.array([kp.pt + (kp.size, kp.angle) for kp in keypoints])
    vlad_vector = vlad_vector.flatten()
    vlad_vector = np.sign(vlad_vector) * np.sqrt(np.abs(vlad_vector))  # Power normalization
    vlad_vector /= np.linalg.norm(vlad_vector)
    encoding = pca_vlad.transform(vlad_vector.reshape(1,-1))[:,:5]
    kps_np[:,0] = kps_np[:,0] / w
    kps_np[:,1] = kps_np[:,1] / h
    kps_np[:,2] = kps_np[:,2] / w
    mean_pos = np.mean(kps_np[:, :2], axis=0)
    var_pos = np.var(kps_np[:, :2], axis=0)
    mean_scale = np.mean(kps_np[:, 2])
    var_scale = np.var(kps_np[:, 2])
    mean_ori = np.mean(kps_np[:, 3])
    num_kp = len(kps_np)
    # print('First output: ', mean_pos, var_pos, mean_scale, var_scale, mean_ori)

    return np.array([mean_pos[0],mean_pos[1],var_pos[0],var_pos[1],mean_scale,var_scale,mean_ori]).reshape(1,7), num_kp*np.ones((1,1))/1000, encoding

def very_close(a, b, tol=4.0):
    """Checks if the points a, b are within
    tol distance of each other."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) < tol


def S(si, sj, sigma=1):
    """Computes the 'S' function mentioned in
    the research paper."""
    q = (-abs(si - sj)) / (sigma * (si + sj))
    return np.exp(q ** 2)


def reisfeld(phi, phj, theta):
    return 1 - np.cos(phi + phj - 2 * theta)


def midpoint(i, j):
    return (i[0] + j[0]) / 2, (i[1] + j[1]) / 2


def angle_with_x_axis(i, j):
    x, y = i[0] - j[0], i[1] - j[1]
    if x == 0:
        return np.pi / 2
    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle


def superm2(image):
    """Performs the symmetry detection on image.
    Somewhat clunky at the moment -- first you
    must comment out the last two lines: the
    call to `draw` and `cv2.imshow` and uncomment
    `hex` call. This will show a 3d histogram, where
    bright orange/red is the maximum (most voted for
    line of symmetry). Manually get the coordinates,
    and re-run but this time uncomment draw/imshow."""
    sift = cv.SIFT_create()
    mimage = np.fliplr(image)
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(mimage, None)
    for p, mp in zip(kp1, kp2):
        p.angle = np.deg2rad(p.angle)
        mp.angle = np.deg2rad(mp.angle)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    houghr = np.zeros(len(matches))
    houghth = np.zeros(len(matches))
    weights = np.zeros(len(matches))
    i = 0
    good = []
    for match, match2 in matches:
        point = kp1[match.queryIdx]
        mirpoint = kp2[match.trainIdx]
        mirpoint2 = kp2[match2.trainIdx]
        mirpoint2.angle = np.pi - mirpoint2.angle
        mirpoint.angle = np.pi - mirpoint.angle
        if mirpoint.angle < 0.0:
            mirpoint.angle += 2 * np.pi
        if mirpoint2.angle < 0.0:
            mirpoint2.angle += 2 * np.pi
        mirpoint.pt = (mimage.shape[1] - mirpoint.pt[0], mirpoint.pt[1])
        if very_close(point.pt, mirpoint.pt):
            mirpoint = mirpoint2
            good.append(match2)
        else:
            good.append(match)
        theta = angle_with_x_axis(point.pt, mirpoint.pt)
        xc, yc = midpoint(point.pt, mirpoint.pt)
        r = xc * np.cos(theta) + yc * np.sin(theta)
        Mij = reisfeld(point.angle, mirpoint.angle, theta) * S(
            point.size, mirpoint.size
        )
        houghr[i] = r
        houghth[i] = theta
        weights[i] = Mij
        i += 1
    # matches = sorted(matches, key = lambda x:x.distance)
    good = sorted(good, key=lambda x: x.distance)

    return good, (houghr, houghth, weights)

def symmetry_lines(image):
    if image is None:
        return np.zeros((1,3))
    _, hough_data = superm2(image)
    bins_input = []
    N = 3
    for i in range(N+1):
        bins_input.append(5/N*i)
    out = plt.hist(hough_data[2], bins=bins_input)
    return out[0].reshape(1,-1)/10
