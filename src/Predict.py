import sys
import os
new_dir = os.path.join(os.getcwd(), "src")
sys.path.append(new_dir)

import cv2 as cv
import numpy as np
import torch
from gensim import models
import xgboost as xgb
import XGBoost_utils
import joblib
from DL_models import CustomResNet

#Ad/Brand Gaze Prediction

#Now the model is only able to process magazine images or images with full-page counterpages
#Please indicate where is the ad by ad_location parameter: left <- ad_location=0, right <- ad_location=1; otherwise, set it as None
def Ad_Gaze_Prediction(input_ad_path, input_ctpg_path, ad_location,
                       text_detection_model_path, LDA_model_pth, training_ad_text_dictionary_path, training_lang_preposition_path,
                       training_language, ad_embeddings, ctpg_embeddings,
                       Ad_var=None, Ctpg_var=None,
                       flag_full_page_ad=False,
                       surface_sizes=None, Product_Group=None, Media_Category=None, TextBoxes=None, Obj_and_Topics=None,
                       filesize_ad=None, filesize_ctpg=None,
                       obj_detection_model_pth=None, num_topic=20, Gaze_Time_Type='Brand', Info_printing=True, Ad_Features_Only=False,
                       save_Var=False, Ad_Nr=None, Ctpg_Nr=None, task=None,
                       save_index=None, return_save_fts=False,
                       avgerage_out_index=None, average_out_data=None,
                       zeroing_out_index=None):
    
    Ad_ind = np.array([0,1,2,3,4,6,7,8,12,13,14,18,20,22]+list(range(24,31))+[38]+list(range(40,45))+list(range(50,53))+list(range(67,109))+[110])
    Ctpg_ind = np.array([5,9,10,11,15,16,17,19,21,23]+list(range(31,38))+[39]+list(range(45,50))+list(range(53,56))+list(range(56,65))
                  +[65,66]+[109])
    
    if Ad_var is not None and Ctpg_var is not None:
        gaze = 0
        if Gaze_Time_Type == 'ALL':
            gaze_brand = 0
            gaze_ad = 0
            gaze_bs = 0
        Vars_10_input = []
        num_samples = Ctpg_var[0].shape[0]
        for i in range(10):
            Var = np.zeros((num_samples,111))
            Var[:,Ad_ind] = Ad_var[i]
            Var[:,Ctpg_ind] = Ctpg_var[i]
            Vars_10_input.append(Var)
    else:
        Vars_10_input = None

        ##Image Loading
        if Info_printing: print('Loading Image ......')
        # flag_full_page_ad = False
        has_ctpg = True
        if type(input_ad_path) == str:
            ad_img = cv.imread(input_ad_path)
            ad_img = cv.cvtColor(ad_img, cv.COLOR_BGR2RGB)
            ad_img_dim1, ad_img_dim2 = ad_img.shape[:2]
            dim1_scale = int(np.ceil(ad_img_dim1/32))
            dim2_scale = int(np.ceil(ad_img_dim2/32))
            ad_img = cv.resize(ad_img, (32*dim2_scale,32*dim1_scale))
        else:
            ad_img = input_ad_path

        if input_ctpg_path is None:
            ctpg_img = None #Initialization
            has_ctpg = False
        else:
            if type(input_ctpg_path) == str:
                ctpg_img = cv.imread(input_ctpg_path)
                ctpg_img = cv.cvtColor(ctpg_img, cv.COLOR_BGR2RGB)
                ctpg_img_dim1, ctpg_img_dim2 = ctpg_img.shape[:2]
                dim1_scale = int(np.ceil(ctpg_img_dim1/32))
                dim2_scale = int(np.ceil(ctpg_img_dim2/32))
                ctpg_img = cv.resize(ctpg_img, (32*dim2_scale,32*dim1_scale))
            else:
                ctpg_img = input_ctpg_path

        if Info_printing: print()

        ##File Size
        if Info_printing: print('Calculating complexity (filsize) ......')
        if filesize_ad is None or filesize_ctpg is None:
            filesize_ad = XGBoost_utils.filesize_individual(input_ad_path)
            if has_ctpg:
                filesize_ctpg = XGBoost_utils.filesize_individual(input_ctpg_path)
            else:
                filesize_ctpg = 0
            if Info_printing: print()
        
        ##Salience
        if Info_printing: print('Processing Salience Information ......')
        #Salience Map
        S_map_ad = XGBoost_utils.Itti_Saliency(ad_img, scale_final=3)
        if has_ctpg:
            S_map_ctpg = XGBoost_utils.Itti_Saliency(ctpg_img, scale_final=3)

        #K-Mean
        threshold = 0.001
        enhance_rate = 1
        num_clusters = 3

        if flag_full_page_ad:
            width = S_map_ad.shape[1]

            left = S_map_ad[:, :width//2]
            vecs_left, km_left = XGBoost_utils.salience_matrix_conv(left,threshold,num_clusters,enhance_rate=enhance_rate)
            _,scores_left,widths_left,D_left = XGBoost_utils.img_clusters(num_clusters, left, km_left.labels_, km_left.cluster_centers_, vecs_left)

            right = S_map_ad[:, width//2:]
            vecs_right, km_right = XGBoost_utils.salience_matrix_conv(right,threshold,num_clusters,enhance_rate=enhance_rate)
            _,scores_right,widths_right,D_right = XGBoost_utils.img_clusters(num_clusters, right, km_right.labels_, km_right.cluster_centers_, vecs_right)

            ad_sal = np.array(scores_left) + np.array(scores_right)
            ad_width = np.array(widths_left) + np.array(widths_right); ad_width = np.log(ad_width+1)
            ad_sig_obj = D_left + D_right

            ctpg_sal = np.zeros_like(ad_sal)
            ctpg_width = np.zeros_like(ad_width)
            ctpg_sig_obj = 0

        else:
            vecs, km = XGBoost_utils.salience_matrix_conv(S_map_ad,threshold,num_clusters,enhance_rate=enhance_rate)
            _,scores,widths,D = XGBoost_utils.img_clusters(num_clusters, S_map_ad, km.labels_, km.cluster_centers_, vecs)
            ad_sal = np.array(scores)
            ad_width = np.log(np.array(widths)+1)
            ad_sig_obj = D

            if has_ctpg:
                vecs, km = XGBoost_utils.salience_matrix_conv(S_map_ctpg,threshold,num_clusters,enhance_rate=enhance_rate)
                _,scores,widths,D = XGBoost_utils.img_clusters(num_clusters, S_map_ctpg, km.labels_, km.cluster_centers_, vecs)
                ctpg_sal = np.array(scores)
                ctpg_width = np.log(np.array(widths)+1)
                ctpg_sig_obj = D
            else:
                ctpg_sal = np.zeros_like(ad_sal)
                ctpg_width = np.zeros_like(ad_width)
                ctpg_sig_obj = 0
        if Info_printing: print()

        ##Texture
        if Info_printing: print('Processing Textures and Symmetries ......')
        kp_stat_ad, num_kp_ad, vlad_enc_ad = XGBoost_utils.VLAD_Encoding_SIFT(ad_img)
        kp_stat_ctpg, num_kp_ctpg, vlad_enc_ctpg = XGBoost_utils.VLAD_Encoding_SIFT(ctpg_img)
        symmetry_ad = XGBoost_utils.symmetry_lines(ad_img)
        symmetry_ctpg = XGBoost_utils.symmetry_lines(ctpg_img)

        ##Number of Textboxes
        if Info_printing: print('Processing Textboxes ......')
        if TextBoxes is None:
            #Need multiples of 32 in both dimensions
            ad_num_textboxes = XGBoost_utils.text_detection_east(ad_img, text_detection_model_path)
            if has_ctpg:
                ctpg_num_textboxes = XGBoost_utils.text_detection_east(ctpg_img, text_detection_model_path)
            else:
                ctpg_num_textboxes = 0
        else:
            ad_num_textboxes, ctpg_num_textboxes = TextBoxes
        if Info_printing: print()

        ##Objects and Topic Difference
        if Info_printing: print('Processing Object and Topic Information ......')
        if Info_printing: print('Loading Object Detection Model')
        if Obj_and_Topics is None:
            if obj_detection_model_pth is None:
                model_obj = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True, verbose=False)
            else:
                # model_obj = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True, verbose=False)
                # model = torch.hub.load('ultralytics/yolov5', 'custom', path=obj_detection_model_pth, source='local')
                model_obj = torch.load(obj_detection_model_pth)
            model_lda = None
            dictionary = None
            dutch_preposition = None
            ad_num_objs, ctpg_num_objs, ad_topic_weights, topic_Diff = XGBoost_utils.object_and_topic_variables(ad_img, ctpg_img, has_ctpg, dictionary, 
                                                                                                dutch_preposition, training_language, model_obj, 
                                                                                                model_lda, num_topic)
        else:
            ad_num_objs, ctpg_num_objs, ad_topic_soft_weights, ctpg_topic_soft_weights = Obj_and_Topics
            indx = np.argmax(ad_topic_soft_weights)
            ad_topic_weights = np.zeros(num_topic)
            ad_topic_weights[indx] = 1
            topic_Diff = XGBoost_utils.KL_dist(ad_topic_soft_weights, ctpg_topic_soft_weights)
        
        if Info_printing: print()

        ##Left and Right Indicator
        if Info_printing: print('Getting Left/Right Indicator ......')
        if flag_full_page_ad:
            Left_right_indicator = [1,1]
        else:
            if has_ctpg:
                if ad_location == 0:
                    Left_right_indicator = [1,0]
                elif ad_location == 1:
                    Left_right_indicator = [0,1]
                else:
                    Left_right_indicator = [1,1]
            else:
                Left_right_indicator = [1,0]
        if Info_printing: print()

        ##Product Category
        if Info_printing: print('Getting Product Category Indicator ......')
        if Product_Group is None:
            group_ind = XGBoost_utils.product_category()
        else:
            group_ind = Product_Group
        if Info_printing: print()

        ##Surface Sizes
        if Info_printing: print('Getting Surface Sizes ......')
        if surface_sizes is None:
            ad_img = cv.cvtColor(ad_img, cv.COLOR_RGB2BGR)
            
            print('Please select the bounding box for your ad (from top left to bottom right)')
            A = XGBoost_utils.Region_Selection(ad_img)
            print()

            print('Please select the bounding box for brands (from top left to bottom right)')
            B = XGBoost_utils.Region_Selection(ad_img)
            print()

            print('Please select the bounding box for texts (from top left to bottom right)')
            T = XGBoost_utils.Region_Selection(ad_img)
            surface_sizes = [B/A*100,(1-B/A-T/A)*100,T/A*100,np.log(sum(Left_right_indicator)*5)]
        # else:
        #     surface_sizes[-1] = np.log(surface_sizes[-1]+1e-3)

        ##Typicality Measure
        # if Info_printing: print('Calculating Typicality Measure ......')
        # if Info_printing: print()

    ##Get All things together
    if Info_printing: print('Predicting ......')
    gaze = 0
    if Gaze_Time_Type == 'ALL':
        gaze_brand = 0
        gaze_ad = 0
        gaze_bs = 0
    
    Vars_10 = []
    Ad_Features = []
    if save_index is not None:
        saved_Features = []
    for i in range(10):
        if Vars_10_input is None:
            #Var construction
            pca_topic_transform = joblib.load('src/Topic_Embedding_PCAs/pca_model_'+str(i)+'.pkl')
            ad_topics_curr = pca_topic_transform.transform(ad_embeddings)[:,:4][0]
            ctpg_topics_curr = pca_topic_transform.transform(ctpg_embeddings)[:,:4][0]
            ad_topic_weights = ad_topics_curr
            topic_Diff = np.linalg.norm(ad_embeddings-ctpg_embeddings)
            X = surface_sizes+[filesize_ad,filesize_ctpg]+list(ad_sal)+list(ctpg_sal)+list(ad_width)+list(ctpg_width)+[ad_sig_obj,ctpg_sig_obj]+[ad_num_textboxes,ctpg_num_textboxes,ad_num_objs,ctpg_num_objs]
            X = np.array(X).reshape(1,len(X))
            X = np.concatenate((X,kp_stat_ad,kp_stat_ctpg,num_kp_ad,num_kp_ctpg,vlad_enc_ad,vlad_enc_ctpg,symmetry_ad,symmetry_ctpg),axis=1)
            #+list(group_ind)+list(ad_topic_weights)
            X_for_typ = list(X[0,[0,1,2,3,4,6,7,8,12,13,14,18,20,22,38]+list(range(40,45))+list(range(24,31))+list(range(50,53))])+list(group_ind)+list(ad_topic_weights)
            X_for_typ = np.array(X_for_typ).reshape(1,len(X_for_typ))
            Ad_Features.append(X_for_typ)
            
            if Gaze_Time_Type == 'Brand':
                med = torch.load('src/Brand_Gaze_Model/typicality_train_medoid')
            elif Gaze_Time_Type == 'Ad':
                med = torch.load('src/Ad_Gaze_Model/typicality_train_medoid')
            elif Gaze_Time_Type == 'BS':
                med = torch.load('src/Brand_Share_Model/typicality_train_medoid')
            elif Gaze_Time_Type == 'ALL':
                med = torch.load('src/Brand_Gaze_Model/typicality_train_medoid')
            
            typ = XGBoost_utils.typ_cat(med, X_for_typ, group_ind, np.abs)
            if Media_Category is None:
                Media_Category = np.zeros((1,9))
            Var = np.concatenate([X,Media_Category,np.array(Left_right_indicator).reshape(1,2),ad_topic_weights.reshape(1,4),group_ind.reshape(1,38),np.array([topic_Diff.item(),typ.item()]).reshape(1,2)],axis=1)
            
            if avgerage_out_index is not None:
                Var[0, avgerage_out_index] = average_out_data
            
            if zeroing_out_index is not None:
                Var[0, zeroing_out_index] = 0

            Vars_10.append(Var)
            if save_index is not None:
                saved_Features.append(Var[saved_Features])
        else:
            Var = Vars_10_input[i]

        if Ad_Features_Only:
            continue
    

        xgb_model = xgb.XGBRegressor()
        if Gaze_Time_Type == 'Brand':
            xgb_model.load_model('src/Brand_Gaze_Model/10_models/Model_'+str(i+1)+'.json')
        elif Gaze_Time_Type == 'Ad':
            xgb_model.load_model('src/Ad_Gaze_Model/10_models/Model_'+str(i+1)+'.json')
        elif Gaze_Time_Type == 'BS':
            xgb_model.load_model('src/Brand_Share_Model/10_models/Model_'+str(i+1)+'.json')
        elif Gaze_Time_Type == 'ALL':
            xgb_model.load_model('src/Brand_Gaze_Model/10_models/Model_'+str(i+1)+'.json')
            gaze_brand += xgb_model.predict(Var)
            xgb_model.load_model('src/Ad_Gaze_Model/10_models/Model_'+str(i+1)+'.json')
            gaze_ad += xgb_model.predict(Var)
            xgb_model.load_model('src/Brand_Share_Model/10_models/Model_'+str(i+1)+'.json')
            gaze_bs += xgb_model.predict(Var)
            
        gaze += xgb_model.predict(Var)
    if Ad_Features_Only:
        return Ad_Features
    if return_save_fts:
        return saved_Features
    gaze = gaze/10
    if Gaze_Time_Type == 'ALL':
        gaze_brand = gaze_brand/10
        gaze_ad = gaze_ad/10
        gaze_bs = gaze_bs/10
        
        if len(gaze_brand) == 1:
            return (np.exp(gaze_ad)-1).item(), (np.exp(gaze_brand)-1).item(), gaze_bs.item(), Vars_10
        else:
            return (np.exp(gaze_ad)-1), (np.exp(gaze_brand)-1), gaze_bs, Vars_10
    else:
        if Info_printing: print('The predicted '+Gaze_Time_Type+' gaze time is: ', (np.exp(gaze)-1).item() if Gaze_Time_Type != 'BS' else gaze.item())
        if len(gaze) == 1:
            return (np.exp(gaze)-1).item() if Gaze_Time_Type != 'BS' else gaze.item(), Vars_10
        else:
            return (np.exp(gaze)-1) if Gaze_Time_Type != 'BS' else gaze, Vars_10


def CNN_Prediction(adv_imgs, ctpg_imgs, ad_locations, Gaze_Type='AG'): #Gaze_Type='AG' or 'BG'
    gaze = 0
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    if Gaze_Type == 'AG':
        a_temp = 0.2590; b_temp = 1.1781 #AG
    elif Gaze_Type == 'BG':
        a_temp = 0.2100; b_temp = 0.3541 #BG
    elif Gaze_Type == 'BS':
        a_temp = 1; b_temp = 0 #BS

    for i in range(1):
        net = CustomResNet()
        net.load_state_dict(torch.load('src/CNN_Gaze_Model/Fine-tune_'+Gaze_Type+'/Model_'+str(i)+'.pth',map_location=torch.device('cpu')))
        net = net.to(device)
        if Gaze_Type != 'BS':
            with torch.no_grad():
                pred = net.forward(adv_imgs, ctpg_imgs, ad_locations)
                pred = torch.exp(pred*a_temp+b_temp) - 1
                gaze += pred/10
        else:
            with torch.no_grad():
                pred = net.forward(adv_imgs, ctpg_imgs, ad_locations)
                gaze += pred/10

    return gaze

def HeatMap_CNN(adv_imgs, ctpg_imgs, ad_locations, Gaze_Type='AG'):
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    net = CustomResNet()
    net.load_state_dict(torch.load('src/CNN_Gaze_Model/Fine-tune_'+Gaze_Type+'/Model_'+str(0)+'.pth',map_location=torch.device('cpu')))
    net = net.to(device)
    pred = net(adv_imgs/255.0,ctpg_imgs/255.0,ad_locations)
    print('heatmap pred: ', pred)

    pred.backward()

    # pull the gradients out of the model
    gradients = net.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = net.get_activations(adv_imgs).detach()

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze().to('cpu')

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    img = torch.permute(adv_imgs[0],(1,2,0)).to(torch.uint8).numpy()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    heatmap = cv.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_TURBO)
    superimposed_img = heatmap * 0.8 + img * 0.5
    superimposed_img /= np.max(superimposed_img)
    superimposed_img = np.uint8(255 * superimposed_img)

    return superimposed_img