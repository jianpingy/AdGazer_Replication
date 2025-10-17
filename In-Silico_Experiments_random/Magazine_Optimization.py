import sys
sys.path.append('XGBoost_Prediction_Model/')

import warnings
warnings.filterwarnings("ignore")

import cv2 as cv
import numpy as np
from pulp import *
import Predict
import torch
from torchvision.io import read_image

#Global Paths for Models and Dictionaries
obj_detection_model_pth = None #'../XGBoost_Prediction_Model/yolov5s.pt'
text_detection_model_path = '../XGBoost_Prediction_Model/EAST-Text-Detection/frozen_east_text_detection.pb'
LDA_model_pth = '../XGBoost_Prediction_Model/LDA_Model_trained/lda_model_best_tot.model'
training_ad_text_dictionary_path = '../XGBoost_Prediction_Model/LDA_Model_trained/object_word_dictionary'
training_lang_preposition_path = '../XGBoost_Prediction_Model/LDA_Model_trained/dutch_preposition'

avg_indices = None #np.array([20,21,22,23,50,51,52,53,54,55,109]) #np.array(list(range(4,56))+list(range(67,71))+[109,110])
avg_high_level = None #torch.load('../XGBoost_Prediction_Model/average_high_level.pt')
zero_indices = None #np.array(list(range(56,65))+list(range(67,109))+[110])

def Preference_Matrix(Magazine_Pages, Magazine_Slots, Ad_Groups, Ad_Element_Sizes, 
                      Ad_embeddings, Ctpg_embeddings,
                      Textboxes=None, Obj_and_Topics=None, Costs=None, Magazine_Type=None, 
                      Filesizes_ad=None, Filesizes_ctpg=None,
                      Method='XGBoost', length_only=False,
                      Magazine_Nr=-1, task=''):
#Magazine_Pages: A list containing all paths to Magazine Ads and Editorials
#Magazine_Slots: 0 (right), 1 (left), 2 (full-page)

    Ad_ind = np.array([0,1,2,3,4,6,7,8,12,13,14,18,20,22]+list(range(24,31))+[38]+list(range(40,45))+list(range(50,53))+list(range(67,109))+[110])
    Ctpg_ind = np.array([5,9,10,11,15,16,17,19,21,23]+list(range(31,38))+[39]+list(range(45,50))+list(range(53,56))+list(range(56,65))
                  +[65,66]+[109])

    save_Var = True
    
    #Costs Specification
    if Costs is None:
        Costs = np.ones(len(Magazine_Pages))

    #Separate Images into Ads and Counterpages
    Ads = []
    Counterpages = []
    Assign_ids = []
    Costs_Ctpg = []

    ad_locations = []
    prod_groups = []
    media_types = []
    ad_elements = []
    ad_embeds = []
    ctpg_embeds = []
    filesize_ads = []
    filesize_ctpgs = []

    if Textboxes is not None:
        ad_textbox = []; ctpg_textbox = []
    
    if Obj_and_Topics is not None:
        ad_num_obj = []; ctpg_num_obj = []
        ad_topic_weight = []; ctpg_topic_weight = []

    double_page_ad_attention = []
    double_page_ad_attention_true = []
    double_page_brand_attention = []
    double_page_brand_attention_true = []
    double_page_brand_share = []
    double_page_brand_share_true = []

    for i, path in enumerate(Magazine_Pages):
        if Magazine_Slots[i] == 2:
            if Textboxes is None:
                textboxes_curr = None
            else:
                textboxes_curr = Textboxes[i]

            if Obj_and_Topics is None:
                obj_and_topics_curr = None
            else:
                obj_and_topics_curr = Obj_and_Topics[i]

            if Method == 'XGBoost':
                ad_attention, brand_attention, brand_share, _ = Predict.Ad_Gaze_Prediction(input_ad_path=path, input_ctpg_path=None, ad_location=None, 
                                                          text_detection_model_path=text_detection_model_path, LDA_model_pth=LDA_model_pth, 
                                                          training_ad_text_dictionary_path=training_ad_text_dictionary_path, training_lang_preposition_path=training_lang_preposition_path, training_language='dutch', 
                                                          ad_embeddings=Ad_embeddings[i].reshape(1,768), ctpg_embeddings=Ctpg_embeddings[i].reshape(1,768),
                                                          Ad_var=None, Ctpg_var=None,
                                                          flag_full_page_ad=True,
                                                          surface_sizes=list(Ad_Element_Sizes[i]), Product_Group=np.array(Ad_Groups[i]), Media_Category=Magazine_Type[i].reshape(1,9),
                                                          filesize_ad=Filesizes_ad[i], filesize_ctpg=Filesizes_ctpg[i],
                                                          Obj_and_Topics=obj_and_topics_curr, TextBoxes=textboxes_curr,
                                                          obj_detection_model_pth=obj_detection_model_pth, num_topic=20, Gaze_Time_Type='ALL', Info_printing=False,
                                                          save_Var=save_Var, Ad_Nr=i, Ctpg_Nr=i, task=task+'_double_Magazine_'+str(Magazine_Nr),
                                                          avgerage_out_index=avg_indices, average_out_data=avg_high_level,
                                                          zeroing_out_index=zero_indices)
                if avg_indices is not None or zero_indices is not None:
                    ad_attention_true, brand_attention_true, brand_share_true, _ = Predict.Ad_Gaze_Prediction(input_ad_path=path, input_ctpg_path=None, ad_location=None, 
                                                          text_detection_model_path=text_detection_model_path, LDA_model_pth=LDA_model_pth, 
                                                          training_ad_text_dictionary_path=training_ad_text_dictionary_path, training_lang_preposition_path=training_lang_preposition_path, training_language='dutch', 
                                                          ad_embeddings=Ad_embeddings[i].reshape(1,768), ctpg_embeddings=Ctpg_embeddings[i].reshape(1,768),
                                                          Ad_var=None, Ctpg_var=None,
                                                          flag_full_page_ad=True,
                                                          surface_sizes=list(Ad_Element_Sizes[i]), Product_Group=np.array(Ad_Groups[i]), Media_Category=Magazine_Type[i].reshape(1,9),
                                                          filesize_ad=Filesizes_ad[i], filesize_ctpg=Filesizes_ctpg[i],
                                                          Obj_and_Topics=obj_and_topics_curr, TextBoxes=textboxes_curr,
                                                          obj_detection_model_pth=obj_detection_model_pth, num_topic=20, Gaze_Time_Type='ALL', Info_printing=False,
                                                          save_Var=save_Var, Ad_Nr=i, Ctpg_Nr=i, task=task+'_double_Magazine_'+str(Magazine_Nr),
                                                          avgerage_out_index=None, average_out_data=None,
                                                          zeroing_out_index=None)
                
            elif Method == 'CNN':
                img_curr = read_image(path)[:,89:921,:].unsqueeze(0)
                ad_img_CNN = img_curr[:,:,:,:640]
                ctpg_img_CNN = img_curr[:,:,:,640:]
                ad_location = torch.tensor([[1,1]])
                ad_attention = Predict.CNN_Prediction(ad_img_CNN, ctpg_img_CNN, ad_location, Gaze_Type='AG').item()
                brand_attention = Predict.CNN_Prediction(ad_img_CNN, ctpg_img_CNN, ad_location, Gaze_Type='BG').item()
                brand_share = Predict.CNN_Prediction(ad_img_CNN, ctpg_img_CNN, ad_location, Gaze_Type='BS').item()
            
            double_page_ad_attention.append(ad_attention/Costs[i])
            double_page_brand_attention.append(brand_attention/Costs[i])
            double_page_brand_share.append(brand_share/Costs[i])

            if avg_indices is not None or zero_indices is not None:
                double_page_ad_attention_true.append(ad_attention_true/Costs[i])
                double_page_brand_attention_true.append(brand_attention_true/Costs[i])
                double_page_brand_share_true.append(brand_share_true/Costs[i])
        else:
            Assign_ids.append(i)
            img_curr = cv.imread(path)
            img_curr = cv.resize(img_curr, (1280,1024))
            _, w, _ = img_curr.shape
            page_width = w // 2
            ad_locations.append(1-Magazine_Slots[i])
            ctpg_location = Magazine_Slots[i]
            ad_img = img_curr[:, (Magazine_Slots[i]*page_width):((Magazine_Slots[i]+1)*page_width)]
            ctpg_img = img_curr[:, (ctpg_location*page_width):((ctpg_location+1)*page_width)]
            Ads.append(ad_img)
            Counterpages.append(ctpg_img)
            prod_groups.append(Ad_Groups[i])
            ad_elements.append(Ad_Element_Sizes[i])
            Costs_Ctpg.append(Costs[i])
            ad_embeds.append(Ad_embeddings[i])
            ctpg_embeds.append(Ctpg_embeddings[i])
            media_types.append(Magazine_Type[i])
            filesize_ads.append(Filesizes_ad[i])
            filesize_ctpgs.append(Filesizes_ctpg[i])

            if Textboxes is not None:
                ad_textbox_curr, ctpg_textbox_curr = Textboxes[i]
                ad_textbox.append(ad_textbox_curr); ctpg_textbox.append(ctpg_textbox_curr)

            if Obj_and_Topics is not None:
                ad_obj_curr, ctpg_obj_curr, ad_topic_curr, ctpg_topic_curr = Obj_and_Topics[i]
                ad_num_obj.append(ad_obj_curr); ctpg_num_obj.append(ctpg_obj_curr)
                ad_topic_weight.append(ad_topic_curr); ctpg_topic_weight.append(ctpg_topic_curr)

    Ad_Attention_Preference = np.zeros((len(Ads),len(Counterpages)))
    Ad_Attention_Preference_true = np.zeros((len(Ads),len(Counterpages)))
    Brand_Attention_Preference = np.zeros((len(Ads),len(Counterpages)))
    Brand_Attention_Preference_true = np.zeros((len(Ads),len(Counterpages)))
    Brand_Share_Preference = np.zeros((len(Ads),len(Counterpages)))
    Brand_Share_Preference_true = np.zeros((len(Ads),len(Counterpages)))

    if length_only:
        return len(Ads)#+len(double_page_ad_attention)

    Ad_Vars = [None]*len(Ads)
    Ctpg_Vars = [None]*len(Counterpages)
    number_of_none_ctpg = len(Counterpages)
    print('There are '+str(len(Ads))+' Ads and '+str(len(Counterpages))+' Counterpages')
    for i, ad in enumerate(Ads):
        print('Ad '+str(i)+" Assigning...")
        if Method == 'CNN':
            ad_images_stack = []
            ctpg_images_stack = []
            ad_locations_stack = []
            
        for j, ctpg in enumerate(Counterpages):
            # print('number of none ctpg: ', number_of_none_ctpg)
            if (j+1)%10 == 0:
                print('Counterpage '+str(j+1)+' Assigning...')
            # if ad_locations[j] == 0:
            #     new_image = np.concatenate((ad,ctpg),axis=1)
            # else:
            #     new_image = np.concatenate((ctpg,ad),axis=1)

            if number_of_none_ctpg == 0 and j == 1:
                Ctpg_input_vars = []
                for k in range(10):
                    Ad_Vars[i][k][:,4] = filesize_ads[i]
                    temp = []
                    for ctpg_num in range(1,len(Counterpages)):
                        Ctpg_Vars[ctpg_num][k][:,-1] = np.linalg.norm(ad_embeds[i]-ctpg_embeds[ctpg_num])
                        Ctpg_Vars[ctpg_num][k][:,-12:-3] = media_types[ctpg_num]
                        Ctpg_Vars[ctpg_num][k][:,-3:-1] = np.array([ad_locations[ctpg_num]])
                        Ctpg_Vars[ctpg_num][k][:,0] = filesize_ctpgs[ctpg_num]
                        temp.append(Ctpg_Vars[ctpg_num][k])
                    Ctpg_input_vars.append(np.concatenate(temp,axis=0))
                    
                # print('Ctpg_Vars: ', Ctpg_input_vars[0].shape)
                ad_attention, brand_attention, brand_share, _ = Predict.Ad_Gaze_Prediction(input_ad_path=ad, input_ctpg_path=ctpg, ad_location=ad_locations[j], 
                                                          text_detection_model_path=text_detection_model_path, LDA_model_pth=LDA_model_pth, 
                                                          training_ad_text_dictionary_path=training_ad_text_dictionary_path, training_lang_preposition_path=training_lang_preposition_path, training_language='dutch', 
                                                          ad_embeddings=ad_embeds[i].reshape(1,768), ctpg_embeddings=ctpg_embeds[j].reshape(1,768),
                                                          Ad_var=Ad_Vars[i], Ctpg_var=Ctpg_input_vars,
                                                          flag_full_page_ad=False,
                                                          surface_sizes=list(ad_elements[i]), Product_Group=np.array(prod_groups[i]), Media_Category=media_types[j].reshape(1,9),
                                                          filesize_ad=Filesizes_ad[i], filesize_ctpg=Filesizes_ctpg[j],
                                                          Obj_and_Topics=obj_and_topics_curr, TextBoxes=textboxes_curr,
                                                          obj_detection_model_pth=obj_detection_model_pth, num_topic=20, Gaze_Time_Type='ALL', Info_printing=False,
                                                          save_Var=save_Var, Ad_Nr=i, Ctpg_Nr=j, task=task+'_nondouble_atonce_Magazine_'+str(Magazine_Nr),
                                                          avgerage_out_index=avg_indices, average_out_data=avg_high_level,
                                                          zeroing_out_index=zero_indices)
                if avg_indices is not None or zero_indices is not None:
                    ad_attention_true, brand_attention_true, brand_share_true, _ = Predict.Ad_Gaze_Prediction(input_ad_path=ad, input_ctpg_path=ctpg, ad_location=ad_locations[j], 
                                                          text_detection_model_path=text_detection_model_path, LDA_model_pth=LDA_model_pth, 
                                                          training_ad_text_dictionary_path=training_ad_text_dictionary_path, training_lang_preposition_path=training_lang_preposition_path, training_language='dutch', 
                                                          ad_embeddings=ad_embeds[i].reshape(1,768), ctpg_embeddings=ctpg_embeds[j].reshape(1,768),
                                                          Ad_var=Ad_Vars[i], Ctpg_var=Ctpg_input_vars,
                                                          flag_full_page_ad=False,
                                                          surface_sizes=list(ad_elements[i]), Product_Group=np.array(prod_groups[i]), Media_Category=media_types[j].reshape(1,9),
                                                          filesize_ad=Filesizes_ad[i], filesize_ctpg=Filesizes_ctpg[j],
                                                          Obj_and_Topics=obj_and_topics_curr, TextBoxes=textboxes_curr,
                                                          obj_detection_model_pth=obj_detection_model_pth, num_topic=20, Gaze_Time_Type='ALL', Info_printing=False,
                                                          save_Var=save_Var, Ad_Nr=i, Ctpg_Nr=j, task=task+'_nondouble_atonce_Magazine_'+str(Magazine_Nr),
                                                          avgerage_out_index=None, average_out_data=None,
                                                          zeroing_out_index=None)
                Ad_Attention_Preference[i,1:] = ad_attention.reshape(len(Counterpages)-1)/np.array(Costs_Ctpg[1:])
                Brand_Attention_Preference[i,1:] = brand_attention.reshape(len(Counterpages)-1)/np.array(Costs_Ctpg[1:])
                Brand_Share_Preference[i,1:] = brand_share.reshape(len(Counterpages)-1)/np.array(Costs_Ctpg[1:])
                
                if avg_indices is not None or zero_indices is not None:
                    Ad_Attention_Preference_true[i,1:] = ad_attention_true.reshape(len(Counterpages)-1)/np.array(Costs_Ctpg[1:])
                    Brand_Attention_Preference_true[i,1:] = brand_attention_true.reshape(len(Counterpages)-1)/np.array(Costs_Ctpg[1:])
                    Brand_Share_Preference_true[i,1:] = brand_share_true.reshape(len(Counterpages)-1)/np.array(Costs_Ctpg[1:])
                
                break

            else:
            
                if Textboxes is not None:
                    textboxes_curr = [ad_textbox[i],ctpg_textbox[j]]
                else:
                    textboxes_curr = None

                if Obj_and_Topics is not None:
                    obj_and_topics_curr = [ad_num_obj[i],ctpg_num_obj[j],ad_topic_weight[i],ctpg_topic_weight[j]]
                else:
                    obj_and_topics_curr = None
                
                if Method == 'XGBoost':
                    ad_attention, brand_attention, brand_share, Var_10 = Predict.Ad_Gaze_Prediction(input_ad_path=ad, input_ctpg_path=ctpg, ad_location=ad_locations[j], 
                                                            text_detection_model_path=text_detection_model_path, LDA_model_pth=LDA_model_pth, 
                                                            training_ad_text_dictionary_path=training_ad_text_dictionary_path, training_lang_preposition_path=training_lang_preposition_path, training_language='dutch', 
                                                            ad_embeddings=ad_embeds[i].reshape(1,768), ctpg_embeddings=ctpg_embeds[j].reshape(1,768),
                                                            Ad_var=Ad_Vars[i], Ctpg_var=Ctpg_Vars[j],
                                                            flag_full_page_ad=False,
                                                            surface_sizes=list(ad_elements[i]), Product_Group=np.array(prod_groups[i]), Media_Category=media_types[j].reshape(1,9),
                                                            filesize_ad=Filesizes_ad[i], filesize_ctpg=Filesizes_ctpg[j],
                                                            Obj_and_Topics=obj_and_topics_curr, TextBoxes=textboxes_curr,
                                                            obj_detection_model_pth=obj_detection_model_pth, num_topic=20, Gaze_Time_Type='ALL', Info_printing=False,
                                                            save_Var=save_Var, Ad_Nr=i, Ctpg_Nr=j, task=task+'_nondouble_1by1_Magazine_'+str(Magazine_Nr),
                                                            avgerage_out_index=avg_indices, average_out_data=avg_high_level,
                                                            zeroing_out_index=zero_indices)
                    if avg_indices is not None or zero_indices is not None:
                        ad_attention_true, brand_attention_true, brand_share_true, Var_10 = Predict.Ad_Gaze_Prediction(input_ad_path=ad, input_ctpg_path=ctpg, ad_location=ad_locations[j], 
                                                            text_detection_model_path=text_detection_model_path, LDA_model_pth=LDA_model_pth, 
                                                            training_ad_text_dictionary_path=training_ad_text_dictionary_path, training_lang_preposition_path=training_lang_preposition_path, training_language='dutch', 
                                                            ad_embeddings=ad_embeds[i].reshape(1,768), ctpg_embeddings=ctpg_embeds[j].reshape(1,768),
                                                            Ad_var=Ad_Vars[i], Ctpg_var=Ctpg_Vars[j],
                                                            flag_full_page_ad=False,
                                                            surface_sizes=list(ad_elements[i]), Product_Group=np.array(prod_groups[i]), Media_Category=media_types[j].reshape(1,9),
                                                            filesize_ad=Filesizes_ad[i], filesize_ctpg=Filesizes_ctpg[j],
                                                            Obj_and_Topics=obj_and_topics_curr, TextBoxes=textboxes_curr,
                                                            obj_detection_model_pth=obj_detection_model_pth, num_topic=20, Gaze_Time_Type='ALL', Info_printing=False,
                                                            save_Var=save_Var, Ad_Nr=i, Ctpg_Nr=j, task=task+'_nondouble_1by1_Magazine_'+str(Magazine_Nr),
                                                            avgerage_out_index=None, average_out_data=None,
                                                            zeroing_out_index=None)
                    if Ad_Vars[i] is None:
                        Ad_Vars[i] = [Var_10[j][0,Ad_ind].reshape(1,-1) for j in range(10)] #Var[0,Ad_ind].reshape(1,-1)
                    if number_of_none_ctpg > 0:
                        number_of_none_ctpg -= 1
                        Ctpg_Vars[j] = [Var_10[j][0,Ctpg_ind].reshape(1,-1) for j in range(10)]

                    Ad_Attention_Preference[i,j] = ad_attention/Costs_Ctpg[j]
                    Brand_Attention_Preference[i,j] = brand_attention/Costs_Ctpg[j]
                    Brand_Share_Preference[i,j] = brand_share/Costs_Ctpg[j]

                    if avg_indices is not None or zero_indices is not None:
                        Ad_Attention_Preference_true[i,j] = ad_attention_true/Costs_Ctpg[j]
                        Brand_Attention_Preference_true[i,j] = brand_attention_true/Costs_Ctpg[j]
                        Brand_Share_Preference_true[i,j] = brand_share_true/Costs_Ctpg[j]

                elif Method == 'CNN':
                    ad_img_CNN = torch.tensor(ad).permute(2,0,1).unsqueeze(0)[:,:,89:921,:]
                    ad_images_stack.append(ad_img_CNN)
                    ctpg_img_CNN = torch.tensor(ctpg).permute(2,0,1).unsqueeze(0)[:,:,89:921,:]
                    ctpg_images_stack.append(ctpg_img_CNN)
                    ad_locations_stack.append(torch.tensor([[1,0]]))
                    # ad_attention = Predict.CNN_Prediction(ad_img_CNN, ctpg_img_CNN, ad_location, Gaze_Type='AG').item()
                    # brand_attention = Predict.CNN_Prediction(ad_img_CNN, ctpg_img_CNN, ad_location, Gaze_Type='BG').item()

        if Method == 'CNN':
            ad_images_stack = torch.cat(ad_images_stack,dim=0)
            ctpg_images_stack = torch.cat(ctpg_images_stack,dim=0)
            ad_locations_stack = torch.cat(ad_locations_stack,dim=0)
            ad_attentions = Predict.CNN_Prediction(ad_images_stack, ctpg_images_stack, ad_locations_stack, Gaze_Type='AG').to('cpu').squeeze()
            brand_attentions = Predict.CNN_Prediction(ad_images_stack, ctpg_images_stack, ad_locations_stack, Gaze_Type='BG').to('cpu').squeeze()
            brand_shares = Predict.CNN_Prediction(ad_images_stack, ctpg_images_stack, ad_locations_stack, Gaze_Type='BS').to('cpu').squeeze()
            Ad_Attention_Preference[i] = ad_attentions.numpy()/np.array(Costs_Ctpg)
            Brand_Attention_Preference[i] = brand_attentions.numpy()/np.array(Costs_Ctpg)
            Brand_Share_Preference[i] = brand_shares.numpy()/np.array(Costs_Ctpg)

    result = [Ad_Attention_Preference, Brand_Attention_Preference, Brand_Share_Preference, 
              double_page_ad_attention, double_page_brand_attention, double_page_brand_share, Assign_ids,
              Ad_Attention_Preference_true, Brand_Attention_Preference_true, Brand_Share_Preference_true,
              double_page_ad_attention_true, double_page_brand_attention_true, double_page_brand_share_true]

    return result

def Preference_Matrix_different_magazine(Magzine_Target, Magzine_Ad, 
                                         Magazine_Slots_Target, Magazine_Slots_Ad, 
                                         Ad_Groups, Ad_Element_Sizes,
                                         Ctpg_embeddings_Target,
                                         Ad_embeddings_Ad,
                                         Textboxes_Target=None, Textboxes_Ad=None,
                                         Obj_and_Topics_Target=None, Obj_and_Topics_Ad=None,
                                         Magazine_Type_Target=None,
                                         Costs=None, length_only=False,
                                         Method='XGBoost'):
    #Separate Images into Ads and Counterpage
    Ads = []
    Counterpages = []
    Assign_ids_target = []
    Assign_ids_ad = []

    ad_locations = []
    media_types = []
    prod_groups = []
    ad_elements = []
    ad_embeds = []
    ctpg_embeds = []

    if Textboxes_Target is not None:
        ad_textbox = []; ctpg_textbox = []
    
    if Obj_and_Topics_Target is not None:
        ad_num_obj = []; ctpg_num_obj = []
        ad_topic_weight = []; ctpg_topic_weight = []

    double_page_ad_attention = []
    double_page_brand_attention = []
    double_page_brand_share = []

    #Target magazine (Counterpage)
    for i, path in enumerate(Magzine_Target):
        if Magazine_Slots_Target[i] == 2:
            continue
        else:
            Assign_ids_target.append(i)
            img_curr = cv.imread(path)
            img_curr = cv.resize(img_curr, (1280,1024))
            _, w, _ = img_curr.shape
            page_width = w // 2
            ad_locations.append(Magazine_Slots_Target[i])
            ctpg_location = 1-Magazine_Slots_Target[i]
            ctpg_img = img_curr[:, (ctpg_location*page_width):((ctpg_location+1)*page_width)]
            Counterpages.append(ctpg_img)
            media_types.append(Magazine_Type_Target[i])
            ctpg_embeds.append(Ctpg_embeddings_Target[i])

            if Textboxes_Target is not None:
                _, ctpg_textbox_curr = Textboxes_Target[i]
                ctpg_textbox.append(ctpg_textbox_curr)

            if Obj_and_Topics_Target is not None:
                _, ctpg_obj_curr, _, ctpg_topic_curr = Obj_and_Topics_Target[i]
                ctpg_num_obj.append(ctpg_obj_curr)
                ctpg_topic_weight.append(ctpg_topic_curr)

    #Ad magazine (Ads)
    for i, path in enumerate(Magzine_Ad):
        if Magazine_Slots_Ad[i] == 2:
            continue
        else:
            Assign_ids_ad.append(i)
            img_curr = cv.imread(path)
            img_curr = cv.resize(img_curr, (1280,1024))
            _, w, _ = img_curr.shape
            page_width = w // 2
            ad_img = img_curr[:, (Magazine_Slots_Ad[i]*page_width):((Magazine_Slots_Ad[i]+1)*page_width)]
            Ads.append(ad_img)
            prod_groups.append(Ad_Groups[i])
            ad_elements.append(Ad_Element_Sizes[i])
            ad_embeds.append(Ad_embeddings_Ad[i])

            if Textboxes_Ad is not None:
                ad_textbox_curr, _ = Textboxes_Ad[i]
                ad_textbox.append(ad_textbox_curr)

            if Obj_and_Topics_Ad is not None:
                ad_obj_curr, _, ad_topic_curr, _ = Obj_and_Topics_Ad[i]
                ad_num_obj.append(ad_obj_curr)
                ad_topic_weight.append(ad_topic_curr)

    #Check costs on Ad position
    if Costs is None:
        Costs = np.ones(len(Counterpages))

    #Matrix
    if len(Ads) > len(Counterpages):
        Ads = Ads[:len(Counterpages)]
    elif len(Ads) < len(Counterpages):
        Counterpages = Counterpages[:len(Ads)]
    
    Ad_Attention_Preference = np.zeros((len(Ads),len(Counterpages)))
    Brand_Attention_Preference = np.zeros((len(Ads),len(Counterpages)))
    Brand_Share_Preference = np.zeros((len(Ads),len(Counterpages)))

    if length_only:
        return len(Ads)

    for i, ad in enumerate(Ads):
        print('Ad '+str(i)+" Assigning...")
        if Method == 'CNN':
            ad_images_stack = []
            ctpg_images_stack = []
            ad_locations_stack = []
        for j, ctpg in enumerate(Counterpages):
            
            # if ad_locations[j] == 0:
            #     new_image = np.concatenate((ad,ctpg),axis=1)
            # else:
            #     new_image = np.concatenate((ctpg,ad),axis=1)
            
            if Textboxes_Target is not None:
                textboxes_curr = [ad_textbox[i],ctpg_textbox[j]]
            else:
                textboxes_curr = None

            if Obj_and_Topics_Target is not None:
                obj_and_topics_curr = [ad_num_obj[i],ctpg_num_obj[j],ad_topic_weight[i],ctpg_topic_weight[j]]
            else:
                obj_and_topics_curr = None
            
            if Method == 'XGBoost':
                ad_attention, brand_attention, brand_share, _ = Predict.Ad_Gaze_Prediction(input_ad_path=ad, input_ctpg_path=ctpg, ad_location=ad_locations[j], 
                                                            text_detection_model_path=text_detection_model_path, LDA_model_pth=LDA_model_pth, 
                                                            training_ad_text_dictionary_path=training_ad_text_dictionary_path, training_lang_preposition_path=training_lang_preposition_path, training_language='dutch', 
                                                            ad_embeddings=ad_embeds[i].reshape(1,768), ctpg_embeddings=ctpg_embeds[j].reshape(1,768),
                                                            surface_sizes=list(ad_elements[i]), Product_Group=np.array(prod_groups[i]), Media_Category=media_types[j].reshape(1,9),
                                                            Obj_and_Topics=obj_and_topics_curr, TextBoxes=textboxes_curr,
                                                            obj_detection_model_pth=obj_detection_model_pth, num_topic=20, Gaze_Time_Type='ALL', Info_printing=False)
                Ad_Attention_Preference[i,j] = ad_attention/Costs[j]
                Brand_Attention_Preference[i,j] = brand_attention/Costs[j]
                Brand_Share_Preference[i,j] = brand_share/Costs[j]

            elif Method == 'CNN':
                ad_img_CNN = torch.tensor(ad).permute(2,0,1).unsqueeze(0)[:,:,89:921,:]
                ad_images_stack.append(ad_img_CNN)
                ctpg_img_CNN = torch.tensor(ctpg).permute(2,0,1).unsqueeze(0)[:,:,89:921,:]
                ctpg_images_stack.append(ctpg_img_CNN)
                ad_locations_stack.append(torch.tensor([[1,0]]))

        if Method == 'CNN':
            ad_images_stack = torch.cat(ad_images_stack,dim=0)
            ctpg_images_stack = torch.cat(ctpg_images_stack,dim=0)
            ad_locations_stack = torch.cat(ad_locations_stack,dim=0)
            ad_attentions = Predict.CNN_Prediction(ad_images_stack, ctpg_images_stack, ad_locations_stack, Gaze_Type='AG').to('cpu').squeeze()
            brand_attentions = Predict.CNN_Prediction(ad_images_stack, ctpg_images_stack, ad_locations_stack, Gaze_Type='BG').to('cpu').squeeze()
            brand_shares = Predict.CNN_Prediction(ad_images_stack, ctpg_images_stack, ad_locations_stack, Gaze_Type='BS').to('cpu').squeeze()
            Ad_Attention_Preference[i] = ad_attentions.numpy()
            Brand_Attention_Preference[i] = brand_attentions.numpy()
            Brand_Share_Preference[i] = brand_shares.numpy()
            
    return Ad_Attention_Preference, Brand_Attention_Preference, Brand_Share_Preference, double_page_ad_attention, double_page_brand_attention, double_page_brand_share, Assign_ids_ad, Assign_ids_target

def Assignment_Problem(costs, workers, jobs):
    #https://machinelearninggeek.com/solving-assignment-problem-using-linear-programming-in-python/

    prob = LpProblem("Assignment Problem", LpMinimize) 

    # The cost data is made into a dictionary
    costs= makeDict([workers, jobs], costs, 0)

    # Creates a list of tuples containing all the possible assignments
    assign = [(w, j) for w in workers for j in jobs]

    # A dictionary called 'Vars' is created to contain the referenced variables
    vars = LpVariable.dicts("Assign", (workers, jobs), 0, None, LpBinary)

    # The objective function is added to 'prob' first
    prob += (
        lpSum([vars[w][j] * costs[w][j] for (w, j) in assign]),
        "Sum_of_Assignment_Costs",
    )

    # There are row constraints. Each job can be assigned to only one employee.
    for j in jobs:
        prob+= lpSum(vars[w][j] for w in workers) == 1

    # There are column constraints. Each employee can be assigned to only one job.
    for w in workers:
        prob+= lpSum(vars[w][j] for j in jobs) == 1

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    return prob

