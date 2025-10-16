import sys
import os

new_dir = os.path.join(os.getcwd(), "src")
sys.path.append(new_dir)

import warnings
warnings.filterwarnings("ignore")
import Predict
import XGBoost_utils
import torch
import numpy as np
import os
from os.path import isfile, isdir, join

text_detection_model_path = os.path.join(new_dir, 'EAST-Text-Detection/frozen_east_text_detection.pb')
Adversarial_types = ['Newspaper', 'Banner', 'Multiple Ads', 'Outdoor', 'Others']
mypath = os.path.join(os.getcwd(), 'Out-of-Distribution_Samples')

ad_locations_total = {
    'Newspaper': [0,0,0,0,0,0,0,0,0,0,0],
    'Banner': [2,2,2,2,2,2,2,2,2,2,2],
    'Multiple Ads': [0,0,1,0,0,1,0,0,0,0,0],
    'Outdoor': [2,2,2,2,2,2,2,2,2,2,2],
    'Others': [1,0,1,0,2,0,1,1,0,0,0]
}

Predicted_AG = {}
Predicted_BG = {}
Predicted_BS = {}
GT_AG = {}
GT_BG = {}
GT_BS = {}

for f in os.listdir(mypath):
    # if f != 'Outdoor':
    #     continue
    if isdir(join(mypath, f)):
        print('Currently processing samples of type '+f+'......')
        path_temp = join(mypath, f)
        surfaces = torch.load(join(path_temp,f+'_Adversarial_Surfaces'))
        for i in range(len(surfaces)):
            surfaces[i][-1] = np.log(surfaces[i][-1]+1e-3) 
        categories = torch.load(join(path_temp,f+'_Adversarial_Categories'))
        ad_locations_curr = ad_locations_total[f]
        ad_embeddings = torch.load(join(path_temp,f+'_ad_topic_embeddings'))
        ctpg_embeddings = torch.load(join(path_temp,f+'_ctpg_topic_embeddings'))
        media_categories = torch.load(join(path_temp,f+'_Adversarial_Media_Categories'))
        GT_AG[f] = torch.load(join(path_temp,'AGs'))
        GT_BG[f] = torch.load(join(path_temp,'BGs'))
        GT_BS[f] = np.array(GT_BG[f])/np.array(GT_AG[f])

        #f is, e.g. Outdoor
        #sub_f is, e.g. 1,2,...,11
        AG_predictions_per_type = np.zeros(11)
        BG_predictions_per_type = np.zeros(11)
        BS_predictions_per_type = np.zeros(11)
        for sub_f in os.listdir(path_temp):
            if isdir(join(path_temp, sub_f)):
                print('Sample Number '+sub_f+'...... ')
                context_pth = None
                for imgs in os.listdir(join(path_temp, sub_f)):
                    jpg_removed_split = imgs[:-4].split(' ')
                    if len(jpg_removed_split) > 1:
                        img_type = jpg_removed_split[-1]
                        if img_type == 'Ad':
                            ad_pth = join(join(path_temp, sub_f),imgs)
                            print(ad_pth)
                        elif img_type == 'Context':
                            context_pth = join(join(path_temp, sub_f),imgs)
                
                #Start Predicting
                sample_num = int(sub_f)-1
                
                AG,_ = Predict.Ad_Gaze_Prediction(input_ad_path=ad_pth, input_ctpg_path=context_pth, text_detection_model_path=text_detection_model_path, LDA_model_pth=None, 
                     training_ad_text_dictionary_path=None, training_lang_preposition_path=None, training_language='dutch', 
                     ad_embeddings=ad_embeddings[sample_num].reshape(1,768), ctpg_embeddings=ctpg_embeddings[sample_num].reshape(1,768),
                     surface_sizes=list(surfaces[sample_num]), Product_Group=categories[sample_num], Media_Category=media_categories[sample_num].reshape(1,9),
                     obj_detection_model_pth=None, ad_location=ad_locations_curr[sample_num], num_topic=20, Gaze_Time_Type='Ad')
                AG_predictions_per_type[sample_num] = AG

                BG,_ = Predict.Ad_Gaze_Prediction(input_ad_path=ad_pth, input_ctpg_path=context_pth, text_detection_model_path=text_detection_model_path, LDA_model_pth=None, 
                     training_ad_text_dictionary_path=None, training_lang_preposition_path=None, training_language='dutch', 
                     ad_embeddings=ad_embeddings[sample_num].reshape(1,768), ctpg_embeddings=ctpg_embeddings[sample_num].reshape(1,768),
                     surface_sizes=list(surfaces[sample_num]), Product_Group=categories[sample_num], Media_Category=media_categories[sample_num].reshape(1,9),
                     obj_detection_model_pth=None, ad_location=ad_locations_curr[sample_num], num_topic=20, Gaze_Time_Type='Brand')
                BG_predictions_per_type[sample_num] = BG

                BS,_ = Predict.Ad_Gaze_Prediction(input_ad_path=ad_pth, input_ctpg_path=context_pth, text_detection_model_path=text_detection_model_path, LDA_model_pth=None, 
                     training_ad_text_dictionary_path=None, training_lang_preposition_path=None, training_language='dutch', 
                     ad_embeddings=ad_embeddings[sample_num].reshape(1,768), ctpg_embeddings=ctpg_embeddings[sample_num].reshape(1,768),
                     surface_sizes=list(surfaces[sample_num]), Product_Group=categories[sample_num], Media_Category=media_categories[sample_num].reshape(1,9),
                     obj_detection_model_pth=None, ad_location=ad_locations_curr[sample_num], num_topic=20, Gaze_Time_Type='BS')
                BS_predictions_per_type[sample_num] = BS

        Predicted_AG[f] = AG_predictions_per_type
        Predicted_BG[f] = BG_predictions_per_type
        Predicted_BS[f] = BS_predictions_per_type

print("Final results: ")
diffs_rmse = {}
diffs_rmsrpd = {}
GT_AG_tot = []
GT_BG_tot = []
GT_BS_tot = []
Pred_AG_tot = []
Pred_BG_tot = []
Pred_BS_tot = []
for key in Predicted_AG.keys():
    print(key)
    print('AGs', Predicted_AG[key])
    print('BGs', Predicted_BG[key])
    print('BSs', Predicted_BS[key])
    Pred_AG_tot.append(Predicted_AG[key])
    Pred_BG_tot.append(Predicted_BG[key])
    Pred_BS_tot.append(Predicted_BS[key])
    GT_AG_tot.append(GT_AG[key])
    GT_BG_tot.append(GT_BG[key])
    GT_BS_tot.append(GT_BS[key])

    rmse1 = np.sqrt(np.mean((GT_AG[key]-Predicted_AG[key])**2))
    rmse2 = np.sqrt(np.mean((GT_BG[key]-Predicted_BG[key])**2))
    rmse3 = np.sqrt(np.mean((GT_BS[key]-Predicted_BS[key])**2))
    diffs_rmse[key] = (rmse1, rmse2, rmse3)

    rmsrpd1 = XGBoost_utils.RMSRPD(GT_AG[key],Predicted_AG[key])
    rmsrpd2 = XGBoost_utils.RMSRPD(GT_BG[key],Predicted_BG[key])
    rmsrpd3 = XGBoost_utils.RMSRPD(GT_BS[key],Predicted_BS[key])
    diffs_rmsrpd[key] = (rmsrpd1, rmsrpd2, rmsrpd3)

    print()

Pred_AG_tot = np.concatenate(Pred_AG_tot)
Pred_BG_tot = np.concatenate(Pred_BG_tot)
Pred_BS_tot = np.concatenate(Pred_BS_tot)
GT_AG_tot = np.concatenate(GT_AG_tot)
GT_BG_tot = np.concatenate(GT_BG_tot)
GT_BS_tot = np.concatenate(GT_BS_tot)
print("RMSE: ")
print("Total AG: ", np.sqrt(np.mean((Pred_AG_tot-GT_AG_tot)**2)))
print("Total BG: ", np.sqrt(np.mean((Pred_BG_tot-GT_BG_tot)**2)))
print("Total BS: ", np.sqrt(np.mean((Pred_BS_tot-GT_BS_tot)**2)))
print()
for key in diffs_rmse.keys():
    print(key, diffs_rmse[key][0], diffs_rmse[key][1], diffs_rmse[key][2])
print()

print("RMSRPD: ")
print("Total AG: ", XGBoost_utils.RMSRPD(Pred_AG_tot,GT_AG_tot))
print("Total BG: ", XGBoost_utils.RMSRPD(Pred_BG_tot,GT_BG_tot))
print("Total BS: ", XGBoost_utils.RMSRPD(Pred_BS_tot,GT_BS_tot))
print()
for key in diffs_rmsrpd.keys():
    print(key, diffs_rmsrpd[key][0], diffs_rmsrpd[key][1], diffs_rmsrpd[key][2])
print()

print("Correlation: ")
print("Total AG: ", np.corrcoef(Pred_AG_tot, GT_AG_tot))
print("Total BG: ", np.corrcoef(Pred_BG_tot, GT_BG_tot))
print("Total BS: ", np.corrcoef(Pred_BS_tot, GT_BS_tot))
