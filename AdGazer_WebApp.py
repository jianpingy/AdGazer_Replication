import sys
import os
new_dir = os.path.join(os.getcwd(), "src")
sys.path.append(new_dir)

import gradio as gr
from gradio_image_prompter import ImagePrompter
import Predict
import XGBoost_utils
import numpy as np
import cv2 as cv
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch
from PIL import Image
from gradio_pdf import PDF
from pdf2image import convert_from_path
from pathlib import Path

GENERAL_CATEGORY = {'Potatoes / Vegetables / Fruit': 0, 'Chemical products': 1, 'Photo / Film / Optical items': 2, 'Catering industry': 3, 'Industrial products other': 4, 'Media': 5, 'Real estate': 6, 'Government': 7, 'Personnel advertisements': 8, 'Cars / Commercial vehicles': 9, 'Cleaning products': 10, 'Retail': 11, 'Fragrances': 12, 'Footwear / Leather goods': 13, 'Software / Automation': 14, 'Telecommunication equipment': 15, 'Tourism': 16, 'Transport/Communication companies': 17, 'Transport services': 18, 'Insurances': 19, 'Meat / Fish / Poultry': 20, 'Detergents': 21, 'Foods General': 22, 'Other services': 23, 'Banks and Financial Services': 24, 'Office Products': 25, 'Household Items': 26, 'Non-alcoholic beverages': 27, 'Hair, Oral and Personal Care': 28, 'Fashion and Clothing': 29, 'Other products and Services': 30, 'Paper products': 31, 'Alcohol and Other Stimulants': 32, 'Medicines': 33, 'Recreation and Leisure': 34, 'Electronics': 35, 'Home Furnishings': 36, 'Products for Business Use': 37}
CATEGORIES = list(GENERAL_CATEGORY.keys())
CATEGORIES.sort()
LOCATIONS = ['Left', 'Right', 'Full']
GAZE_TYPE = ['Ad', 'Brand']

def calculate_areas(prompts, brand_num, pictorial_num, text_num):
    image_entire = prompts["image"]
    w, h = image_entire.size
    image_entire = np.array(image_entire.convert('RGB'))
    points_all = prompts["points"]
    brand_surf = 0
    for i in range(brand_num):
        x1 = points_all[i][0]; y1 = points_all[i][1]
        x2 = points_all[i][3]; y2 = points_all[i][4]
        brand_surf += np.abs((x1-x2)*(y1-y2))

    pictorial_surf = 0
    for i in range(brand_num, brand_num+pictorial_num):
        x1 = points_all[i][0]; y1 = points_all[i][1]
        x2 = points_all[i][3]; y2 = points_all[i][4]
        pictorial_surf += np.abs((x1-x2)*(y1-y2))
    
    text_surf = 0
    for i in range(brand_num+pictorial_num, brand_num+pictorial_num+text_num):
        x1 = points_all[i][0]; y1 = points_all[i][1]
        x2 = points_all[i][3]; y2 = points_all[i][4]
        text_surf += np.abs((x1-x2)*(y1-y2))

    ad_size = 0
    x1 = points_all[-1][0]; y1 = points_all[-1][1]
    x2 = points_all[-1][3]; y2 = points_all[-1][4]
    ad_size += np.abs((x1-x2)*(y1-y2))
    ad_image = image_entire[int(y1):int(y2), int(x1):int(x2), :]
    left_margin = x1; right_margin = w-x2
    if left_margin <=100 and right_margin <= 100:
        upper_margin = y1; lower_margin = h-y2
        if upper_margin <= 100 and lower_margin <= 100:
            context_image = None
        else:
            if upper_margin >= lower_margin:
                context_image = image_entire[:int(y1), :, :]
            else:
                context_image = image_entire[int(y2):, :, :]
    else:
        if left_margin >= right_margin:
            context_image = image_entire[:, :int(x1), :]
        else:
            context_image = image_entire[:, int(x2):, :]

    whole_size = 0
    whole_size += w*h

    return (brand_surf/whole_size*100, pictorial_surf/whole_size*100, text_surf/whole_size*100, ad_size/whole_size*100, ad_image, context_image)

def convert(note, doc):
    print(doc)
    img = convert_from_path(doc)[0]
    img.save(f'pdf_to_imgs/pdf_img.png', 'PNG')
    return 'Done!', gr.DownloadButton(label='Download converted image', value='pdf_to_imgs/pdf_img.png')

def attention(note, button1, button2,
              whole_display_prompt, 
              brand_num, pictorial_num, text_num, check,
              category, ad_location, gaze_type):
    text_detection_model_path = 'src/EAST-Text-Detection/frozen_east_text_detection.pb'
    LDA_model_pth = 'LDA_Model_trained/lda_model_best_tot.model'
    training_ad_text_dictionary_path = 'LDA_Model_trained/object_word_dictionary'
    training_lang_preposition_path = 'LDA_Model_trained/dutch_preposition'

    prod_group = np.zeros(38)
    prod_group[GENERAL_CATEGORY[category]] = 1

    if not check:
        print('No ad bounding box available!!')
        return -1, None

    if ad_location == 'left':
        ad_loc = 0
    elif ad_location == 'right':
        ad_loc = 1
    else:
        ad_loc = None

    brand_percent, visual_percent, text_percent, adv_size_percent, ad_image, context_image = calculate_areas(whole_display_prompt, brand_num, pictorial_num, text_num)
    surfaces = [brand_percent, visual_percent, text_percent, adv_size_percent*10/100]

    #### Note: The following lines are commented out because they require GPU and additional resources to run.
    # caption_ad = XGBoost_utils.Caption_Generation(Image.fromarray(np.uint8(ad_image)))
    # if context_image is not None:
    #     caption_context = XGBoost_utils.Caption_Generation(Image.fromarray(np.uint8(context_image)))
    # else:
    #     caption_context = ''
    # ad_topic = XGBoost_utils.Topic_emb(caption_ad)
    # ctpg_topic = XGBoost_utils.Topic_emb(caption_context)
    np.random.seed(42)
    ad_topic = np.random.randn(1,768)
    ctpg_topic = np.random.randn(1,768)

    ad = cv.resize(ad_image, (640, 832))
    print('ad shape: ', ad.shape)
    if context_image is None:
        context = None
    else:
        context = cv.resize(context_image, (640, 832))

    adv_imgs = torch.permute(torch.tensor(ad), (2,0,1)).unsqueeze(0)
    if context is None:
        ctpg_imgs = torch.zeros_like(adv_imgs)
    else:
        ctpg_imgs = torch.permute(torch.tensor(context), (2,0,1)).unsqueeze(0)
    ad_locations = torch.tensor([1,0]).unsqueeze(0)
    heatmap = Predict.HeatMap_CNN(adv_imgs, ctpg_imgs, ad_locations, Gaze_Type='AG')

    Gaze = Predict.Ad_Gaze_Prediction(input_ad_path=ad, input_ctpg_path=context, ad_location=ad_loc,
                                    text_detection_model_path=text_detection_model_path, LDA_model_pth=LDA_model_pth, 
                                    training_ad_text_dictionary_path=training_ad_text_dictionary_path, training_lang_preposition_path=training_lang_preposition_path, training_language='dutch', 
                                    Ad_var=None, Ctpg_var=None,
                                    flag_full_page_ad=False,
                                    ad_embeddings=ad_topic, ctpg_embeddings=ctpg_topic,
                                    surface_sizes=surfaces, Product_Group=prod_group,
                                    obj_detection_model_pth=None, num_topic=20, Gaze_Time_Type=gaze_type, Ad_Features_Only=False, Info_printing=False)
    return np.round(Gaze[0],2), Image.fromarray(np.flip(heatmap, axis=2))

with gr.Blocks() as demo:
    gr.Markdown("""
                <div style='text-align: center; padding: 10px; font-size:40px'>
                    <p> <b>Gazer 1.0: Ad Attention Prediction</b> </p>
                </div>
                """)
    gr.Markdown("""
                This app accompanies: "Contextual Advertising with Theory-Informed Machine Learning", manuscript submitted to the Journal of Marketing.  
                App Version: 1.0, Date: 10/24/2024.  
                Note: Gazer 1.0 does not yet include LLM generated ad topics. Future updates will include this in a GPU environment.
                """)
    gr.Interface(
        fn=convert,
        inputs=[gr.Markdown("""
                            <div style='font-size:20px'>
                                <p> <b>If you only have a pdf image file, first convert it here to png file and download:</b> </p>
                            </div>
                           
                           """),
                PDF(label="PDF Converter")],
        outputs=[gr.Text(label='Progress'), gr.DownloadButton(label='Wait to be downloadable', value=None)]
    )
    
    gr.Interface(
        fn=attention,
        inputs=[gr.Markdown("""
                ## Instructions:  
                0. The screen size should remain the same during processing.
                1. Click to upload or drag the entire image (jpg/jpeg/png file) that contains BOTH ad and its context;  
                2. Draw bounding boxes in the order of: (each element can have more than 1 boxes; remember the number of boxes for each element you draw)  
                &nbsp;&nbsp;&nbsp;(a) Brand element(s) (skip if N.A.)  
                &nbsp;&nbsp;&nbsp;(b) Pictorial element(s), e.g. Objects, Person etc (skip if N.A.)  
                &nbsp;&nbsp;&nbsp;(c) Text element(s) (skip if N.A.)  
                &nbsp;&nbsp;&nbsp;(d) The advertisement.  
                3. Put in number of bounding boxes for each element, product category, ad location and attention type.  

                ***NOTE:*** *ResNet50 Heatmap could take around 20-80 seconds under current CPU environment.*
                    
                Two example ads are avialable for download: """),
                gr.DownloadButton(label="Download Example Image 1 of Ad and Context", value='Demo/Ad_Example1.jpg'),
                gr.DownloadButton(label="Download Example Image 2 of Ad and Context", value='Demo/Ad_Example2.jpg'),
                ImagePrompter(label="Upload Entire (Ad+Context) Image in jpg/jpeg/png format, and Draw Bounding Boxes", sources=['upload'], type="pil"),
                gr.Number(label="Number of brand bounding boxes drawn"),
                gr.Number(label="Number of pictorial bounding boxes drawn"),
                gr.Number(label="Number of text bounding boxes drawn"),
                gr.Checkbox(label="Check if you draw a bounding box for the entire ad (Note: this is a must-do)"),
                gr.Dropdown(CATEGORIES, label="Product Category"),
                gr.Dropdown(LOCATIONS, label='Ad Location'),
                gr.Dropdown(GAZE_TYPE, label='Gaze Type')
                ],
        outputs=[gr.Number(label="Predicted Gaze (sec). If you see a value of -1, it means no ad bounding box is drawn!!"),
                 gr.Image(label="ResNet50 Heatmap (Hotter/Redder regions show more pixel contribution.)")],
        title=None,
        description=None,
        theme=gr.themes.Soft()
    )
    gr.Markdown(
            """
            <div style='text-align: center; padding: 1px;'>
                <p>Copyright © 2024 Manuscript Authors. All Rights Reserved.</p>
                <p>Disclaimer: This app is provided for free and for academic use only. The authors take no responsibility for your use of the information contained in or linked from these web pages.</p>
            </div>
            """
        )

demo.launch(share=False)