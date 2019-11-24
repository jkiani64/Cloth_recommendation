from selenium import webdriver
import os
import ast
import requests
import time
import urllib
from bs4 import BeautifulSoup as Soup
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
from tqdm import tqdm

def getting_image_url(URL,directory, cloth_type):
    '''
    To get the url for each image
    '''
    driver = webdriver.Chrome(ChromeDriverManager().install())
    Images_info = []
    temp_image_name = {}
    image_id = 1
    # To start driver
    driver.get(URL)
    #To create the directory
    if not os.path.isdir(directory):
        os.mkdir(directory)
    # Find all elements in product-container
    product_container_ls = driver.find_elements_by_class_name(
        'product-container')
    #Loop over each product
    for prd in product_container_ls:
        # Locating Elements by CSS Selectors
        product_lm = prd.find_element_by_css_selector('a')

        # Getting the url for the product
        product_url = product_lm.get_attribute('href')

        # Finding elements of images by class name
        image_lm = prd.find_element_by_class_name(r'main')
        #image_lm = prd.get_attribute('data-original')

        # The url to image
        #image_url = image_lm.get_attribute('src')
        #image_url = image_lm.get_attribute('data-original')
        #print(image_id, ': ', image_url)

        # The name of image
        image_name = image_lm.get_attribute('title').replace(' ', '_')

        # Check whether the name is avaliable or not
        if temp_image_name.get(image_name, 0) >= 1:
            temp_image_name[image_name] += 1
            image_name = image_name + '_' + \
                str(temp_image_name[image_name] - 1)
        else:
            temp_image_name[image_name] = 1

        # Image Path
        image_path = os.path.join(directory, f'{image_name}')

        # Making a a dict to save the results
        temp = {'image_id': cloth_type + str(image_id),
                'image_name': image_name,
                'product_url': product_url,
                'image_path': image_path}

        # Appending the info of the image
        Images_info.append(temp)
        image_id += 1
        # time.sleep(3)
    driver.quit()
    df_cloth = pd.DataFrame(Images_info)
    df_cloth.to_csv(directory + '/' + cloth_type + '.csv')
    return Images_info

def getting_images(img_dict):
    '''
    To get images for different product!
    '''
    driver = webdriver.Chrome(ChromeDriverManager().install())
    for img_info in tqdm(img_dict):
        try:
            img_names = {}
            url = img_info['product_url']
            directory = img_info['image_path']
            image_id = img_info['image_id']
            # To create the directory
            if not os.path.isdir(directory):
                os.mkdir(directory)
            driver.get(url)
            productcontainer = driver.find_element_by_class_name(
                'productcontainer')
            for img in productcontainer.find_elements_by_tag_name('img'):
                preview_url = img.get_attribute('src').split("?preset")[0]
                if preview_url[-3:] == 'jpg':
                    try:
                        title_img = img.get_attribute('title').split()[0]
                    except:
                        continue
                    if img_names.get(title_img, 0) > 0:
                        img_names[title_img] += 1
                        title_img = title_img + '_' +\
                            str(img_names[title_img] - 1)
                    else:
                        img_names[title_img] = 1

                    # Image Path
                    image_path = os.path.join(directory, f'{title_img}.jpg')
                    # Getting and saving the image
                    urllib.request.urlretrieve(preview_url, image_path)
        except:
            print(f'{image_id} is not avalaible!')
    driver.quit()
    print('All images are downloaded and saved')
    
# directory = 'data/venus/women/tops/long_sleeve'
# if not os.path.isdir(directory):
#     os.mkdir(directory)

# driver = webdriver.Chrome(ChromeDriverManager().install())
# for idx in range(2):
#     img_names = {}
#     url = Images_info[idx]['product_url']
#     image_name = Images_info[idx]['product_url']
#     driver.get(url)
#     productcontainer = driver.find_element_by_class_name('productcontainer')
#     for img in productcontainer.find_elements_by_tag_name('img'):
#         preview_url = img.get_attribute('src').split("?preset")[0]
#         if preview_url[-3:] == 'jpg':
#             try:
#                 title_img = img.get_attribute('title').split()[0]
#             except:
#                 continue

#             if img_names.get(title_img, 0) > 0:
#                 img_names[title_img] += 1
#                 title_img = title_img + '_' +\
#                     str(img_names[title_img] - 1)
#             else:
#                 img_names[title_img] = 1
#             # Image Path
#             image_path = os.path.join(directory, f'{title_img}.jpg')
            
#             # Getting and saving the image
#             urllib.request.urlretrieve(image_url, image_path)
# driver.quit()