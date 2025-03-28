import shutil
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
from PIL import Image
import time
import os
from collections import defaultdict
import difflib
service = Service(executable_path="chromedriver.exe")
DRIVER = webdriver.Chrome(service=service)
DATA_PATH = 'datasets/logos/test'
SLEEP = 0.3

def is_similar(str1, str2):
    sequence_matcher = difflib.SequenceMatcher(None, str1, str2)
    similarity_ratio = sequence_matcher.ratio()
    return (similarity_ratio>=.65)

def get_logo_images(urls, driver=DRIVER, sleep_time=SLEEP):
    url_list = []
    url_names = []
    cnt = 0
    for url in urls:
        try:
            driver.get("https://www." + url)
            time.sleep(sleep_time)
            cnt += 1
            print(cnt)
            icon_element = driver.find_element(By.XPATH, "//link[contains(@rel, 'icon')]")
            icon_url = icon_element.get_attribute("href")
            url_names.append(url)
            url_list.append(icon_url)

        except Exception as e:
            pass
        except ConnectionError as e:
            pass

    driver.quit()
    return (url_list,url_names)


def download_img(folders):
    for folder in folders:
        folder_path = f'{DATA_PATH}/{folder}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(folder_path)
        for i, img_link in enumerate(folders[folder]):
            img_name = f'logo_{i}.png'
            img_path = os.path.join(folder_path, img_name)

            try:
                response = requests.get(img_link)

                if response.status_code == 200:
                    with open(img_path, 'wb') as file:
                        file.write(response.content)
                    try:
                        img = Image.open(img_path)
                        img.verify()
                    except (IOError, SyntaxError) as e:
                        print(f"Error with image {img_name}: {e}")
                        os.remove(img_path)
                else:
                    print(f"Download failed at {img_link}")
            except requests.exceptions.RequestException as e:
                print(f"Download failed at {img_link}: {e}")
        if len(os.listdir(folder_path)) == 0:
           shutil.rmtree(folder_path)


df = pd.read_parquet('./datasets/logos.snappy.parquet')
websites_domain = df["domain"].to_numpy()
img_links,urls = get_logo_images(websites_domain[:3400])

folders = defaultdict(list)
for i in range(len(urls)):
    folders[urls[i]].append(img_links[i])
    for j in range(len(urls)):
        if i != j:
            if is_similar(urls[i], urls[j]):
                folders[urls[i]].append(img_links[j])

folders = dict(folders)

download_img(folders)




