#!/usr/bin/env python
# coding: utf-8

# In[78]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time as time
from collections import Counter
import pandas as pd

NAME_LIST = []
d = {}
w_list = []
time_list = []

page_c = int(input("페이지 수 : "))
for count in range(page_c):
    print(f"{count+1}페이지")
    link = f'-'
    driver = webdriver.Chrome()
    driver.get(link)
    
    
    iframe = driver.find_element(By.ID, "cafe_main")
    driver.switch_to.frame(iframe)
    notice = driver.find_elements(By.XPATH, '//*[@id="upperArticleList"]/table/tbody/tr')

    c_box = driver.find_element(By.XPATH, '//*[@id="main-area"]/div[4]/table/tbody')
    contents = c_box.find_elements(By.CLASS_NAME, "article")
    #times = c_box.find_elements(By.CLASS_NAME, "td_date")

    for i in contents:
        w_list.append(i.text)
    name_list = [name.split()[1] for name in w_list if '인증' in name]
    for n in name_list:
        NAME_LIST.append(n)
    #for t in times:
        #time_list.append(t.text)

    #auth_list = tuple(zip(name_list,time_list))
    #for n, t in auth_list.items():
        #if len(t) == 5:
            #auth_list[n] = time.strftime('%Y.%m.%d.')
#del l[:len(notice)]

#for s in l:
    #print(s)
    
driver.quit()

counter = Counter(NAME_LIST)
s_counter = dict(sorted(counter.items()))

name = [n for n in s_counter.keys()]
count = [c for c in s_counter.values()]
discount = [c*100 for c in s_counter.values()]

auth_l = pd.DataFrame({'name' : name, 'count' : count, 'discount' : discount})
auth_l.to_csv('학생 인증 현황.csv',index=False,encoding='utf-8-sig')
print('완료')

