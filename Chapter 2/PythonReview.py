# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:47:09 2019

@author: nguyentrang
"""

#%%
import pandas as pd
new_list = ['A','B','C','D']
pd.Series(new_list)
#%%

new_dict = {"first letter": "A", 
            "second letter": "B",
            "third letter": "C"}

pd.Series(new_dict)

#%%