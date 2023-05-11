!pip install mlxtend


#implementataion of aprori algorithm
# Importing the libraries
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt



dataset = pd.read_csv('retail_dataset.csv')
dataset.describe()


#Print Total number of Rows & columns in dataset
print(dataset.shape[1])


#Print Information about data
dataset.info()


#Count total number of classes in Data
items = (dataset['0'].unique())
items


dataset.isnull().sum()


#Create list 
transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 7)])

transactions



import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
te_ary

df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)


df=df[['Bagel','Bread','Cheese','Diaper','Eggs','Meat','Milk','Pencil','Wine']]


#might have error so leave it 
df.drop(['nan'], axis = 1)


rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
rules

freq_items = apriori(df, min_support=0.2, use_colnames=True)
freq_items


freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))
freq_items

req_items[ (freq_items['length'] == 2) &
                   (freq_items['support'] >= 0.3) ]


plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()



plt.scatter(rules['support'], rules['lift'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs lift')
plt.show()


fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
 fit_fn(rules['lift'])

import pickle 
print("[INFO] Saving model...")
# Save the trained model as a pickle string. 
saved_model=pickle.dump(freq_items,open('aprioriexample.pkl', 'wb')) 
# Saving model to disk
         

# Load the pickled model 
model = pickle.load(open('aprioriexample.pkl','rb'))  
# Use the loaded pickled model to make predictions

         
%%writefile app.py
import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
st.set_option('deprecation.showfileUploaderEncoding', False)
def find_associatio_rule(support):
  # Load the pickled model
  model = pickle.load(open('aprioriexample.pkl','rb'))
    if uploaded_file is not None:
        dataset= pd.read_csv(uploaded_file)
    else:
        dataset= pd.read_csv('retail_dataset.csv')
                        #Create list 
    transactions = []
for i in range(0, 315):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 7)])

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
df=df[['Bagel','Bread','Cheese','Diaper','Eggs','Meat','Milk','Pencil','Wine']]
freq_items = apriori(df, min_support=support, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
return rules
def find_frequent_items(support):
  # Load the pickled model
  model = pickle.load(open('aprioriexample.pkl','rb'))     
      if uploaded_file is not None:
    dataset= pd.read_csv(uploaded_file)
      else:
    dataset= pd.read_csv('retail_dataset.csv')
  #Create list 
  transactions = []
    for i in range(0, 315):
        transactions.append([str(dataset.values[i,j]) for j in range(0, 7)])

    from mlxtend.preprocessing import TransactionEncoder
     te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    df=df[['Bagel','Bread','Cheese','Diaper','Eggs','Meat','Milk','Pencil','Wine']]
    freq_items = apriori(df, min_support=support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
  
  return freq_items
         
         
         
         
  html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
st.header("Identification of items Purchased together ")
  
uploaded_file = st.file_uploader("Upload dataset", help='Please upload retail_dataset.csv otherwise leave  blank') 
support = st.number_input('Insert a minimum suppport to find association rule ',0,1)
if st.button("Association Rule"):
    rules=find_associatio_rule(support)
    st.success('Apriori has found Following rules {}'.format(rules))
if st.button("Frequent Items"):
    frequent_items=find_frequent_items(support)
st.success('Apriori has found Frequent itemsets {}'.format(frequent_items))      
if st.button("About"):
    st.subheader("Developed by Deepak Moud")
    st.subheader("Head , Department of Computer Engineering")
html_temp = """
<div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Machine learning Experiment No. 9</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)       
         
  
         
         
         
         
         

