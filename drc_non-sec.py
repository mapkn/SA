# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:59:13 2017

@author: patemi
"""
import time
import numpy as np
import pandas as pd
import math
import numpy.matlib
import frtbsa

# Risk Weights, to be applied to NetJTD 
#rw_AAA=0.005
#rw_AA=0.02
#rw_A=0.03
#rw_BBB=0.06
#rw_BB=0.15
#rw_B=0.30
#rw_CCC=0.50
#rw_U=0.15
#rw_D=1

start=time.time()


def calcGrossJTDLong(LGD, N, PL): 
    Loss = np.multiply(np.array(LGD),np.array(N))+np.array(PL)
    Floor = np.zeros(len(LGD)) 
    return np.maximum(Loss,Floor)

def calcGrossJTDShort(LGD, N, PL): 
    Loss = np.multiply(np.array(LGD),np.array(N))+np.array(PL)
    Floor = np.zeros(len(LGD)) 
    return np.minimum(Loss,Floor)

def rankSeniority(Seniority):
    #Applies an index to given Seniority
    Sen=np.array(Seniority)
    
    Rank = np.empty(len(Seniority),dtype=str)
    Rank[Sen=='Equity']=2
    Rank[Sen=='Senior']=1
    Rank[Sen=='Non-Senior']=2
    return Rank

def wgtNetJTD(NJTD,RW):
    # Input: netJTD and the risk weights
    
    NetJTD=np.array(NJTD)
    RWgt=np.array(RW)
    
    return NetJTD*RWgt
    
#def Wts_B(NJTDL, NJTDS, Bucket):
    # Inputs are net long/short JTD for issuers in the bucket
    
def netJTD(Sen, GJTDL, GJTDS):
    # Given Seniorities and Gross JTD Long / Short for the issuer, return Net JTD 
    
    # Convert inputs to numpy
    GJTDL=np.array(GJTDL)
    GJTDS=np.array(GJTDS)
    
    # Give a number to the input Seniorities
    rank=rankSeniority(Sen)
    #print(rank)
    #print(GJTDL[rank==1])
    #print(GJTDS[rank==1])
    
    #Senior
    NetS1=GJTDL[rank==1] + GJTDS[rank==1]
    
    #Equity/Non-Senior
    NetS2=GJTDL[rank==2] + GJTDS[rank==2]
    
    
    if NetS1 > 0 and NetS2 > 0:
        NJTDL=NetS1+NetS2
        NJTDS=0
    elif NetS1 < 0 and NetS2 < 0:
        NJTDS=NetS1+NetS2
        NJTDL=0
    elif NetS1 > 0 and NetS2 < 0:
        NJTDL=max(NetS1+NetS2,0)
        NJTDS=min(NetS1+NetS2,0)
    elif NetS1 < 0 and NetS2 > 0:
        NJTDL=NetS2
        NJTDS=NetS1     
    #NJTDL=max(GJTDL[rank==1]+GJTDS[rank==1],0)+max(GJTDL[rank==2]+GJTDS[rank==2],0)+max(GJTDL[rank==2]+GJTDS[rank==1],0)
    #NJTDS=min(GJTDS[rank==1]+GJTDL[rank==1],0)+min(GJTDS[rank==2]+GJTDL[rank==2],0)+min(GJTDS[rank==2]+GJTDL[rank==1],0)
        
    return [NJTDL,NJTDS]
    
    
df_input=pd.read_csv('DRC_NonSec_Input.csv', encoding='latin-1')
df_input.drop([col for col in df_input.columns if "Unnamed" in col], axis=1, inplace=True)


# Unique list of Buckets
# Need to add a static list here
buckets=['Corporates', 'Sovereigns', 'local governments/municipalities']
df_all_buckets=pd.DataFrame(buckets, columns=[frtbsa.col_bucket])
n_buckets=len(df_all_buckets)

# Unique list of Test IDs
df_test_ids=pd.DataFrame(df_input['DRC_ID'].unique(), columns=['DRC_ID'])
# Filter for a specific case
df_test_ids=df_test_ids[df_test_ids['DRC_ID']=='DRC_2']


df_test=df_input

df_test[['Market Value','Notional','LGD']]=df_test[['Market Value','Notional','LGD']].apply(pd.to_numeric)

df_test['PL']=df_test['Market Value'].subtract(df_test['Notional'])

df_test['Gross JTD long']=df_test['Notional']>0
df_test['Gross JTD long'][df_test['Gross JTD long']==True]=df_test['Notional'].multiply(df_test['LGD']).add(df_test['PL']).clip(lower=0)

df_test['Gross JTD short']=df_test['Notional']<0
df_test['Gross JTD short'][df_test['Gross JTD short']==True]=df_test['Notional'].multiply(df_test['LGD']).add(df_test['PL']).clip(upper=0)

df_test['Maturity Scaling']=df_test['Matu']
df_test['Maturity Scaling'][df_test['Matu']>1]=1
df_test['Maturity Scaling'][df_test['Matu']<=0.25]=0.25


df_test['Weighted Gross JTD long']=df_test['Gross JTD long'].multiply(df_test['Matu'])
df_test['Weighted Gross JTD short']=df_test['Gross JTD short'].multiply(df_test['Matu'])

# Outputs for all portfolios containing bucket risk positions
df_outputs=pd.DataFrame()    
# Outputs for all portfolios containing Risk Charges only
df_output_rc=pd.DataFrame()

# Inter bucket correlation matrix


for index, row in df_test_ids.iterrows():
    # Iterate through each test id (considered as portfolio)
    
    #Filter for a particular portfolio
    df_ptfolio=df_test[df_test['DRC_ID']==row['DRC_ID']]
    
    grouping=['ObligorID', 'ObligorCategory', 'Seniority.1', 'Notional','Market Value']
    measures=['Weighted Gross JTD long','Weighted Gross JTD short']
    
    # Aggregated by Issuer, Seniority
    df_ptfolio_agg=df_ptfolio.groupby(grouping, as_index=False)[measures].sum()
    
    # Get the issuers in the portfolio
    issuers=df_ptfolio_agg['ObligorID'].unique()
    
    NetJTDLS=pd.DataFrame()
    
    #For each issuer in the portfolio
    for x in np.nditer(issuers):
        
        ob_cat=df_ptfolio_agg['ObligorCategory'][df_ptfolio_agg['ObligorID']==x].iloc[0]
        
        # Get the seniorities/WGJTDs for the issuer
        s=df_ptfolio_agg['Seniority.1'][df_ptfolio_agg['ObligorID']==x]
        wgjtdl=df_ptfolio_agg['Weighted Gross JTD long'][df_ptfolio_agg['ObligorID']==x]
        wgjtds=df_ptfolio_agg['Weighted Gross JTD short'][df_ptfolio_agg['ObligorID']==x]
        
        # Get the total Net JTD L/S for the issuer (taking care of seniority netting)
        njtd_ls=netJTD(s,wgjtdl, wgjtds)
    
        a=np.array([[x,njtd_ls[0],njtd_ls[1]]])
        
        data_add=pd.DataFrame(a,columns=['ObligorID','NetJTDL','NetJTDS'])
        data_add.set_index(data_add.ObligorID, drop=True,inplace=True)
        print(a)
        
        NetJTDLS=NetJTDLS.append(data_add)
        
        #df3=pd.merge(NetJTDLS,df_ptfolio_agg, on='ObligorCategory')
        
        #df_ptfolio_agg['NetJTDL','NetJTDS']    
        
    
    
    
    