# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:20:10 2017

@author: patemi
"""
import os.path
import math
import pandas as pd
import numpy as np
#import pandas as pd

col_ptfolio='PositionID'
col_delta='Delta'
col_vega='Vega'
col_wdelta='WDelta'
col_wvega='WVega'
col_bucket='Bucket'
col_kb_med='Kb_med'
col_kb_high='Kb_high'
col_kb_low='Kb_low'
col_risk_type='RiskType'
col_fact_type='FactorTypeID'
col_factorccy1='FactorCcy1'
col_risk_weight='RW'
col_curvature_risk_weight='CurvatureRW'
col_curvature_up='CurvaturePointUp'
col_curvature_dn='CurvaturePointDown'
col_issuer_id='IssuerID'
col_s_b='S_b'
col_risk_class='RiskClass'
col_factorgrid1='FactorGrid1'
col_sector='FRTB sector'
col_rating='CSRQuality'
col_lh='LH'
col_cvr='CVR'
col_cvr_k_down='CVR_k_dn'
col_cvr_k_up='CVR_k_up'
col_cvr='CVR'
col_tranche='Tranche'


col_CVR_RC_med='CurvatureRC_Med'
col_CVR_RC_high='CurvatureRC_High'
col_CVR_RC_low='CurvatureRC_Low'

col_Delta_RC_med='DeltaRC_Med'
col_Delta_RC_high='DeltaRC_High'
col_Delta_RC_low='DeltaRC_Low'

rw_sigma=0.55
rw_fx_delta=0.15
rw_fx_vega=1
rw_girr_vega=1
rw_csr_nonsec_vega=1.2


alpha=0.01

high_corr_factor=1.25
low_corr_factor=0.75


corr_fx_ybc=0.6

path=''


df_corr_csrnonsec_sectors=pd.read_excel(path+'Cross Bucket Correlations.xlsx', sheet_name='CSRNonSecSectors')


#df_corr_csrnonsec_sectors=pd.read_csv(path+'Corr_CSRNonSec_Sectors.csv')




def getcorr(corr, high_low_med):
    """ Given input correlation corr and scenario, return the corresponding correlation      
    """
    if high_low_med=='h':
        return high_corr(corr)
    elif high_low_med=='l':
        return low_corr(corr)
    elif high_low_med=='m':
        return corr

#getcorr.__doc__="Given input correlation corr and scenario (H/M/L), return the "

def high_corr(corr):
    return min(1.25*corr,1)
    
def low_corr(corr):
    return max(2*corr-1,0.75*corr)


def get_stress_corr_matrix(corr_matrix,high_low_med):
    """ Given the input correlation matrix and scenario (h/l/m), return corresponding correlation matrix
    """
    n=len(corr_matrix)
    
    I=np.identity(n)
    
    if high_low_med=='h':
        m=np.clip(1.25*corr_matrix,a_min=0,a_max=1)
        return m
    elif high_low_med=='l':
        
        #2*corr_matrix-I
        
        m=np.minimum(np.clip(2*corr_matrix-I,a_min=0, a_max=1),np.clip(0.75*corr_matrix+0.25*I,a_min=0, a_max=1))
        
        return m
    elif high_low_med=='m':
        m=corr_matrix

        return m



inter_bucket_corr=0.15



def calc_weighted_sensi(sensi,weights):
    return sensi*weights
    

def corr_matrix_equity_delta_ybc(buckets, corr_high_low_med):
    return corr_matrix_equity_ybc(buckets, corr_high_low_med)

def corr_matrix_equity_vega_ybc(buckets, corr_high_low_med):
    return corr_matrix_equity_ybc(buckets, corr_high_low_med)

def corr_matrix_equity_ybc(buckets, corr_high_low_med):
    
    buckets_index=[(bucket-1) for bucket in buckets]
    buckets_index = [int(x) for x in buckets_index]    
    
    y_bc=np.array([[1.00, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.00, 0.45, 0.45, 0.45, 0.45],
                          [0.15, 1.00, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.00, 0.45, 0.45, 0.45, 0.45],
                          [0.15, 0.15, 1.00, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.00, 0.45, 0.45, 0.45, 0.45],
                          [0.15, 0.15, 0.15, 1.00, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.00, 0.45, 0.45, 0.45, 0.45],
                          [0.15, 0.15, 0.15, 0.15, 1.00, 0.15, 0.15, 0.15, 0.15, 0.15, 0.00, 0.45, 0.45, 0.45, 0.45],
                          [0.15, 0.15, 0.15, 0.15, 0.15, 1.00, 0.15, 0.15, 0.15, 0.15, 0.00, 0.45, 0.45, 0.45, 0.45],
                          [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 1.00, 0.15, 0.15, 0.15, 0.00, 0.45, 0.45, 0.45, 0.45],
                          [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 1.00, 0.15, 0.15,0.00, 0.45, 0.45, 0.45, 0.45],
                          [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 1.00, 0.15, 0.00, 0.45, 0.45, 0.45, 0.45],
                          [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 1.00, 0.00, 0.45, 0.45, 0.45, 0.45],
                          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
                          [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.00, 1.00, 0.75, 0.45, 0.45],
                          [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.00, 0.75, 1.00, 0.45, 0.45],
                          [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.00, 0.45, 0.45, 1.00, 0.45],
                          [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.00, 0.45, 0.45, 0.45, 1.00],
                          ])
    
    y_bc=y_bc[:,buckets_index]
    y_bc=y_bc[buckets_index,:]
    
    
    y_bc=get_stress_corr_matrix(y_bc,corr_high_low_med)
    
    #print(y_bc)
    
    return y_bc
    #m()
    


def corr_matrix_csrnonsec_ratings_ybc(buckets, corr_high_low_med):
    
    buckets_index=[(bucket-1) for bucket in buckets]
    buckets_index = [int(x) for x in buckets_index]    
    
    y_bc=np.array([[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 0.50, 0.50, 1.00, 1.00, 1.00],
                  [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 0.50, 1.00, 1.00, 1.00],
                  [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 1.00, 1.00, 1.00, 1.00],
                  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
                  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
                  [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
                  ])
    
    y_bc=y_bc[:,buckets_index]
    y_bc=y_bc[buckets_index,:]
    
    y_bc=get_stress_corr_matrix(y_bc,corr_high_low_med)
    
    return y_bc


def corr_matrix_csrnonsec_sectors_ybc(buckets, corr_high_low_med):
    
    buckets_index=[(bucket-1) for bucket in buckets]
    buckets_index = [int(x) for x in buckets_index]    

    y_bc=np.asarray(df_corr_csrnonsec_sectors)
    
    y_bc=y_bc[:,buckets_index]
    y_bc=y_bc[buckets_index,:]
    
    y_bc=get_stress_corr_matrix(y_bc,corr_high_low_med)
    
    return y_bc



def delta_corr_matrix(fact_type,issuer,corr_spt_spt) :
    # Inputs:
    # fact_type: array of factor types
    # issuer: array of issuer IDs
    # bucket: array of buckets
    
    # Output: numpy array , m
    
    n=len(fact_type)
    m = np.zeros((n,n))
    
    # Check if it's equity spot
    is_spt=np.zeros((n,1))
    is_spt=(fact_type=='EQEQPVOL00').astype(int).values.reshape(n,1)    
    is_spt_t=is_spt.transpose()
    
    spt_spt=np.matmul(is_spt,is_spt_t)
    spt_spt=spt_spt*corr_spt_spt

    m=spt_spt
        
    for i in range(n):
        for j in range(n):
            if (issuer.iloc[i]==issuer.iloc[j]):
                m[i,j]=1
    return m        


## TO BE DELETED #################
def maturity_corr_matrix(maturities):
    n=len(maturities)
    m = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            m[i,j]=math.exp(-alpha*abs(maturities.iloc[i]-maturities.iloc[j])/min(maturities.iloc[i],maturities.iloc[j]))
    return m
    

def corr_matrix_maturity(maturities,corr_high_low_med):
    """ Correlation along option/underlying maturity
    """
    n=len(maturities)
    m = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            
            m[i,j]=getcorr(math.exp(-alpha*abs(maturities[i]-maturities[j])/min(maturities[i],maturities[j])),corr_high_low_med)
            
    return m
    

def corr_matrix_vega(delta_corr_matrix,maturity_corr_matrix):
    # Inputs: delta and option maturity correlation matrices
    
    m=delta_corr_matrix*maturity_corr_matrix
    
    n=len(delta_corr_matrix)
    I=np.ones((n,n))
    z=np.subtract(m,I)
    
    y=z>0
    
    m[y==True]=1
    
    return m



## TO BE DELETED #################
def vega_corr_matrix(delta_corr_matrix,maturity_corr_matrix):
    n=len(delta_corr_matrix)
    
    I=np.ones((n,n))
    m=delta_corr_matrix*maturity_corr_matrix
    
    z=np.subtract(m,I)
    
    y=z>0
    
    m[y==True]=1
    
    return m
    
    
def corr_matrix_bucket(corr, n):
    """ Function to return the correlation matrix for aggregation within particular bucket b
    # given the correlation value for the bucket and the number of items in the bucket (n)
    # Return : n by n matrix with all correlations set to corr    
    
    """
    M1=np.full([n,n],corr)
    # Identity matrix
    I1=np.matlib.identity(n)
    M2=np.subtract(M1,corr*I1)    
    return np.add(I1,M2)



def corr_matrix_name(issuers,corr_high_low_med):
    
    n=len(issuers)
    m = np.zeros((n,n))
    
    for i in range(0,n):
        for j in range(0,n):
            #if (issuers.iloc[i]==issuers.iloc[j])=True:
            #if issuers.iloc[i].equals(issuers.iloc[j]):
            if str(issuers[i])==str(issuers[j]):    
                m[i,j]=1
            else:
                m[i,j]=getcorr(0.35,corr_high_low_med)
    return m


def equity_delta_corr_matrix_bucket(fact_type,issuer,corr_spt_spt, corr_repo_repo, corr_spot_repo_same_issuer,corr_spot_repo_diff_issuer) :
    # Function to return the correlation matrix to apply for the bucket level risk position calculation
    
    # Inputs
    # fact_type: array of factor types
    # issuer: array of issuer IDs
    # correlation values  (spot-spot, repo-repo, spot-repo same issuer, spot -repo different issuer) as arrays
    
    n=len(fact_type)
    # Create the matrix with dimension
    m = np.zeros((n,n))
    
    is_spt=np.zeros((n,1))
    is_spt=(fact_type=='EQEQPSPT00').astype(int).values.reshape(n,1)    
    is_spt_t=is_spt.transpose()
    
    # matrix of booleans for items which are spot-spot
    spt_spt=np.matmul(is_spt,is_spt_t)
        
    is_repo=np.zeros((n,1))
    is_repo=(fact_type=='EQIBRCUR00').astype(int).values.reshape(n,1)
    is_repo_t=is_repo.transpose()
    
    # matrix of booleans for items which are repo-repo
    repo_repo=np.matmul(is_repo,is_repo_t)
    
    # multiply boolean matrix by input correlations to get matrix of spot-spot and repo-repo correlations
    spt_spt=spt_spt*corr_spt_spt
    repo_repo=repo_repo*corr_repo_repo
        
    # combine the above two matrices
    m=np.add(spt_spt,repo_repo)
    

    
    for i in range(n):
        for j in range(n):
            if (i==j):
                # diagonal elements are 1
                m[i,j]=1
            elif m[i,j]==0:
                # remaining items must be spot-repo of same issuer
                ## what about spot-repo of different issuer????
                m[i,j]=corr_spot_repo_same_issuer
                continue                    
    return m        


    

    


def corr_matrix_tranche(tranches):
    # Tranche Correlation matrix , used for CSR Sec
    n=len(tranches)
    m = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            if tranches.iloc[i].equals(tranches.iloc[j]):
                m[i,j]=1
            else:
                m[i,j]=0.40
    return m




def corr_matrix_tenor(tenors,corr_high_low_med):
    # Tenor Correlation matrix , used for CSR NonSec
    n=len(tenors)
    m = np.zeros((n,n))
    
    for i in range(0,n):
        for j in range(0,n):
            if tenors[i]==tenors[j]:
                m[i,j]=1
            else:
                m[i,j]=getcorr(0.65,corr_high_low_med)
    return m

    
    
def corr_matrix_tenor_1(tenors):
    # Tenor Correlation matrix , used for CSR NonSec
    n=len(tenors)
    m = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            if tenors.iloc[i].equals(tenors.iloc[j]):
                m[i,j]=1
            else:
                m[i,j]=0.65
    return m

    
def corr_matrix_sec_tenor(tenors):
    # Tenor Correlation matrix , used for CSR Sec 
    n=len(tenors)
    m = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            if tenors.iloc[i].equals(tenors.iloc[j]):
                m[i,j]=1
            else:
                m[i,j]=0.80
    return m


    
    
def corr_matrix_basis(curves,corr_high_low_med):
    # Basis (CDS-Bonds) Correlation matrix , used for CSR NonSec 
    n=len(curves)
    m = np.zeros((n,n))
    
    for i in range(0,n):
        for j in range(0,n):
           # if curves.iloc[i].equals(curves.iloc[j]):
            if str(curves[i])==str(curves[j]):
                m[i,j]=1
            else:
                m[i,j]=sa.getcorr(0.999,corr_high_low_med)
    return m

    
def corr_matrix_sector_1(buckets,sectors):
    # Inuputs:  buckets: array of buckets
    #           sectors: array of sectors , corresponding to buckets         
    
    n=len(sectors)
    m = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            if sectors.iloc[i].equals(sectors.iloc[j]):
                m[i,j]=1
            elif (buckets.iloc[i].real.astype(int) in (1,9) and buckets.iloc[j].real.astype(int) in (2,10)) or (buckets.iloc[j].real.astype(int) in (1,9) and buckets.iloc[i].real.astype(int) in (2,10)):
                m[i,j]=0.75
            elif (buckets.iloc[i].real.astype(int) in (1,9) and buckets.iloc[j].real.astype(int) in (3,11)) or (buckets.iloc[j].real.astype(int) in (1,9) and buckets.iloc[i].real.astype(int) in (3,11)):
                m[i,j]=0.10
            elif (buckets.iloc[i].real.astype(int) in (1,9) and buckets.iloc[j].real.astype(int) in (4,12))or (buckets.iloc[j].real.astype(int) in (1,9) and buckets.iloc[i].real.astype(int) in (4,12)):
                m[i,j]=0.20
            elif (buckets.iloc[i].real.astype(int) in (1,9) and buckets.iloc[j].real.astype(int) in (5,13)) or (buckets.iloc[j].real.astype(int) in (1,9) and buckets.iloc[i].real.astype(int) in (5,13)):
                m[i,j]=0.25
            elif (buckets.iloc[i].real.astype(int) in (1,9) and buckets.iloc[j].real.astype(int) in (6,14)) or (buckets.iloc[j].real.astype(int) in (1,9) and buckets.iloc[i].real.astype(int) in (6,14)):
                m[i,j]=0.20
            elif (buckets.iloc[i].real.astype(int) in (1,9) and buckets.iloc[j].real.astype(int) in (7,15)) or (buckets.iloc[j].real.astype(int) in (1,9) and buckets.iloc[i].real.astype(int) in (7,15)):
                m[i,j]=0.15
            elif (buckets.iloc[i].real.astype(int) in (1,9) and buckets.iloc[j].real.astype(int) == 8) or (buckets.iloc[j].real.astype(int) in (1,9) and buckets.iloc[i].real.astype(int) == 8):
                m[i,j]=0.10
            elif (buckets.iloc[i].real.astype(int) in (2,10) and buckets.iloc[j].real.astype(int) in (3,11)) or (buckets.iloc[j].real.astype(int) in (2,10) and buckets.iloc[i].real.astype(int) in (3,11)):
                m[i,j]=0.05
            elif (buckets.iloc[i].real.astype(int) in (2,10) and buckets.iloc[j].real.astype(int) in (4,12)) or (buckets.iloc[j].real.astype(int) in (2,10) and buckets.iloc[i].real.astype(int) in (4,12)):
                m[i,j]=0.15
            elif (buckets.iloc[i].real.astype(int) in (2,10) and buckets.iloc[j].real.astype(int) in (5,13)) or (buckets.iloc[j].real.astype(int) in (2,10) and buckets.iloc[i].real.astype(int) in (5,13)):
                m[i,j]=0.20
            elif (buckets.iloc[i].real.astype(int) in (2,10) and buckets.iloc[j].real.astype(int) in (6,14)) or (buckets.iloc[j].real.astype(int) in (2,10) and buckets.iloc[i].real.astype(int) in (6,14)):
                m[i,j]=0.15
            elif (buckets.iloc[i].real.astype(int) in (2,10) and buckets.iloc[j].real.astype(int) in (7,15)) or (buckets.iloc[j].real.astype(int) in (2,10) and buckets.iloc[i].real.astype(int) in (7,15)):
                m[i,j]=0.10
            elif (buckets.iloc[i].real.astype(int) in (2,10) and buckets.iloc[j].real.astype(int) == 8) or (buckets.iloc[j].real.astype(int) in (2,10) and buckets.iloc[i].real.astype(int) == 8):
                m[i,j]=0.10
            elif (buckets.iloc[i].real.astype(int) in (3,11) and buckets.iloc[j].real.astype(int) in (4,12)) or (buckets.iloc[j].real.astype(int) in (3,11) and buckets.iloc[i].real.astype(int) in (4,12)):
                m[i,j]=0.05
            elif (buckets.iloc[i].real.astype(int) in (3,11) and buckets.iloc[j].real.astype(int) in (5,13)) or (buckets.iloc[j].real.astype(int) in (3,11) and buckets.iloc[i].real.astype(int) in (5,13)):
                m[i,j]=0.15
            elif (buckets.iloc[i].real.astype(int) in (3,11) and buckets.iloc[j].real.astype(int) in (6,14)) or (buckets.iloc[j].real.astype(int) in (3,11) and buckets.iloc[i].real.astype(int) in (6,14)):
                m[i,j]=0.20    
            elif (buckets.iloc[i].real.astype(int) in (3,11) and buckets.iloc[j].real.astype(int) in (7,15)) or (buckets.iloc[j].real.astype(int) in (3,11) and buckets.iloc[i].real.astype(int) in (7,15)):
                m[i,j]=0.05
            elif (buckets.iloc[i].real.astype(int) in (3,11) and buckets.iloc[j].real.astype(int) ==8) or (buckets.iloc[j].real.astype(int) in (3,11) and buckets.iloc[i].real.astype(int) ==8):
                m[i,j]=0.20
            elif (buckets.iloc[i].real.astype(int) in (4,12) and buckets.iloc[j].real.astype(int) in (5,13)) or (buckets.iloc[j].real.astype(int) in (4,12) and buckets.iloc[i].real.astype(int) in (5,13)):
                m[i,j]=0.20
            elif (buckets.iloc[i].real.astype(int) in (4,12) and buckets.iloc[j].real.astype(int) in (6,14)) or (buckets.iloc[j].real.astype(int) in (4,12) and buckets.iloc[i].real.astype(int) in (6,14)):
                m[i,j]=0.25    
            elif (buckets.iloc[i].real.astype(int) in (4,12) and buckets.iloc[j].real.astype(int) in (7,15)) or (buckets.iloc[j].real.astype(int) in (4,12) and buckets.iloc[i].real.astype(int) in (7,15)):
                m[i,j]=0.05
            elif (buckets.iloc[i].real.astype(int) in (4,12) and buckets.iloc[j].real.astype(int) ==8) or (buckets.iloc[j].real.astype(int) in (4,12) and buckets.iloc[i].real.astype(int) ==8):
                m[i,j]=0.05
            elif (buckets.iloc[i].real.astype(int) in (5,13) and buckets.iloc[j].real.astype(int) in (6,14)) or (buckets.iloc[j].real.astype(int) in (5,13) and buckets.iloc[i].real.astype(int) in (6,14)):
                m[i,j]=0.25    
            elif (buckets.iloc[i].real.astype(int) in (5,13) and buckets.iloc[j].real.astype(int) in (7,15)) or (buckets.iloc[j].real.astype(int) in (5,13) and buckets.iloc[i].real.astype(int) in (7,15)):
                m[i,j]=0.05
            elif (buckets.iloc[i].real.astype(int) in (5,13) and buckets.iloc[j].real.astype(int) ==8) or (buckets.iloc[j].real.astype(int) in (5,13) and buckets.iloc[i].real.astype(int) ==8):
                m[i,j]=0.15
            elif (buckets.iloc[i].real.astype(int) in (6,14) and buckets.iloc[j].real.astype(int) in (7,15)) or (buckets.iloc[j].real.astype(int) in (6,14) and buckets.iloc[i].real.astype(int) in (7,15)):
                m[i,j]=0.05
            elif (buckets.iloc[i].real.astype(int) in (6,14) and buckets.iloc[j].real.astype(int) ==8) or (buckets.iloc[j].real.astype(int) in (6,14) and buckets.iloc[i].real.astype(int) ==8):
                m[i,j]=0.20
            elif (buckets.iloc[i].real.astype(int) in (7,15) and buckets.iloc[j].real.astype(int) ==8) or (buckets.iloc[j].real.astype(int) in (7,15) and buckets.iloc[i].real.astype(int) ==8):
                m[i,j]=0.05
                
    return m



def corr_matrix_sector(buckets,sectors):
    # Inuputs:  buckets: array of buckets
    #           sectors: array of sectors , corresponding to buckets         
    
    n=len(sectors)
    m = np.zeros((n,n))
    
    for i in range(0,n):
        for j in range(0,n):
            if sectors[i]==sectors[j]:
                m[i,j]=1
            elif (buckets[i] in (1,9) and buckets[j] in (2,10)) or (buckets[j] in (1,9) and buckets[i] in (2,10)):
                m[i,j]=0.75
            elif (buckets[i] in (1,9) and buckets[j] in (3,11)) or (buckets[j] in (1,9) and buckets[i] in (3,11)):
                m[i,j]=0.10
            elif (buckets[i] in (1,9) and buckets[j] in (4,12))or (buckets[j] in (1,9) and buckets[i] in (4,12)):
                m[i,j]=0.20
            elif (buckets[i] in (1,9) and buckets[j] in (5,13)) or (buckets[j] in (1,9) and buckets[i] in (5,13)):
                m[i,j]=0.25
            elif (buckets[i] in (1,9) and buckets[j] in (6,14)) or (buckets[j] in (1,9) and buckets[i] in (6,14)):
                m[i,j]=0.20
            elif (buckets[i] in (1,9) and buckets[j] in (7,15)) or (buckets[j] in (1,9) and buckets.iloc[i] in (7,15)):
                m[i,j]=0.15
            elif (buckets[i] in (1,9) and buckets[j] == 8) or (buckets[j] in (1,9) and buckets[i] == 8):
                m[i,j]=0.10
            elif (buckets[i] in (2,10) and buckets[j] in (3,11)) or (buckets[j] in (2,10) and buckets[i] in (3,11)):
                m[i,j]=0.05
            elif (buckets[i] in (2,10) and buckets[j] in (4,12)) or (buckets[j] in (2,10) and buckets[i] in (4,12)):
                m[i,j]=0.15
            elif (buckets[i] in (2,10) and buckets[j] in (5,13)) or (buckets[j] in (2,10) and buckets[i] in (5,13)):
                m[i,j]=0.20
            elif (buckets[i] in (2,10) and buckets[j] in (6,14)) or (buckets[j] in (2,10) and buckets[i] in (6,14)):
                m[i,j]=0.15
            elif (buckets[i] in (2,10) and buckets[j] in (7,15)) or (buckets[j] in (2,10) and buckets[i] in (7,15)):
                m[i,j]=0.10
            elif (buckets[i] in (2,10) and buckets[j] == 8) or (buckets[j] in (2,10) and buckets[i] == 8):
                m[i,j]=0.10
            elif (buckets[i] in (3,11) and buckets[j] in (4,12)) or (buckets[j] in (3,11) and buckets[i] in (4,12)):
                m[i,j]=0.05
            elif (buckets[i] in (3,11) and buckets[j] in (5,13)) or (buckets[j] in (3,11) and buckets[i] in (5,13)):
                m[i,j]=0.15
            elif (buckets[i] in (3,11) and buckets[j] in (6,14)) or (buckets[j] in (3,11) and buckets[i] in (6,14)):
                m[i,j]=0.20    
            elif (buckets[i] in (3,11) and buckets[j] in (7,15)) or (buckets[j] in (3,11) and buckets[i] in (7,15)):
                m[i,j]=0.05
            elif (buckets[i] in (3,11) and buckets[j] ==8) or (buckets[j] in (3,11) and buckets[i] ==8):
                m[i,j]=0.20
            elif (buckets[i] in (4,12) and buckets[j] in (5,13)) or (buckets[j] in (4,12) and buckets[i] in (5,13)):
                m[i,j]=0.20
            elif (buckets[i] in (4,12) and buckets[j] in (6,14)) or (buckets[j] in (4,12) and buckets[i] in (6,14)):
                m[i,j]=0.25    
            elif (buckets[i] in (4,12) and buckets[j] in (7,15)) or (buckets[j] in (4,12) and buckets[i] in (7,15)):
                m[i,j]=0.05
            elif (buckets[i] in (4,12) and buckets[j] ==8) or (buckets[j] in (4,12) and buckets[i] ==8):
                m[i,j]=0.05
            elif (buckets[i] in (5,13) and buckets[j] in (6,14)) or (buckets[j] in (5,13) and buckets[i] in (6,14)):
                m[i,j]=0.25    
            elif (buckets[i] in (5,13) and buckets[j] in (7,15)) or (buckets[j] in (5,13) and buckets[i] in (7,15)):
                m[i,j]=0.05
            elif (buckets[i] in (5,13) and buckets[j] ==8) or (buckets[j] in (5,13) and buckets[i] ==8):
                m[i,j]=0.15
            elif (buckets[i] in (6,14) and buckets[j] in (7,15)) or (buckets[j] in (6,14) and buckets[i] in (7,15)):
                m[i,j]=0.05
            elif (buckets[i] in (6,14) and buckets[j] ==8) or (buckets[j] in (6,14) and buckets[i] ==8):
                m[i,j]=0.20
            elif (buckets[i] in (7,15) and buckets[j] ==8) or (buckets[j] in (7,15) and buckets[i] ==8):
                m[i,j]=0.05
    
    
    #m=
        
    return m
    

def corr_matrix_rating_1(ratings):
    # Input: array of bucket level ratings
     # Outputs : array (matrix) of correlations for bucket aggregation
    
    n=len(ratings)
    m = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            if ratings.iloc[i].equals(ratings.iloc[j]):
                m[i,j]=1
            else:
                m[i,j]=0.50
    return m



def corr_matrix_rating(ratings):
    # Input: array of bucket level ratings
     # Outputs : array (matrix) of correlations for bucket aggregation
    
    n=len(ratings)
    m = np.zeros((n,n))
    
    for i in range(0,n):
        for j in range(0,n):
            if str(ratings[i])==str(ratings[j]):
                m[i,j]=1
            else:
                m[i,j]=0.50
    return m
    

def corr_matrix_girr(factor_structure, factor_category, tenors, corr_high_low_med):
    #Inputs: (df) factor structure, factor categoy, tenors
    
    # Output: numpy array of correlations
    
    n=len(factor_structure)
    m = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            #same tenors, different curves
            if tenors.iloc[i]==tenors.iloc[j] and factor_structure.iloc[i]!=factor_structure.iloc[j]:                
                m[i,j]=0.999
            #different tenors, same curve
            elif tenors.iloc[i]!=tenors.iloc[j] and factor_structure.iloc[i]==factor_structure.iloc[j]:
                m[i,j]=max(0.4,math.exp(-0.03*abs(tenors.iloc[i]-tenors.iloc[j])/min(tenors.iloc[i],tenors.iloc[j])))
            elif (tenors.iloc[i]!=tenors.iloc[j] and factor_structure.iloc[i]!=factor_structure.iloc[j]) or (tenors.iloc[j]!=tenors.iloc[i] and factor_structure.iloc[j]!=factor_structure.iloc[i]):
                m[i,j]=0.999*max(0.4,math.exp(-0.03*abs(tenors.iloc[i]-tenors.iloc[j])/min(tenors.iloc[i],tenors.iloc[j])))
            # Currency Basis
            elif factor_category.iloc[i]=='Currency_Basis' or factor_structure.iloc[j]=='Currency_Basis' :
                m[i,j]=0
            #Inflation and Not Currency Basis
            elif (factor_category.iloc[i]=='Currency_Basis' and factor_category.iloc[j]!='Inflation') or (factor_category.iloc[j]=='Inflation' and factor_category.iloc[i]!='Currency_Basis'):
                m[i,j]=0.4                   
            else:
                m[i,j]=1
    
    m=get_stress_corr_matrix(m,corr_high_low_med)
    
    return m
    

def corr_matrix_girr_curvature(factor_structure, factor_category, tenors):
    m = corr_matrix_girr(factor_structure, factor_category, tenors)**2
    return m
    

    
#def K_b_csrnonsec_curvature_other():


def S_b_curvature(cvr_up,cvr_down, k_b, k_b_up, k_b_down):
    """ Computes S_b for a particular bucket given 
     Input:    ws: array of weighted sensi for factors in the bucket
               k_b: bucket risk position
               alternative: flag to indicate if alternative approach should be used for the calculation
     Output: S_b for individual bucket b
     """ 
    if k_b==k_b_up:
        S_b_cvr=sum(cvr_up)
    elif k_b==k_b_down:
        S_b_cvr=sum(cvr_down)
    return S_b_cvr 
    


def S_b_delta(ws, k_b, alternative):
    # Computes S_b for a particular bucket given 
    # Input:    ws: array of weighted sensi for factors in the bucket
    #           k_b: array of bucket risk position
    #           alternative: boolean flag to indicate if alternative approach should be used for the calculation
    # Output: S_b for individual bucket b
    #print(ws,k_b, alternative)
    #print()
    
    
    if alternative==True: 
        return max(min(sum(ws),k_b),-k_b)
    else: 
        if isinstance(ws,float):
            return ws
        else:    
            return sum(ws)
    







def CVR_up(V_up_V,rw,delta):
    return V_up_V-rw*delta

def CVR_down(V_down_V,rw,delta):
    return V_down_V+rw*delta



def K_b_csrnonsec_delta(ws,corr):
    return K_b_delta_vega(ws,corr)

def K_b_csrnonsec_delta_other(ws):
     """Input: weighted sensis of the other bucket (as an array)"""
     return K_b_delta_vega_other(ws)

def K_b_csrnonsec_curvature_up(cvr_up,corr):
    return K_b_curvature_up(cvr_up,corr)
    
def K_b_csrnonsec_curvature_down(cvr_down,corr):
    return K_b_curvature_down(cvr_down,corr)

def K_b_csrnonsec_curvature_other(cvr_down,corr):
    return K_b_curvature_down(cvr_down,corr)


def K_b_equity_delta(ws,corr):
    return K_b_delta_vega(ws,corr)

def K_b_equity_vega(ws,corr):
    return K_b_delta_vega(ws,corr)

def K_b_equity_vega_other(ws):
     """Input: weighted sensis of the other bucket (as an array)"""
     return K_b_delta_vega_other(ws)

def K_b_equity_delta_other(ws):
     """Input: weighted sensis of the other bucket (as an array)"""
     return K_b_delta_vega_other(ws)
    
def K_b_equity_curvature_other(cvr_up, cvr_down):
    """Input: curvature sensis of the other bucket (as an array)"""
    floor_up=np.zeros(len(cvr_up))
    floor_down=np.zeros(len(cvr_down))
    
    max_cvr_up=np.maximum(cvr_up,floor_up)
    max_cvr_down=np.maximum(cvr_down,floor_down)
    
    return max(sum(max_cvr_up), sum(max_cvr_down))

def K_b_equity_curvature_up(cvr_up,corr):
    return K_b_curvature_up(cvr_up,corr)

def K_b_equity_curvature_down(cvr_down,corr):
    return K_b_curvature_down(cvr_down,corr)



def K_b_girr_delta(ws,corr):
    return K_b_delta_vega(ws,corr)

def K_b_girr_vega(ws,corr):
    return K_b_delta_vega(ws,corr)

def K_b_girr_curvature_up(cvr_up,corr):
    return K_b_curvature_up(cvr_up,corr)
    
def K_b_girr_curvature_down(cvr_down,corr):
    return K_b_curvature_down(cvr_down,corr)



def K_b_delta_vega(ws,corr):
    """     Delta/Vega Risk Position (K_b) for bucket b
    ==============================================================================
         Inputs
           ws: Vector of weighted sensitivities
           corr: Correlation matrix for bucket level aggregation 
    
    ==============================================================================
    """
    return math.sqrt(max(0,np.matmul(ws.transpose(),np.matmul(corr,ws))))
    
def K_b_delta_vega_other(ws):
    return sum(np.absolute(ws))


def K_b_curvature_up(cvr_up,corr):
    """Given CVR+ as input, and the relevant correlations
         Inputs:      cvr: (array) curvature risk exposures for each factor in bucket b
                         corr: (array) correlation matrix to be applied to curvature risk exposures to aggregate for bucket risk position
    """
    k_b_up=K_b_curvature(cvr_up,corr)
    return k_b_up

def K_b_curvature_down(cvr_down,corr):
    """Given CVR+ as input, and the relevant correlations
         Inputs:      cvr: (array) curvature risk exposures for each factor in bucket b
                         corr: (array) correlation matrix to be applied to curvature risk exposures to aggregate for bucket risk position
    """
    k_b_down=K_b_curvature(cvr_down,corr)
    return k_b_down


    
def K_b_curvature(cvr,corr):
     
    """Curvature Risk Position (K_b) for bucket b
    
    # Inputs: cvr: (array) curvature risk exposures for bucket b
    #       corr: (array) correlation matrix to be applied to curvature risk exposures to aggregate for bucket risk position
   """
    n=len(corr)
    psi=np.zeros((n,n))
    
    for i in range(0,n):
         for j in range(0,n):
             if i==j:
                 corr[i,j]=0
                        
    # Construct psi matrix
    for i in range(0,n):
        for j in range(0,n):
            if (cvr[i]<0 and cvr[j]<0):
                psi[i,j]=0
            else: 
                psi[i,j]=1
                
    corr_psi=corr*psi
    
    cvr[cvr<0]=0
    max_cvr=cvr
      
    max_cvr_2=max_cvr**2
    
    sum_=np.vdot(np.array(max_cvr_2),np.ones((n,1)))
    
    m=math.sqrt(max(0,sum_+ np.matmul(cvr.transpose(),np.matmul(corr_psi,cvr))))
           
    return m



    
#def RC2(K,corr_bc,S):
#    # TO BE DELETED IF NOT BEING USED!!!
#    I1=np.matlib.identity(len(S))
#    sum_K_b2= np.matmul(K.astype(float).transpose(),np.matmul(I1,K.astype(float)))
#    print(sum_K_b2)
#    #I2=np.matlib.identity(len(corr_bc))    
#    sum_sum=np.matmul(S.astype(float).transpose(),np.matmul(corr_bc,S.astype(float)))
#    print(sum_sum)
#   
#    return math.sqrt(sum_K_b2+sum_sum)

    

def equity_vega_risk_charge(K,corr_bc,WS):     
    return delta_vega_risk_charge(K,corr_bc,WS)

def girr_vega_risk_charge(K,corr_bc,WS):     
    return delta_vega_risk_charge(K,corr_bc,WS)


def fx_delta_risk_charge(K,corr_bc,WS):     
    return delta_vega_risk_charge(K,corr_bc,WS)
    
def equity_delta_risk_charge(K,corr_bc,WS):     
    return delta_vega_risk_charge(K,corr_bc,WS)

def girr_delta_risk_charge(K,corr_bc,WS):     
    return delta_vega_risk_charge(K,corr_bc,WS)



def delta_vega_risk_charge(K,corr_bc,WS):
    # Aggregate Risk Charge across buckets
    # given K: bucket level Risk positions 
    #       corr_bc: correlation matrix between buckets
    #       WS: weighted sensitivities for each bucket in the risk class
    # Need to incorporate the case where the sqrt is negative

    #print(K)
    #print(corr_bc)
    #print(WS)
    
    sum_K_b2= np.dot(K,K)
    I=np.matlib.identity(len(corr_bc))    
    matrix_corr_bc=np.asmatrix(corr_bc)
    
    # By default, computs S_b for each bucket (set of weighted sensitivities) 
    #S=[S_b(ws,K,False) for ws in WS]
    
    S=[S_b_delta(ws,k,False) for ws,k in zip(WS,K)]
    #S=[S_b(np.array([ws]),np.array([k]),False) for ws,k in zip(WS,K)]
    
    
    matrix_S=np.asmatrix(S)
    
    M=np.matmul(matrix_S,np.subtract(matrix_corr_bc,I))
    sum_sum=np.matmul(matrix_S,M.transpose())
    
    if (sum_K_b2+sum_sum)<0:
    # if we have a negative root, recompute S_b using alternative approach
    # and the double sum
        #S=S_b(WS,K,True)
        #S=[S_b(ws,K,True) for ws in WS]
        S=[S_b(ws,k,True) for ws,k in zip(WS,K)]
        matrix_S=np.asmatrix(S)
        M=np.matmul(matrix_S,np.subtract(matrix_corr_bc,I))
        sum_sum=np.matmul(matrix_S,M.transpose())

    return math.sqrt(sum_K_b2+sum_sum)


    
def curvature_risk_charge(K_b,corr_bc,S_b):
    # Aggregate Risk Charge across buckets
    # given - bucket level Risk positions K
    #       - correlation matrix corr_bc
    #       - curvature sensitivities cvr
    # Need to incorporate the case where the sqrt is negative
    """
    Input: array of bucket risk positions K, correlation matrix for aggregation across buckets corr_bc, array of S_b
    """
    n=len(K_b)
    
    # Initialise psi matrix
    psi=np.zeros((n,n))
    sum_K_b2= np.dot(K_b,K_b)
    matrix_corr_bc=np.asmatrix(corr_bc)

    #S=[S_b(cvr,k,False) for cvr,k in zip(CVR,K)]    
    
    matrix_S=np.asmatrix(S_b)
    
    for i in range(n):
        for j in range(n):        
            if (float(S_b[i])<0 and float(S_b[j])<0):
                psi[i,j]=0
            else: 
                psi[i,j]=1
    
    matrix_psi=np.asmatrix(psi)
    matrix_corr_psi=np.matmul(matrix_corr_bc, matrix_psi)
    
    I=np.matlib.identity(n)    
    
    M=np.matmul(matrix_S,np.subtract(matrix_corr_psi,I))
    
    sum_sum=np.matmul(matrix_S,M.transpose())

    return math.sqrt(max(0,sum_K_b2+sum_sum))
    



## !!!!!!!!!!!BELOW FUNCTION TO BE REMOVED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
def RC_curvature(K,corr_bc,S):
    # Aggregate Risk Charge across buckets
    # given - bucket level Risk positions K
    #       - correlation matrix corr_bc
    #       - sum of weighted sensitivities in each bucket S
    # Need to incorporate the case where the sqrt is negative
    
    n=len(S)
 
    psi=np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            #print(i)
            #print(j)
            
            #if (S.iloc[i].real<0 and S.iloc[j].real<0):
            if (float(S.iloc[i])<0 and float(S.iloc[j])<0):
                psi[i,j]=0
            else: 
                psi[i,j]=1
    
    corr_psi=corr_bc * psi
       
    I1=np.matlib.identity(len(S))
    sum_K_b2= np.matmul(K.astype(float).transpose(),np.matmul(I1,K.astype(float)))
   
    I2=np.matlib.identity(len(corr_bc))    
    sum_sum=np.matmul(S.astype(float).transpose(),np.matmul(np.subtract(corr_psi,I2),S.astype(float)))

    return math.sqrt(sum_K_b2+sum_sum)
    
    
def NetJTD(seniorities, WGJTD_long, WGJTD_short):
    
    n=len(seniorities)
    
    seniority_ids=np.zeros(n)
    
    for i in range(1,len(seniorities)+1):    
        if seniorities[i-1]=='Senior':
            seniority_ids[i-1]=1
        else:
            seniority_ids[i-1]=2
        
    
    
            
            
    return seniority_ids
        
        
        
    