# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:59:13 2017

@author: patemi
"""
#import time
import numpy as np
import pandas as pd
from scipy import interpolate
import math
import frtbsa_calc_d457_wip as sa
from functools import reduce 
import os.path

#import profile



#path = '//apw-grskfs01/GVAR2/Global Risk Management/FRTB/Sample Data/sample portfolio for SA and IMA/SR_samples_by_ProductType v3.85.xlsb.Export_181218_15_01/'
path = ''
alpha=0.01


##########################################################################################
# Read the input files to DataFrames
#########################################################################################
df_input=pd.read_csv(path+'Factors_decomposed_trimmed.csv', encoding='utf_8', low_memory=False)
#df_input=pd.read_csv(path+'Factors_decomposed.csv', encoding='utf_8', low_memory=False, dtype={'SR.X': np.str,'SR.Y': np.str})
df_input.drop([col for col in df_input.columns if "Unnamed" in col], axis=1, inplace=True)
df_map_riskclass=pd.read_csv(path+'MapRiskClass.csv')
df_map_sectors=pd.read_csv(path+'Map_Sectors.csv',encoding='ISO-8859-1',dtype={'GICSCode':np.str, 'Equity Sector': np.str})
df_map_credit_quality=pd.read_csv(path+'Map_Credit_Quality.csv')
df_countrygroups=pd.read_csv(path+'CountryGroups_v2.csv')
df_issue=pd.read_csv(path+'Issue.csv',encoding='ISO-8859-1', low_memory=False)
df_issuer=pd.read_csv(path+'Issuer.csv')
df_fx_rates=pd.read_csv(path+'FXRates.csv')
df_equity_buckets=pd.read_csv(path+'Equities_Buckets.csv')
df_csr_nonsec_buckets=pd.read_csv(path+'CSR_NonSec_Buckets.csv')
df_params_equities=pd.read_csv(path+'Parameters_Equities.csv')
df_params_csr_nonsec=pd.read_csv(path+'Parameters_CSR_NonSec.csv')
############################################################################################


def get_position_df(position_id,df):
    # Filter the Position ID
    df=pd.DataFrame(df[df['PositionID']==position_id])
    return df


class datestring():
    def __init__(self, string):
        self.string = string.upper()
        
    def getyears(self):
        s=self.string
        if s.find('Y')==-1:
            return 0
        else:
            s=float(s[0:s.find('Y')])
        return s
    
    def getmonths(self):
        s=self.string
        if s.find('M')==-1:
            return 0
        else:
            s=float(s[s.find('Y')+1:s.find('M')])
        return s

    def getdays(self):
        s=self.string
        if s.find('D')==-1:
            return 0
        elif s.find('M')==-1:
            s=float(s[s.find('Y')+1:s.find('D')])
        else:
            s=float(s[s.find('M')+1:s.find('D')])
        return s

    def getdays365(self):
        d=self.getdays()
        m=self.getmonths()
        y=self.getyears()
        
        days365=y*365+m/12*365+d

        return days365


    def getyears365(self):
        d=self.getdays()
        m=self.getmonths()
        y=self.getyears()
        
        years365=y+m/12+d/365
        
        return years365

        

class girrcalc():
    def __init__(self):
        risk_weights=[['0.25y',datestring('0.25y').getdays365(), 0.024],
               ['0.5y',datestring('0.5y').getdays365(), 0.024],
               ['1y', datestring('1y').getdays365(),0.0225],
               ['2y', datestring('2y').getdays365(),0.0188],
               ['3y', datestring('3y').getdays365(),0.0173],
               ['5y', datestring('5y').getdays365(),0.015],
               ['10y', datestring('10y').getdays365(),0.015],
               ['15y', datestring('15y').getdays365(),0.015],
               ['20y', datestring('20y').getdays365(),0.015],
               ['30y', datestring('30y').getdays365(),0.015],
               ['0y', datestring('0y').getdays365(),0.0]
               ]
        #['0y', datestring('0y').getdays365(),0.0]
        self.risk_weights=risk_weights
        
    def getriskweight(self, factor_category,t):
        #g=[item[2]for item in self.risk_weights if item[1]==t]
        if factor_category=='Currency_Basis': g=0.0225    
        else:g=[item[2]for item in self.risk_weights if item[1]==t][0]
        return g

    def calc_weighted_sensi(self,sensi):
        return np.dot(sensi*self.risk_weights)    
        
 



# Function to convert list elements to float
def convertlistofliststofloat(_list):
    output_list=[]
    for l in _list:
        #print(l)
        l=[float(i) for i in l]
        output_list.append(l)
    return output_list


def map_risk_class(df,df_map_riskclass):
    # Map the Risk Class    
    # Inputs: df: Intput data, df_map_risk_class: mapping dataframe
    # Outputs: Input dataframe with the Risk Classes
    return pd.merge(df,df_map_riskclass, on=['FactorCategory'], how='left')


#############################################################################################
#       Filter functions - applied on dataframes
############################################################################################

def filter_for_sa(df):
    # Filter for SA only
    return df[(df['Derived Reporting Purpose'].str.contains('SA'))|(df['Derived Reporting Purpose']=='*')]
def filter_for_risk_class(df, risk_class):
    # Filter for Risk Class
    return df[df['FRTB Risk Class']==risk_class]
def filter_for_curvature(df):
    # Filter for Curvature flag
    #return df[df['Curvature']=='TRUE']
    return df[df['Curvature']==True]
def filter_for_delta(df):
    # Filter for Delta measure
    return df[df['Measure']=='Delta']
def filter_for_vega(df):
    # Filter for Vega measure
    return df[df['Measure']=='Vega']
def filter_sa_measure(df, measure):
    # Filter for SA only
    return df[df['SA Measure']==measure]
def filter_girr_bucket(df, bucket):
    # Filter for GIRR bucket
    return df[df['Bucket GIRR']==bucket]
def filter_csr_nonsec_bucket(df, bucket):
    # Filter for CSR NonSec bucket
    return df[df['Bucket CSR NonSec']==bucket]
def filter_equity_bucket(df, bucket):
    # Filter for Equity bucket
    return df[df['FRTB Bucket Equity']==bucket]
def add_bucket_girr(df):         
    df.insert(len(df.columns),'Bucket GIRR',0)
    df.loc[df['FactorCategory']=='Currency_Basis', ['Bucket GIRR']]=df['CCY2']
    df.loc[df['FactorCategory']!='Currency_Basis', ['Bucket GIRR']]=df['CCY1']
    return df
def map_bucket_equity(df, df_map_sector):         
    df=pd.merge(df,df_map_riskclass, on=['FactorCategory'], how='left')
    return df
def map_issuer_country(df, df_issuer):
    df=pd.merge(df,df_issuer[['LocalIssuerID','CountryOfRisk']], on=['LocalIssuerID'], how='left')
    return df
def map_issuer_countrygroup(df, df_countrygroup):
    #df=pd.merge(df,df_countrygroup[['CountryOfRisk','CountryGroup']], left_on=['IssuerCountry'], right_on=['CountryOfRisk'], how='left')
    return df
def map_issuer_market_cap(df, df_market_cap):
    df=pd.merge(df,df_market_cap[['LocalIssuerID','Issuer MarketCap USD', 'Market Cap Bucket']], on=['LocalIssuerID'],how='left')
    return df
def map_issuer_sector(df, df_map_sector):
    df=pd.merge(df,df_map_sector[['SectorCode','Equity Sector']], left_on=['SectorCode2'], right_on=['SectorCode'], how='left')
    return df
def map_fx_rates(df, df_fx_rates):
    df=pd.merge(df,df_fx_rates[['Ccy','FXRate']], left_on=['MarketCap Currency'], right_on=['Ccy'], how='left')
    return df
def trim_sector_codes(df):
    df['SectorCode2']=df['SectorCode'].str[0:6]
    return df

#########################################################################################################################



def get_delta_girr_from_shock_records(x,V_x,sr_change_type,shifted_gps):

    delta_girr=0
    
    if len(x)==len(V_x):
        
        if not shifted_gps:
            x.append(0)
            V_x.append(0)
            if sr_change_type=='Absolute':
                V=interpolate.interp1d(np.array(x),np.array(V_x), kind='linear', fill_value='extrapolate')
                delta_girr=V(0.0001)/0.0001    
        else:
            
            if sr_change_type=='Absolute':
            #if sr_change_type=='Absolute' and len(x)==1:
                idx=[i for i,v in enumerate(x) if v==0]
                #i=[int(item) for item in i]
                #delta_girr=float(V_x[0])/0.0001    
                #print(idx[0])
                delta_girr=float(V_x[idx[0]])/0.0001    
        
    return delta_girr


def get_vega_girr_from_shock_records(x,V_x,sr_change_type,shifted_gps, base_value):

    vega_girr=0
    #if not shifted_gps and sr_change_type=='Absolute':
    
    if len(x)==len(V_x):
    
        if not shifted_gps:
            x.append(0)
            V_x.append(0)
            if sr_change_type=='Absolute':
                V=interpolate.interp1d(np.array(x),np.array(V_x), kind='linear', fill_value='extrapolate')
                vega_girr=V(0.01)/0.01*base_value    
            if sr_change_type=='Relative':
                x=[item*base_value for item in x]
                #print(base_value,x,V_x)
                V=interpolate.interp1d(np.array(x),np.array(V_x), kind='linear', fill_value='extrapolate')
                
                vega_girr=V(0.01)/0.01*base_value
        
        elif shifted_gps:
            if sr_change_type=='Absolute' and len(x)==1:
                vega_girr=float(V_x[0])/0.01*base_value    
        
        
    return vega_girr


def get_vega_equity_from_shock_records(x,V_x,sr_change_type,shifted_gps, base_value):

    vega_equity=0
    #if not shifted_gps and sr_change_type=='Absolute':
    
    if len(x)==len(V_x):   
        if not shifted_gps:
            x.append(0)
            V_x.append(0)
            if sr_change_type=='Absolute':
                V=interpolate.interp1d(np.array(x),np.array(V_x), kind='linear', fill_value='extrapolate')
                vega_equity=V(0.01)/0.01*base_value    
            if sr_change_type=='Relative':
                x=[item*base_value for item in x]
                #print(base_value,x,V_x)
                V=interpolate.interp1d(np.array(x),np.array(V_x), kind='linear', fill_value='extrapolate')
                
                vega_equity=V(0.01)/0.01*base_value
        
        elif shifted_gps:
            if sr_change_type=='Absolute' and len(x)==1:
                vega_equity=float(V_x[0])/0.01*base_value    
        else:
            vega_equity=0
    else:
        vega_equity=0
        
    return vega_equity


def get_delta_equity_from_shock_records(x,V_x,sr_change_type,shifted_gps, base_value):

    delta_equity=0
    #if not shifted_gps and sr_change_type=='Absolute':
    #if not shifted_gps:
    x.append(0)
    V_x.append(0)
 
    if len(x)==len(V_x):
        if sr_change_type=='Relative':
            #x=[item*base_value for item in x]
            #print(base_value,x,V_x)
            V=interpolate.interp1d(np.array(x),np.array(V_x), kind='linear', fill_value='extrapolate')
            delta_equity=V(0.01)/0.01

    else:
        delta_equity=0

    return delta_equity

def get_sensitivities_from_shock_records(x,V_x,sr_change_type,risk_class,sensi,shifted_gps):
    
    if not shifted_gps and sr_change_type=='Absolute':
        V=interpolate.interp1d(np.array(x),np.array(V_x), kind='linear', fill_value='extrapolate')
        delta_girr=V(0.0001)/0.0001    
    
    return delta_girr



########################### MHSC Data Only ####################################
df_input=df_input[df_input['SourceEntity']=='MHSC']
df_issue=df_issue[df_issue['SourceEntity']=='MHSC']
df_issuer=df_issuer[df_issuer['SourceEntity']=='MHSC']


################################ Risk Class and SA Reporting Purpose ##################
df_input_mapped=map_risk_class(df_input,df_map_riskclass)
#print(df_input_mapped[(df_input_mapped['PositionID']=='137-EQOPIDX-JPY') & (df_input_mapped['Grid']=='P')])
df_input_mapped=filter_for_sa(df_input_mapped)


###############################       TO BE REMOVED ############################
#df_input_mapped=df_input_mapped[df_input_mapped['FactorCategory']!='CDS']
df_input_mapped=df_input_mapped[df_input_mapped['FactorCategory']!='Gov_FutureBasis']
###################################################################################    

#############################################################################################
##########################################################################################

# Sector Code & Issuer Ratings for CDS (Issuer Sector Codes)
df_input_mapped.rename(columns={'LocalIssuerID':'factor.LocalIssuerID'}, inplace=True)
df_input_mapped=pd.merge(df_input_mapped,df_issuer[['LocalIssuerID','Final SectorCode','IssuerRating']], left_on=['factor.LocalIssuerID'], right_on=['LocalIssuerID'], how='left')
df_input_mapped=df_input_mapped.rename(columns={'LocalIssuerID':'issuer.LocalIssuerID'})
df_input_mapped=df_input_mapped.rename(columns={'Final SectorCode':'SectorCode CDS'})
df_input_mapped=df_input_mapped.rename(columns={'IssuerRating_y':'IssuerRating CDS'})
#df_input_mapped=df_input_mapped.rename(columns={})


# Sector Code & Issuer Rating for Equity/ Bond (Issue Sector Codes)
# Get Issuer ID
df_input_mapped=pd.merge(df_input_mapped,df_issue[['LocalIssuerID','Global_Issue_ID']], left_on=['GlobalIssueID'], right_on=['Global_Issue_ID'], how='left')
df_input_mapped=df_input_mapped.rename(columns={'LocalIssuerID':'LocalIssuerID Equity_Bond'})
df_input_mapped=pd.merge(df_input_mapped,df_issuer[['LocalIssuerID','Final SectorCode','IssuerRating']],left_on=['LocalIssuerID Equity_Bond'], right_on=['LocalIssuerID'], how='left')
df_input_mapped=df_input_mapped.rename(columns={'Final SectorCode':'SectorCode Equity_Bond'})
df_input_mapped=df_input_mapped.rename(columns={'IssuerRating':'IssuerRating Equity_Bond'})




df_input_mapped.insert(len(df_input_mapped.columns),'Final SectorCode',0)
df_input_mapped.insert(len(df_input_mapped.columns),'Final IssuerRating',0)


df_input_mapped.loc[df_input_mapped['FactorCategory']!='CDS', ['Final SectorCode']]=df_input_mapped['SectorCode Equity_Bond']
df_input_mapped.loc[df_input_mapped['FactorCategory']=='CDS', ['Final SectorCode']]=df_input_mapped['SectorCode CDS']

#df_input_mapped['Final SectorCode L3']=df_input_mapped['Final SectorCode'].str[0:6]
#df_input_mapped=trim_sector_codes(df_input_mapped)


df_input_mapped.loc[df_input_mapped['FactorCategory']!='CDS', ['Final IssuerRating']]=df_input_mapped['IssuerRating Equity_Bond']
df_input_mapped.loc[df_input_mapped['FactorCategory']=='CDS', ['Final IssuerRating']]=df_input_mapped['IssuerRating CDS']

#print(df_input_mapped[(df_input_mapped['PositionID']=='533-CPBOND-JPY-20280420') & (df_input_mapped['GlobalIssueID']=='BBG00KK1VBT0')])


# Credit Sector mapped directly from csv

df_input_mapped=pd.merge(df_input_mapped,df_map_sectors[['GICSCode','Credit Sector']], left_on=['Final SectorCode'], right_on=['GICSCode'], how='left')
#print(df_input_mapped[(df_input_mapped['PositionID']=='110-EQIDXFUT-KRW') & (df_input_mapped['GlobalIssueID']=='BBG001H06FY1')])
#print(df_input_mapped[(df_input_mapped['PositionID']=='533-CPBOND-JPY-20280420') & (df_input_mapped['GlobalIssueID']=='BBG00KK1VBT0')])

# Default bucket where not found
df_input_mapped['Credit Sector'].fillna(16, inplace=True)

##########################################################################################
###########################################################################################


########################### GIRR Bucket ############################################
df_input_mapped=add_bucket_girr(df_input_mapped)


########################## Equities Bucket ###############################################

# Preparation for equity issue mapping to FRTB bucket
df_equity_issue=df_issue[df_issue['IssueType']=='Equity']
df_equity_issue_mapped=pd.merge(df_equity_issue,df_fx_rates[['Ccy','FXRate']], left_on=['MarketCap Currency'], right_on=['Ccy'], how='left')
df_equity_issue_mapped[['MarketCap','FXRate']]=df_equity_issue_mapped[['MarketCap','FXRate']].astype(float)

#df_equity_issue_mapped['Issuer MarketCap USD']=df_equity_issue_mapped['MarketCap']/df_equity_issue_mapped['FXRate']
df_equity_issue_mapped['MarketCap USD']=df_equity_issue_mapped['MarketCap']/df_equity_issue_mapped['FXRate']


#df_equity_issue_mapped=df_equity_issue_mapped.groupby(['LocalIssuerID'], as_index=False)['MarketCap USD'].sum()

df_equity_issue_mapped['Market Cap Bucket']='SmallCap'
df_equity_issue_mapped.loc[df_equity_issue_mapped['MarketCap USD']>2000000000,'Market Cap Bucket']='LargeCap'


# Issuer Sector added to Factor data - no default sector has been set
#df_input_mapped=pd.merge(df_input_mapped,df_map_sectors[['SectorCode','Equity Sector']], left_on=['Final SectorCode L3'], right_on=['SectorCode'], how='left')
df_input_mapped=pd.merge(df_input_mapped,df_map_sectors[['GICSCode','Equity Sector']], left_on=['Final SectorCode'], right_on=['GICSCode'], how='left')


# Set Market Cap Default to Small Cap
#df_input_mapped['Market Cap Bucket Default']='SmallCap'

# Bring in Market Cap bucket to Factor data
df_input_mapped['GlobalIssueID']=df_input_mapped['GlobalIssueID'].astype(str)
df_equity_issue_mapped['LocalIssuerID']=df_equity_issue_mapped['LocalIssuerID'].astype(str)
df_input_mapped=pd.merge(df_input_mapped,df_equity_issue_mapped[['Global_Issue_ID','MarketCap USD', 'Market Cap Bucket']], left_on=['GlobalIssueID'],right_on=['Global_Issue_ID'], how='left')


# Default Market Cap: Any empty Market Caps Buckets -> Small Cap
df_input_mapped['Market Cap Bucket'].fillna('SmallCap', inplace=True)

# Bring in Economy (EM/Advanced) to Factor data - so far no need to set a default
df_input_mapped=pd.merge(df_input_mapped,df_countrygroups[['CountryOfRisk','Economy']], left_on=['IssuerCountry'],right_on=['CountryOfRisk'], how='left')


#df_input_mapped['Equity Sector']=df_input_mapped['Equity Sector'].astype(float)


# Bring in Final FRTB Equity bucket to Factor data
df_input_mapped['EquityBucketKey']=df_input_mapped['Market Cap Bucket'].astype(str)+df_input_mapped['Economy'].astype(str)+df_input_mapped['Equity Sector'].astype(str)

df_equity_buckets['EquityBucketKey']=df_equity_buckets['EquityBucketKey'].astype(str)
df_input_mapped.loc[df_input_mapped['Equity Sector']==11,['EquityBucketKey']]=11
df_input_mapped['EquityBucketKey']=df_input_mapped['EquityBucketKey'].astype(str)
df_input_mapped=pd.merge(df_input_mapped,df_equity_buckets[['EquityBucketKey','FRTB Bucket Equity']], on=['EquityBucketKey'], how='left')

##################################################################################################


######################################## CSR Non Sec Bucket ########################################################
# Bring in the Issue Credit Quality
df_input_mapped=pd.merge(df_input_mapped,df_map_credit_quality, left_on=['Final IssuerRating'],right_on=['Rating'], how='left')
df_input_mapped.loc[df_input_mapped['Final IssuerRating'].isnull(),['Credit Quality']]='HY'

# Determine FRTB Credit Bucket
# Construct Key
df_input_mapped['CSR NonSec Bucket Key']=df_input_mapped['Credit Quality'].astype(str)+df_input_mapped['Credit Sector'].astype(str)
df_input_mapped.loc[df_input_mapped['Credit Sector']==16,['CSR NonSec Bucket Key']]=16
df_input_mapped['CSR NonSec Bucket Key']=df_input_mapped['CSR NonSec Bucket Key'].astype(str)
df_input_mapped=pd.merge(df_input_mapped,df_csr_nonsec_buckets[['CSR NonSec Bucket Key','Bucket CSR NonSec']], on=['CSR NonSec Bucket Key'], how='left')

################################################################################################

######################################### FRTB FX Bucket ######################################################################
df_input_mapped['FRTB Bucket FX']=np.nan
df_input_mapped.loc[(df_input_mapped['FactorCategory']=='FX_Rate') & (df_input_mapped['CCY1']!='JPY'),['FRTB Bucket FX']]=df_input_mapped['CCY1'].loc[(df_input_mapped['FactorCategory']=='FX_Rate') & (df_input_mapped['CCY1']!='JPY')]
df_input_mapped.loc[(df_input_mapped['FactorCategory']=='FX_Rate') & (df_input_mapped['CCY2']!='JPY'),['FRTB Bucket FX']]=df_input_mapped['CCY2'].loc[(df_input_mapped['FactorCategory']=='FX_Rate') & (df_input_mapped['CCY2']!='JPY')]
###################################################################################################################################







################## DERIVED SENSITIVITIES - DIRECT OR FROM SHOCK RECORDS ##############################
df_input_mapped.insert(len(df_input_mapped.columns),'Derived Sensitivity',0)
df_input_mapped.loc[df_input_mapped['SA Measure']=='SENS',['Derived Sensitivity']]=df_input_mapped['Sensitivity'].loc[df_input_mapped['SA Measure']=='SENS']

########################## Trim the Shock Record Data for leading / trailing spaces #########
df_input_mapped['SR.X']=df_input_mapped['SR.X'].str.strip()
df_input_mapped['SR.Y']=df_input_mapped['SR.Y'].str.strip()


########################## GIRR Delta sensi from shock records  #################################################
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SENS')&(df_input_mapped['Measure']=='Delta'),['Derived Sensitivity']]=df_input_mapped['Sensitivity'].loc[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Delta')]/0.0001
all_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Delta')]['SR.X'].str.split(' '))
all_x=convertlistofliststofloat(all_x)
all_V_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Delta')]['SR.Y'].str.split(' '))
all_V_x=convertlistofliststofloat(all_V_x)
change_type=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Delta')]['SRChangeType'])
shifted_gps=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Delta')]['ShiftedGPS'])
p=[get_delta_girr_from_shock_records(x,V_x,ct,sgps)for x,V_x,ct,sgps in zip(all_x,all_V_x,change_type,shifted_gps)]
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Delta'),['Derived Sensitivity']]=np.array(p)
########################## GIRR Vega sensi from shock records  #################################################
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Vega'),['Derived Sensitivity']]=df_input_mapped['Sensitivity'].loc[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Delta')]/0.01
all_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Vega')]['SR.X'].str.split(' '))
all_x=convertlistofliststofloat(all_x)
all_V_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Vega')]['SR.Y'].str.split(' '))
all_V_x=convertlistofliststofloat(all_V_x)
change_type=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Vega')]['SRChangeType'])
shifted_gps=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Vega')]['ShiftedGPS'])
base_values=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Vega')]['BaseValue'])
p=[get_vega_girr_from_shock_records(x,V_x,ct,sgps,bv)for x,V_x,ct,sgps,bv in zip(all_x,all_V_x,change_type,shifted_gps,base_values)]
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='GIRR') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Vega'),['Derived Sensitivity']]=np.array(p)
########################## CSR Non Sec Delta  sensi from shock records #################################################
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='CSR Non Sec') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Delta'),['Derived Sensitivity']]=df_input_mapped['Sensitivity'].loc[(df_input_mapped['FRTB Risk Class']=='CSR Non Sec') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Delta')]/0.0001
all_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='CSR Non Sec') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['SR.X'].str.split(' '))
all_x=convertlistofliststofloat(all_x)
all_V_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='CSR Non Sec') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['SR.Y'].str.split(' '))
all_V_x=convertlistofliststofloat(all_V_x)
change_type=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='CSR Non Sec') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['SRChangeType'])
shifted_gps=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='CSR Non Sec') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['ShiftedGPS'])
p=[get_delta_girr_from_shock_records(x,V_x,ct,sgps)for x,V_x,ct,sgps in zip(all_x,all_V_x,change_type,shifted_gps)]
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='CSR Non Sec') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Delta'),['Derived Sensitivity']]=np.array(p)
########################## Equity Delta sensi from shock records  #################################################
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Delta'),['Derived Sensitivity']]=df_input_mapped['Sensitivity'].loc[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Delta')]/0.01
all_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['SR.X'].str.split(' '))
all_x=convertlistofliststofloat(all_x)
all_V_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['SR.Y'].str.split(' '))
all_V_x=convertlistofliststofloat(all_V_x)
change_type=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['SRChangeType'])
shifted_gps=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['ShiftedGPS'])
base_values=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Delta')]['BaseValue'])
p=[get_delta_equity_from_shock_records(x,V_x,ct,sgps,bv)for x,V_x,ct,sgps,bv in zip(all_x,all_V_x,change_type,shifted_gps, base_values)]
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Delta'),['Derived Sensitivity']]=np.array(p)
########################## Equity Vega  sensi from shock records #################################################
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Vega'),['Derived Sensitivity']]=df_input_mapped['Sensitivity'].loc[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Vega')]/0.01
all_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Vega')]['SR.X'].str.split(' '))
all_x=convertlistofliststofloat(all_x)
all_V_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Vega')]['SR.Y'].str.split(' '))
all_V_x=convertlistofliststofloat(all_V_x)
change_type=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Vega')]['SRChangeType'])
shifted_gps=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Vega')]['ShiftedGPS'])
base_values=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Vega')]['BaseValue'])
p=[get_vega_equity_from_shock_records(x,V_x,ct,sgps,bv)for x,V_x,ct,sgps,bv in zip(all_x,all_V_x,change_type,shifted_gps, base_values)]
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='Equity') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Vega'),['Derived Sensitivity']]=np.array(p)
########################## FX  Delta  sensi from shock records #################################################
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Delta'),['Derived Sensitivity']]=df_input_mapped['Sensitivity'].loc[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Delta')]/0.01
all_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['SR.X'].str.split(' '))
all_x=convertlistofliststofloat(all_x)
all_V_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['SR.Y'].str.split(' '))
all_V_x=convertlistofliststofloat(all_V_x)
change_type=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['SRChangeType'])
shifted_gps=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Delta')]['ShiftedGPS'])
base_values=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Delta')]['BaseValue'])
p=[get_delta_equity_from_shock_records(x,V_x,ct,sgps,bv)for x,V_x,ct,sgps,bv in zip(all_x,all_V_x,change_type,shifted_gps, base_values)]
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Delta'),['Derived Sensitivity']]=np.array(p)
########################## FX Vega  sensi from shock records  #################################################
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Vega'),['Derived Sensitivity']]=df_input_mapped['Sensitivity'].loc[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SENS') & (df_input_mapped['Measure']=='Delta')]/0.01
all_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Vega')]['SR.X'].str.split(' '))
all_x=convertlistofliststofloat(all_x)
all_V_x=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Vega')]['SR.Y'].str.split(' '))
all_V_x=convertlistofliststofloat(all_V_x)
change_type=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Vega')]['SRChangeType'])
shifted_gps=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR')& (df_input_mapped['Measure']=='Vega')]['ShiftedGPS'])
base_values=list(df_input_mapped[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Vega')]['BaseValue'])
p=[get_vega_equity_from_shock_records(x,V_x,ct,sgps,bv)for x,V_x,ct,sgps,bv in zip(all_x,all_V_x,change_type,shifted_gps, base_values)]
df_input_mapped.loc[(df_input_mapped['FRTB Risk Class']=='FX') & (df_input_mapped['SA Measure']=='SR') & (df_input_mapped['Measure']=='Vega'),['Derived Sensitivity']]=np.array(p)
#########################################################################################################################################################





positions=df_input_mapped.PositionID.unique().tolist()


## Pre-processing: steps to get the data in the required shape before calcs
## 



class result():
    def __init__(self, position_id, measure, expected_result, risk_class):
        self.position_id = position_id
        self.measure = measure
        self.expected_result=expected_result
        self.risk_class=risk_class



def outside_list_range(mylist,val):
    #returns true if val is not within the range of values in mylist
    if above_list_maximum(mylist,val):
        return True
    elif below_list_minimum(mylist,val):
        return True
    else:
        return False
    
    
        
def above_list_maximum(mylist,val):
    #returns true if val is not within the range of values in mylist
    maxlist=max(mylist)
    if val>maxlist:
        return True
    else:
        return False
    
def below_list_minimum(mylist,val):
    #returns true if val is not within the range of values in mylist
    minlist=min(mylist)
    if val<minlist:
        return True
    else:
        return False
    

def get_boundary_points(mylist,val):
    # Inputs: Given input 
    #   (mylist): list of times  
    #   val: particular value 
    # Output: return the interval [t1,t2] of mylist containing val
    mylist.sort()
    
    for i,v in enumerate(mylist):
        if i<len(mylist)-1 and val>mylist[i] and val<mylist[i+1]:
            t1=mylist[i]
            t2=mylist[i+1]     
            #times=[t1,t2]
            break
        elif i<len(mylist)-1 and val==mylist[i]:
            #print('yes')
            t1=mylist[i]
            t2=mylist[i+1]         
            
    if val<=min(mylist):
        t1=min(mylist)
        t2=mylist[1]
        #times=[t1,t2]
            
    if val>=max(mylist):
        t1=mylist[-2]
        t2=max(mylist)
        
    times=[t1,t2]
    
    return times



def assign_weights(exposure_tenor, tenor1, tenor2):
    # Inputs: exposure tenors, mapped tenors t1/t2
    # Outputs: weights to be applied to input tenors t1/t2
    if exposure_tenor==0:
        weight_tenor1=0
        weight_tenor2=0
    elif (exposure_tenor>tenor1) & (exposure_tenor>tenor2):
        #print(exposure_tenor)
        weight_tenor1=0
        weight_tenor2=1
    elif (exposure_tenor<tenor1) & (exposure_tenor<tenor2):
        weight_tenor1=1
        weight_tenor2=0
    elif tenor1!=tenor2: 
        # Calculate the weights based on distance
        weight_tenor1=1-abs(exposure_tenor-tenor1)/abs(tenor2-tenor1)
        weight_tenor2=1-abs(exposure_tenor-tenor2)/abs(tenor2-tenor1)
    else:
        weight_tenor1=0
        weight_tenor2=0
    return [weight_tenor1,weight_tenor2]


####### TO bE REMOVED - NOT USED #################

def assign_weights2(exposure_tenor, tenor1, tenor2):
    # Inputs: exposure tenors, mapped tenors t1/t2
    # Outputs: weights to be applied to input tenors t1/t2
    if exposure_tenor==0:
        weight_tenor1=0
        weight_tenor2=0
    #elif (exposure_tenor>tenor1) & (exposure_tenor>tenor2):

    elif tenor1==0:        
        #print(exposure_tenor)
        weight_tenor1=0
        weight_tenor2=1
    #elif (exposure_tenor<tenor1) & (exposure_tenor<tenor2):
    elif tenor2==0:        
        weight_tenor1=1
        weight_tenor2=0
    elif tenor1!=tenor2: 
        # Calculate the weights based on distance
        weight_tenor1=1-abs(exposure_tenor-tenor1)/abs(tenor2-tenor1)
        weight_tenor2=1-abs(exposure_tenor-tenor2)/abs(tenor2-tenor1)
    else:
        weight_tenor1=0
        weight_tenor2=0
    return [weight_tenor1,weight_tenor2]




def assign_weights_2d(x,y,x1,x2,y1,y2):
    # Inputs: exposure tenors, mapped tenors t1/t2
    # Outputs: weights to be applied to input tenors t1/t2
    #           x1/x2 being option matrurity
    #           y1/y2 as underlying maturity
    
   # print(x,y)
    
    if x==0 and y==0:
        w_x1_y1=0
        w_x1_y2=0
        w_x2_y1=0
        w_x2_y2=0
    elif x1!=x2 and y1!=y2: 
        # Calculate the weights based on distance
        #weight_tenor1=1-abs(exposure_tenor-tenor1)/abs(tenor2-tenor1)
        #weight_tenor2=1-abs(exposure_tenor-tenor2)/abs(tenor2-tenor1)
        w_x1_y1=1/((x2-x1)*(y2-y1))*(x2-x)*(y2-y)
        w_x1_y2=1/((x2-x1)*(y2-y1))*(x2-x)*(y-y1)
        w_x2_y2=1/((x2-x1)*(y2-y1))*(x-x1)*(y-y1)
        w_x2_y1=1/((x2-x1)*(y2-y1))*(x-x1)*(y2-y)    
    else:
        w_x1_y1=0
        w_x1_y2=0
        w_x2_y1=0
        w_x2_y2=0
    
    return [w_x1_y1,w_x1_y2,w_x2_y1,w_x2_y2]


    

def calc_position_fx_delta_bucket_risk_position(position_id, bucket, corr_high_low_med):
    df=get_position_df(position_id, df_input_mapped)
    df=filter_for_risk_class(df,'FX')
    df=filter_for_delta(df)
    df=pd.merge(df,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')
    
    
    df_bucket=df.copy()
    
    df_bucket=df_bucket[df_bucket['FRTB Bucket FX']==bucket]
    
    df_bucket['Risk Weight']=sa.rw_fx_delta

    df_bucket['Weighted Sensi']=df_bucket['Derived Sensitivity']*df_bucket['Risk Weight']
    df_bucket['Abs Weighted Sensi']=abs(df_bucket['Weighted Sensi'])
    df_bucket['Weighted Sensi USD']=df_bucket['Weighted Sensi']/df_bucket['FXRate']
    df_bucket['Abs Weighted Sensi USD']=df_bucket['Abs Weighted Sensi']/df_bucket['FXRate']
    df_bucket['Derived Sensi USD']=df_bucket['Derived Sensitivity']/df_bucket['FXRate']


    kb=sum(df_bucket['Weighted Sensi USD'])
    
    
    res={'position_id': position_id, 'weighted_sensi': df_bucket['Weighted Sensi USD'].tolist(), 'K_b':kb, 
             'factors': df_bucket['FactorStructure'].tolist(), 'RWs': df_bucket['Risk Weight'].tolist(), 
             'Sensis': df_bucket['Derived Sensi USD'].tolist(), 'Factor_Buckets': df_bucket['FRTB Bucket FX'].tolist()}
    return res



def calc_position_fx_delta_weighted_sensi(position_id):
    
    df=get_position_df(position_id, df_input_mapped)
    df=filter_for_risk_class(df,'FX')
    df=filter_for_delta(df)

    df=pd.merge(df,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')

    #print(df_bucket)
    rw=sa.rw_fx_delta

    if df.empty:
        return
    else:    
        df['Risk Weight']=rw
        df=df.groupby(['FactorStructure', 'FactorCategory', 'Risk Weight','FXRate'], as_index=False)['Derived Sensitivity'].sum()
        
        df['Weighted Sensi']=df['Derived Sensitivity']*df['Risk Weight']
        df['Abs Weighted Sensi']=abs(df['Weighted Sensi'])
        
        df['Weighted Sensi USD']=df['Weighted Sensi']/df['FXRate']
        df['Abs Weighted Sensi USD']=df['Abs Weighted Sensi']/df['FXRate']
        
        df['Derived Sensi USD']=df['Derived Sensitivity']/df['FXRate']

       # ws_b=df['Weighted Sensi USD'].tolist()
        
        
    return [df,df['Weighted Sensi USD'].tolist(),df['FactorStructure'].tolist(),df['Risk Weight'].tolist(), df['Derived Sensi USD'].tolist()]
            


# Calculates the delta risk charge for given position
def calc_position_fx_delta_risk_charge(position_id, corr_high_low_med):
    # Filter the input data
    df=get_position_df(position_id,df_input_mapped)
    df=filter_for_risk_class(df,'FX')
    df=filter_for_delta(df)
    
    if df.empty:
        return
    else:
        buckets=df['FRTB Bucket FX'].dropna().unique().tolist()
        
        output=[calc_position_fx_delta_bucket_risk_position(position_id, bucket, corr_high_low_med) for bucket in buckets]
        
        d={}
        
        for key in output[0]:
            d[key]=list(d[key] for d in output)
        
        weighted_sensi=d['weighted_sensi']
        factors=d['factors']
        risk_weights=d['RWs']
        factor_sensi=d['Sensis']

        weighted_sensi=reduce(lambda x,y: x+y,weighted_sensi,[])
        factors=reduce(lambda x,y: x+y,factors,[])
        risk_weights=reduce(lambda x,y: x+y,risk_weights,[])
        factor_sensi=reduce(lambda x,y: x+y,factor_sensi,[])
        
        corr= sa.getcorr(sa.corr_fx_ybc,corr_high_low_med)
        y_bc=sa.corr_matrix_bucket(corr,len(factors))
        delta_risk_charge=sa.fx_delta_risk_charge(weighted_sensi,y_bc,np.array(weighted_sensi))  
    
    res={'position_id': position_id, 'weighted_sensi': weighted_sensi, 'K_b':weighted_sensi, 
             'factors': factors, 'RWs': risk_weights, 
             'Sensis': factor_sensi, 'delta_rc': delta_risk_charge, 'Buckets': buckets}
    
    return res



def calc_position_fx_vega_bucket_risk_position(position_id,bucket,corr_high_low_med):

    
    frtb_vertices=['0.5y','1y','3y','5y','10y']
    frtb_vertices_days=[datestring(v).getdays365() for v in frtb_vertices]
    frtb_vertices_years=[datestring(v).getyears365() for v in frtb_vertices]
    
    df=get_position_df(position_id, df_input_mapped)
    df=filter_for_risk_class(df,'FX')
    df=filter_for_vega(df)

   
    df=pd.merge(df,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')

    #print(df_bucket)
    #rw=0.30

    if df.empty:
       return

    else:    

        df1=df.copy()
        df2=df.copy()
        
        # Get FG1 and convert to days365
        factorgrid1=np.array(df['FactorGrid1'])    
        factorgrid1_days=[(datestring(fg).getdays365()) for fg in factorgrid1]
        factorgrid1_years=[(datestring(fg).getyears365()) for fg in factorgrid1]

        # FRTB SA Buckets in days
        vertices_days=frtb_vertices_days
        
        # Get nearest points Tenor1/Tenor2 in days for each given FG1
        tenor1_tenor2=[get_boundary_points(vertices_days,d) for d in factorgrid1_days]
        t1=[item[0] for item in tenor1_tenor2]
        t2=[item[1] for item in tenor1_tenor2]
        
        t1_t2_years=[get_boundary_points(frtb_vertices_years,y) for y in factorgrid1_years]
        t1_years=[item[0] for item in t1_t2_years]
        t2_years=[item[1] for item in t1_t2_years]
        

        # Weights to be assigned to sensitivity when assigning to tenor1/tenor2 
        tenor_weights=np.array([assign_weights(factorgrid1_days[i],tenor1_tenor2[i][0],tenor1_tenor2[i][1]) for i,v in enumerate(factorgrid1_days)])
        
        # Assign input sensitivities to nearest points t1/t2
        sensi_t1=tenor_weights[:,0]*df['Derived Sensitivity']
        sensi_t2=tenor_weights[:,1]*df['Derived Sensitivity']    
    
        risk_weights_t1=np.minimum(40**(1/2)*sa.rw_sigma/(math.sqrt(10)),np.ones(len(df['Derived Sensitivity'])))
        risk_weights_t2=np.minimum(40**(1/2)*sa.rw_sigma/(math.sqrt(10)),np.ones(len(df['Derived Sensitivity'])))
        
        # Weighted sensi for Tenor1/Tenor2 factors
        weighted_sensi_t1=sensi_t1*np.array(risk_weights_t1)
        weighted_sensi_t2=sensi_t2*np.array(risk_weights_t2)
     
        ######################################
        # New dataframes to handle Tenor1/Tenor2
        df1['t']=pd.Series(t1, index=df.index)
        df1['t_years']=pd.Series(t1_years, index=df.index)
        df1['Sensi_t']=pd.Series(sensi_t1, index=df.index)
        df1['Risk Weight_t']=pd.Series(risk_weights_t1, index=df.index)
        df1['Weighted Sensi']=pd.Series(weighted_sensi_t1, index=df.index)
        
        df2['t']=pd.Series(t2, index=df.index)
        df2['t_years']=pd.Series(t2_years, index=df.index)        
        df2['Sensi_t']=pd.Series(sensi_t2, index=df.index)
        df2['Risk Weight_t']=pd.Series(risk_weights_t2, index=df.index)
        df2['Weighted Sensi']=pd.Series(weighted_sensi_t2, index=df.index)
        ##############################
        
        ####################################
        # Dataframe to combine df1/df2 before calculation done
        df3=df1.copy()
        df3=df3.append(df2, ignore_index=True)
        df3=df3.groupby(['FactorStructure', 'FactorCategory', 'Risk Weight_t','CCY1' ,'t', 'FXRate','t_years'], as_index=False)['Sensi_t','Weighted Sensi'].sum()
        df3=df3[df3['t']!=0]
        ###########################################
        
        df3['Weighted Sensi USD']=df3['Weighted Sensi']/df3['FXRate']
        df3['Sensi USD']=df3['Sensi_t']/df3['FXRate']
        
#        corr_matrix=sa.corr_matrix_girr(df3.FactorStructure,df3.FactorCategory, df3.t,corr_high_low_med)
#        k_b=sa.K_b_girr_delta(df3['Weighted Sensi USD'],corr_matrix)    
#        rw=df3['Risk Weight_t'].tolist()
#        s_b=df3['Sensi USD'].tolist()
#        
#        ws_b=df3['Weighted Sensi USD'].tolist()
#        factors=df3['FactorStructure'].tolist()
#        factor_grids=df3['t'].tolist()
        
    return [df3,df3['Weighted Sensi USD'].tolist(),df3['FactorStructure'].tolist(),df3['Risk Weight_t'].tolist(), df3['Sensi USD'].tolist(),
            df3['t'].tolist(),df3['t_years'].tolist(),df,df1,df2]



#def calc_position_fx_delta_bucket_risk_position(position_id, bucket, corr_high_low_med):
#    df=get_position_df(position_id, df_input_mapped)
#    df=filter_for_risk_class(df,'FX')
#    df=filter_for_delta(df)
#    df=pd.merge(df,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')
#    
#    
#    df_bucket=df.copy()
#    
#    df_bucket=df_bucket[df_bucket['FRTB Bucket FX']==bucket]
#    
#    df_bucket['Risk Weight']=sa.rw_fx_delta
#    df_bucket['Weighted Sensi']=df_bucket['Derived Sensitivity']*df_bucket['Risk Weight']
#    df_bucket['Abs Weighted Sensi']=abs(df_bucket['Weighted Sensi'])
#    df_bucket['Weighted Sensi USD']=df_bucket['Weighted Sensi']/df_bucket['FXRate']
#    df_bucket['Abs Weighted Sensi USD']=df_bucket['Abs Weighted Sensi']/df_bucket['FXRate']
#    df_bucket['Derived Sensi USD']=df_bucket['Derived Sensitivity']/df_bucket['FXRate']
#
#    kb=sum(df_bucket['Weighted Sensi USD'])
#        
#    res={'position_id': position_id, 'weighted_sensi': df_bucket['Weighted Sensi USD'].tolist(), 'K_b':kb, 
#             'factors': df_bucket['FactorStructure'].tolist(), 'RWs': df_bucket['Risk Weight'].tolist(), 
#             'Sensis': df_bucket['Derived Sensi USD'].tolist(), 'Factor_Buckets': df_bucket['FRTB Bucket FX'].tolist()}
#    return res


# Calculates the delta risk charge for given position
def calc_position_fx_vega_risk_charge(position_id, corr_high_low_med):
    # Filter the input data
    df=get_position_df(position_id,df_input_mapped)
    df=filter_for_risk_class(df,'FX')
    df=filter_for_vega(df)
    
    if df.empty:
        weighted_sensi=[]
        risk_weights=[]
        factor_sensi=[]
        vega_risk_charge=0
        factors=[]
        factorgrid1_years=[]
    else:
        output=calc_position_fx_vega_weighted_sensi(position_id)
        weighted_sensi=output[1]
        factors=output[2]
        risk_weights=output[3]
        factor_sensi=output[4]
        factorgrid1_years=output[6]
    
        corr=sa.getcorr(0.60,corr_high_low_med)
        
        corr_matrix_delta=sa.corr_matrix_bucket(corr,len(factors))
        maturity_corr_matrix=option_maturity_corr_matrix(factorgrid1_years)        
        corr_matrix_vega=vega_corr_matrix(corr_matrix_delta,maturity_corr_matrix)
                
        vega_risk_charge=sa.delta_vega_risk_charge(weighted_sensi,corr_matrix_vega,np.array(weighted_sensi))  
    
    return [position_id,weighted_sensi,vega_risk_charge,factors, risk_weights,factor_sensi,factorgrid1_years]


def calc_position_fx_curvature_sensi(position_id):
    
    df=get_position_df(position_id, df_input_mapped)
    df=filter_for_risk_class(df,'FX')
    df=filter_for_delta(df)

    #df_bucket=df
    df=pd.merge(df,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')
    
    #print(df_bucket)
    rw=0.30

    if df.empty:
        df['CVR USD']=[]
        df['FactorStructure']=[]
        df['Curvature Risk Weight']=[]
        df['V_up USD']=[]
        df['V_down USD']=[]
        df['CVR_up USD']=[]
        df['CVR_down USD']=[]
    else:    
        df.insert(len(df.columns),'V_up',0)
        df.insert(len(df.columns),'V_down',0)
        df.insert(len(df.columns),'CVR_up',0)
        df.insert(len(df.columns),'CVR_down',0)
        df.insert(len(df.columns),'CVR',0)
        df.insert(len(df.columns),'V_up USD',0)
        df.insert(len(df.columns),'V_down USD',0)
        df.insert(len(df.columns),'CVR_up USD',0)
        df.insert(len(df.columns),'CVR_down USD',0)
        
        df['Curvature Risk Weight']=rw
        
        df=df.groupby(['FactorStructure', 'FactorCategory', 'Curvature Risk Weight','FXRate','SR.X','SR.Y'], as_index=False)['Derived Sensitivity'].sum()

        all_x=list(df['SR.X'].str.split(' '))
        all_x=convertlistofliststofloat(all_x)
        
        all_V_x=list(df['SR.Y'].str.split(' '))
        all_V_x=convertlistofliststofloat(all_V_x)
        
        risk_weights=df['Curvature Risk Weight']
        sensitivities=df['Derived Sensitivity']
        
        V_up=np.empty(len(all_V_x))
        V_down=np.empty(len(all_V_x))
        
        CVR_up=np.empty(len(all_V_x))
        CVR_down=np.empty(len(all_V_x))
        CVR=np.empty(len(all_V_x))
        
        for i,v in enumerate(all_x):
        
            ## To be removed/??
            if len(all_x[i])!=len(all_V_x[i]):
                CVR=0
                
            else:                
                
                V=interpolate.interp1d(np.array(all_x[i]),np.array(all_V_x[i]), kind='linear', fill_value='extrapolate')
                cvr_rw=risk_weights[i]
                
                x_up=cvr_rw
                x_down=-cvr_rw
                
                V_up[i]=V(x_up)
                V_down[i]=V(x_down)
                
                #CVR_up[i]=V_up[i]-cvr_rw*sensitivities[i]
                #CVR_down[i]=V_down[i]-cvr_rw*sensitivities[i]
               
                CVR_up[i]=sa.CVR_up(V_up[i],cvr_rw,sensitivities[i])
                CVR_down[i]=sa.CVR_down(V_down[i],cvr_rw,sensitivities[i])
                
                CVR[i]=-min(CVR_up[i],CVR_down[i])
            
        df['V_up']=V_up
        df['V_down']=V_down
        df['CVR_up']=CVR_up
        df['CVR_down']=CVR_down
        df['CVR']=CVR

        #print(df_bucket['V_up'])
        
        df['V_up USD']=df['V_up']/df['FXRate']
        df['V_down USD']=df['V_down']/df['FXRate']        
        df['CVR_up USD']=df['CVR_up']/df['FXRate']
        df['CVR_down USD']=df['CVR_down']/df['FXRate']
        
        df['Abs CVR']=abs(df['CVR'])
        df['CVR USD']=df['CVR']/df['FXRate']
        df['Abs CVR USD']=df['Abs CVR']/df['FXRate']
        
        #cvr_b=df['CVR USD'].tolist()
        #factors=df['FactorStructure'].tolist()
        
    return [df,df['CVR USD'].tolist(),df['FactorStructure'].tolist(),df['Curvature Risk Weight'].tolist(), df['V_up USD'].tolist(),
            df['V_down USD'].tolist(),df['CVR_up USD'].tolist(),df['CVR_down USD'].tolist()]
     

       
# Calculates the delta risk charge for given position
def calc_position_fx_curvature_risk_charge(position_id, corr_high_low_med):

    # Filter the input data
    df=get_position_df(position_id,df_input_mapped)
    #print(df.head())
    df=filter_for_risk_class(df,'FX')
    df=filter_for_delta(df)
    df=filter_for_curvature(df)
    
    if df.empty:
        
        factor_curvature_sensi=[]
        #bucket_risk_positions=0
        output=[]
        curvature_risk_charge=0
        #buckets=[]
        factors=[]
        risk_weights=[]
        factors_V_up=[]
        factors_V_down=[]        
        factors_CVR_up=[]
        factors_CVR_down=[]
        #factors_bucket=[]
    else:
        # Get all buckets of the position        
        output=calc_position_fx_curvature_sensi(position_id)
        # Bucket Risk positions, excluding 'Other' / 11
        # Bucket Weigted Sensis, excluding 'Other' / 11
        
        factor_curvature_sensi=output[1]
        factors=output[2]
        risk_weights=output[3]
        factors_V_up=output[4]
        factors_V_down=output[5]
        factors_CVR_up=output[6]
        factors_CVR_down=output[7]
        #factors_bucket=[item[2] for item in output]
    
        #print(bucket_risk_positions,bucket_curvature_sensi,factors)
    
        corr= sa.getcorr(0.30*0.30,corr_high_low_med)
    
        y_bc=sa.corr_matrix_bucket(corr,len(factors))
       
        curvature_risk_charge=sa.curvature_risk_charge(factor_curvature_sensi,y_bc,np.array(factor_curvature_sensi))  
        
    
    return [position_id,factor_curvature_sensi, curvature_risk_charge,factors,risk_weights,factors_V_up,factors_V_down,
            factors_CVR_up,factors_CVR_down]




def calc_position_equity_delta_bucket_risk_position(position_id,bucket,corr_high_low_med):
    
    df=get_position_df(position_id, df_input_mapped)
    df=filter_for_risk_class(df,'Equity')
    df=filter_for_delta(df)

    df_bucket=df[df['FRTB Bucket Equity']==bucket]
    df_bucket=pd.merge(df_bucket,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')


    if df_bucket.empty:
        kb_med=0
        ws_b=0
        corr_matrix_med=0
        factors=0
        df_bucket['Risk Weight']=[]
        df_bucket['Derived Sensi USD']=[]
    else:    

        df_params_equities_bucket=df_params_equities[df_params_equities['FRTB Bucket Equity']==bucket]
        
        df_bucket=pd.merge(df_bucket, df_params_equities_bucket, on='FRTB Bucket Equity', how='left')
        df_bucket['Risk Weight']=(df_bucket['FactorCategory']=='Equity')
        df_bucket.loc[df_bucket['Risk Weight']==True,['Risk Weight']]=df_bucket['RW Spot']
        df_bucket.loc[df_bucket['Risk Weight']==False,['Risk Weight']]=df_bucket['RW Repo']
        
        
        df_bucket=df_bucket.groupby(['FRTB Bucket Equity','FactorStructure', 'FactorCategory', 'GlobalIssueID', 'Risk Weight','FXRate','Equity Sector',
                                     'CountryOfRisk','Economy'], as_index=False)['Derived Sensitivity'].sum()
        
        
        
        # Join the local Issuer ID
        df_bucket=pd.merge(df_bucket,df_issue[['LocalIssuerID','Global_Issue_ID']], left_on=['GlobalIssueID'], right_on=['Global_Issue_ID'], how='left')
        df_bucket=df_bucket.rename(columns={'LocalIssuerID':'LocalIssuerID Equity_Bond'})
        
        df_bucket['GlobalIssueID']=df_bucket['GlobalIssueID'].astype(str)
        df_bucket=pd.merge(df_bucket,df_equity_issue_mapped[['Global_Issue_ID','MarketCap USD', 'Market Cap Bucket']], left_on=['GlobalIssueID'],right_on=['Global_Issue_ID'], how='left')
        
        
        df_corr_sptspt_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Spot-Spot)']]
        df_corr_reprep_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Repo-Repo)']]
        df_corr_sptrep_diffissuer_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Spot-Repo Different Issuer)']]
        df_corr_sptrep_sameissuer_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Spot-Repo Same Issuer)']]
             
        corr_matrix=sa.equity_delta_corr_matrix_bucket(df_bucket['FactorCategory'],df_bucket['LocalIssuerID Equity_Bond'],
                                            sa.getcorr(df_corr_sptspt_bucket.iloc[0][1],corr_high_low_med), 
                                            sa.getcorr(df_corr_reprep_bucket.iloc[0,1],corr_high_low_med),
                                            sa.getcorr(df_corr_sptrep_sameissuer_bucket.iloc[0,1],corr_high_low_med),
                                            sa.getcorr(df_corr_sptrep_diffissuer_bucket.iloc[0,1],corr_high_low_med))
        
    
        df_bucket['Weighted Sensi']=df_bucket['Derived Sensitivity']*df_bucket['Risk Weight']
        df_bucket['Abs Weighted Sensi']=abs(df_bucket['Weighted Sensi'])
        
        df_bucket['Weighted Sensi USD']=df_bucket['Weighted Sensi']/df_bucket['FXRate']
        df_bucket['Abs Weighted Sensi USD']=df_bucket['Abs Weighted Sensi']/df_bucket['FXRate']
        
        df_bucket['Derived Sensi USD']=df_bucket['Derived Sensitivity']/df_bucket['FXRate']
        

        ws_b=df_bucket['Weighted Sensi USD'].tolist()
        
    
        if bucket==11:
           # kb=df_bucket['Abs Weighted Sensi USD'].sum()
            kb=sa.K_b_equity_delta_other(df_bucket['Weighted Sensi USD'])
        else:    
            kb=sa.K_b_equity_delta(np.array(df_bucket['Weighted Sensi USD']),corr_matrix)
    
        
    
    return [df_bucket,kb,ws_b,df_bucket['FactorStructure'].tolist(),df_bucket['Risk Weight'].tolist(), df_bucket['Derived Sensi USD'].tolist(),
            df_bucket['FRTB Bucket Equity'].tolist(),
            df_bucket['Equity Sector'].tolist(), df_bucket['MarketCap USD'].tolist(), df_bucket['CountryOfRisk'].tolist(),
            df_bucket['Economy'].tolist()]
            

# Calculates the delta risk charge for given position
def calc_position_equity_delta_risk_charge(position_id,corr_high_low_med):

    # Filter the input data
    df=get_position_df(position_id,df_input_mapped)
    df=filter_for_risk_class(df,'Equity')
    df=filter_for_delta(df)
    
    if df.empty:
        
        return
    else:
    
        # Get all buckets of the position
        buckets=df['FRTB Bucket Equity'].dropna().unique().tolist()
        
        output=[calc_position_equity_delta_bucket_risk_position(position_id,bucket, corr_high_low_med) for bucket in buckets]
        
        # Bucket Risk positions, excluding 'Other' / 11
        bucket_risk_positions=np.array([item[1] for item in output])
        
        # Bucket Weigted Sensis, excluding 'Other' / 11
        bucket_weighted_sensi=[item[2] for item in output]
        factors=[item[3] for item in output]
        risk_weights=[item[4] for item in output]
        factor_sensi=[item[5] for item in output]
        factor_buckets=[item[6] for item in output]
        factor_sectors=[item[7] for item in output]    
        factor_mkt_caps=[item[8] for item in output]    
        factor_cty_of_risk=[item[9] for item in output]    
        factor_economy=[item[10] for item in output]    
    

        y_bc=sa.corr_matrix_equity_delta_ybc(buckets,corr_high_low_med)
        
        delta_risk_charge=sa.equity_delta_risk_charge(bucket_risk_positions,y_bc,np.array(bucket_weighted_sensi))  
        
        res={'position_id': position_id, 'weighted_sensi':bucket_weighted_sensi, 'K_b':bucket_risk_positions, 'delta_rc': delta_risk_charge,
             'buckets': buckets, 'factors': factors, 'RWs': risk_weights, 'Sensis': factor_sensi, 'Factor_Buckets': factor_buckets, 
             'Sectors':factor_sectors, 'Market Cap': factor_mkt_caps, 'Country': factor_cty_of_risk, 'Economy':factor_economy}
    
    #return [position_id,bucket_weighted_sensi,bucket_risk_positions,delta_risk_charge,buckets,factors, risk_weights,factor_sensi, 
    #        factor_buckets, factor_sectors, factor_mkt_caps,factor_cty_of_risk, factor_economy]
    
    return res

    


    
def calc_position_equity_curvature_bucket_risk_position(position_id,bucket,corr_high_low_med):
    
    df=get_position_df(position_id, df_input_mapped)
    
    df=filter_for_risk_class(df,'Equity')
    df=filter_for_delta(df)
    df=filter_for_curvature(df)
    
    df_bucket=df[df['FRTB Bucket Equity']==bucket]
    df_bucket=pd.merge(df_bucket,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')

    if df_bucket.empty:
        kb_med=0
        cvr_b=0
        corr_matrix_med=0
        factors=[]
        risk_weights=[]
    else:    
        df_bucket.insert(len(df_bucket.columns),'V_up',0)
        df_bucket.insert(len(df_bucket.columns),'V_down',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_up',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_down',0)
        df_bucket.insert(len(df_bucket.columns),'CVR',0)
        df_bucket.insert(len(df_bucket.columns),'V_up USD',0)
        df_bucket.insert(len(df_bucket.columns),'V_down USD',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_up USD',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_down USD',0)
        
        df_params_equities_bucket=df_params_equities[df_params_equities['FRTB Bucket Equity']==bucket]
        df_bucket=pd.merge(df_bucket, df_params_equities_bucket, on='FRTB Bucket Equity', how='left')
        df_bucket['Curvature Risk Weight']=(df_bucket['FactorCategory']=='Equity')
        df_bucket.loc[df_bucket['Curvature Risk Weight']==True,['Curvature Risk Weight']]=df_bucket['RW Spot']
        df_bucket.loc[df_bucket['Curvature Risk Weight']==False,['Curvature Risk Weight']]=df_bucket['RW Repo']
        
        
        df_bucket=df_bucket.groupby(['FRTB Bucket Equity','FactorStructure', 'FactorCategory', 'GlobalIssueID', 'LocalIssuerID Equity_Bond','Curvature Risk Weight','FXRate','SR.X','SR.Y'], as_index=False)['Derived Sensitivity'].sum()

        all_x=list(df_bucket['SR.X'].str.split(' '))
        all_x=convertlistofliststofloat(all_x)
        
        all_V_x=list(df_bucket['SR.Y'].str.split(' '))
        all_V_x=convertlistofliststofloat(all_V_x)
        
        risk_weights=df_bucket['Curvature Risk Weight']
        sensitivities=df_bucket['Derived Sensitivity']
        
        V_up=np.empty(len(all_V_x))
        V_down=np.empty(len(all_V_x))
        
        CVR_up=np.empty(len(all_V_x))
        CVR_down=np.empty(len(all_V_x))
        CVR=np.empty(len(all_V_x))
        
        for i,v in enumerate(all_x):
        
            ## To be removed/??
            if len(all_x[i])!=len(all_V_x[i]):
                CVR[i]=0
                
            else:                
                
                V=interpolate.interp1d(np.array(all_x[i]),np.array(all_V_x[i]), kind='linear', fill_value='extrapolate')
                cvr_rw=risk_weights[i]
                
                x_up=cvr_rw
                x_down=-cvr_rw
                
                V_up[i]=V(x_up)
                V_down[i]=V(x_down)
                
                CVR_up[i]=sa.CVR_up(V_up[i],cvr_rw,sensitivities[i])
                CVR_down[i]=sa.CVR_down(V_down[i],cvr_rw,sensitivities[i])
                
                #CVR_up[i]=V_up[i]-cvr_rw*sensitivities[i]
                #CVR_down[i]=V_down[i]+cvr_rw*sensitivities[i]
                CVR[i]=-min(CVR_up[i],CVR_down[i])
            
        df_bucket['V_up']=V_up
        df_bucket['V_down']=V_down
        df_bucket['CVR_up']=CVR_up
        df_bucket['CVR_down']=CVR_down
        df_bucket['CVR']=CVR

        #print(df_bucket['V_up'])
        
        df_bucket['V_up USD']=df_bucket['V_up']/df_bucket['FXRate']
        df_bucket['V_down USD']=df_bucket['V_down']/df_bucket['FXRate']        
        df_bucket['CVR_up USD']=df_bucket['CVR_up']/df_bucket['FXRate']
        df_bucket['CVR_down USD']=df_bucket['CVR_down']/df_bucket['FXRate']
        
        df_bucket['Abs CVR']=abs(df_bucket['CVR'])
        df_bucket['CVR USD']=df_bucket['CVR']/df_bucket['FXRate']
        df_bucket['Abs CVR USD']=df_bucket['Abs CVR']/df_bucket['FXRate']
        
        cvr_b=df_bucket['CVR USD'].tolist()
        factors=df_bucket['FactorStructure'].tolist()

        df_params_equities_bucket=df_params_equities[df_params_equities['FRTB Bucket Equity']==bucket]
        df_corr_sptspt_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Spot-Spot)']]
        #print(df_corr_sptspt_bucket)
        
        df_corr_reprep_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Repo-Repo)']]
        df_corr_sptrep_diffissuer_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Spot-Repo Different Issuer)']]
        df_corr_sptrep_sameissuer_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Spot-Repo Same Issuer)']]
             

        corr_matrix=sa.equity_delta_corr_matrix_bucket(df_bucket['FactorCategory'],df_bucket['LocalIssuerID Equity_Bond'],
                                            sa.getcorr(df_corr_sptspt_bucket.iloc[0][1],corr_high_low_med), 
                                            sa.getcorr(df_corr_reprep_bucket.iloc[0,1],corr_high_low_med),
                                            sa.getcorr(df_corr_sptrep_sameissuer_bucket.iloc[0,1],corr_high_low_med),
                                            sa.getcorr(df_corr_sptrep_diffissuer_bucket.iloc[0,1],corr_high_low_med))
    
        corr_matrix_2=corr_matrix**2
    
        k_b_up=sa.K_b_equity_curvature_up(np.array(df_bucket['CVR_up USD']),corr_matrix_2)
        k_b_down=sa.K_b_equity_curvature_down(np.array(df_bucket['CVR_down USD']),corr_matrix_2)
    
        if bucket==11:
            #kb=df_bucket['Abs CVR USD'].sum()        
            K_b=sa.K_b_equity_curvature_other(df_bucket['CVR_up USD'],df_bucket['CVR_down USD'])    
        else:        
            K_b=max(k_b_up,k_b_down)
        if K_b==k_b_up:
            S_b=sa.S_b_curvature(df_bucket['CVR_up USD'],df_bucket['CVR_down USD'],K_b,k_b_up,k_b_down)
        elif K_b==k_b_down:
            S_b=sa.S_b_curvature(df_bucket['CVR_up USD'],df_bucket['CVR_down USD'],K_b,k_b_up,k_b_down)
        else:
            S_b=0
        
    
    
    return {'buckets':df_bucket, 'K_b':K_b, 'CVR_b':cvr_b, 'factors':factors, 'Risk Weights': risk_weights, 'V_up':df_bucket['V_up USD'].tolist(), 
            'V_down':df_bucket['V_down USD'].tolist(), 'CVR_up': df_bucket['CVR_up USD'].tolist(),
            'CVR_down': df_bucket['CVR_down USD'].tolist(),'bucket': df_bucket['FRTB Bucket Equity'].tolist(),'S_b':S_b}
    
    
    #return [df_bucket, K_b, cvr_b, factors, risk_weights, df_bucket['V_up USD'].tolist(), df_bucket['V_down USD'].tolist(),df_bucket['CVR_up USD'].tolist(),
    #            df_bucket['CVR_down USD'].tolist(),df_bucket['FRTB Bucket Equity'].tolist(),S_b]
            


# Calculates the EQ delta risk charge for given position and corr scenario
def calc_position_equity_curvature_risk_charge(position_id,corr_high_low_med):

    # Filter the input data
    df=get_position_df(position_id,df_input_mapped)
    df=filter_for_risk_class(df,'Equity')
    df=filter_for_delta(df)
    df=filter_for_curvature(df)
    
    if df.empty:
        return
    else:
    
        # Get all buckets of the position
        buckets=df['FRTB Bucket Equity'].dropna().unique().tolist()
        n_buckets=len(buckets)
           
        
        output=[calc_position_equity_curvature_bucket_risk_position(position_id,bucket,corr_high_low_med) for bucket in buckets]        
        # Bucket Risk positions, excluding 'Other' / 11
        
        bucket_risk_positions=np.array([item[1] for item in output])
        # Bucket Weigted Sensis, excluding 'Other' / 11
        bucket_curvature_sensi=[item[2] for item in output] # factor level CVR_k???
        factors=[item[3] for item in output]
        risk_weights=[item[4] for item in output]
        factors_V_up=[item[5] for item in output]
        factors_V_down=[item[6] for item in output]        
        factors_CVR_up=[item[7] for item in output]
        factors_CVR_down=[item[8] for item in output]
        factors_bucket=[item[9] for item in output]
        S_b=[item[10] for item in output]
    
   
        y_bc=sa.corr_matrix_equity_ybc(buckets,corr_high_low_med)
        ##########
        #curvature_risk_charge=sa.curvature_risk_charge(bucket_risk_positions,y_bc,np.array(bucket_curvature_sensi))  
        curvature_risk_charge=sa.curvature_risk_charge(bucket_risk_positions,y_bc,S_b)    
        #################
    
    res={'position_id': position_id, 'CVR_k': bucket_curvature_sensi, 'K_b':bucket_risk_positions, 'cvr_rc': curvature_risk_charge,
             'buckets': buckets, 'factors': factors, 'RWs': risk_weights, 'V_up': factors_V_up, 'V_down': factors_V_down,
             'CVR_up': factors_CVR_up, 'CVR_down': factors_CVR_down,
             'Factor_Buckets': factors_bucket}
    
    #return [position_id,bucket_curvature_sensi,bucket_risk_positions,curvature_risk_charge,buckets,factors,risk_weights,
    #        factors_V_up,factors_V_down,factors_CVR_up,factors_CVR_down, factors_bucket]
    
    return res



def calc_position_equity_vega_bucket_risk_position(position_id,bucket,corr_high_low_med):
    
    frtb_vertices=['0.5y','1y','3y','5y','10y']
    frtb_vertices_days=[datestring(v).getdays365() for v in frtb_vertices]
    frtb_vertices_years=[datestring(v).getyears365() for v in frtb_vertices]
    
    df=get_position_df(position_id, df_input_mapped)
    df=filter_for_risk_class(df,'Equity')
    df=filter_for_vega(df)
    
    df_bucket=df[df['FRTB Bucket Equity']==bucket]
    df_bucket=pd.merge(df_bucket,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')

    #fg1=np.array(df_bucket['Grid'])
    #factorgrid1_years=[(datestring(fg).getyears365()) for fg in fg1]

    if df_bucket.empty:
        return
    else:    

        df1=df_bucket.copy()
        df2=df_bucket.copy()
        
        # Get FG1 and convert to days365
        factorgrid1=np.array(df_bucket['FactorGrid1'])    
        factorgrid1_days=[(datestring(fg).getdays365()) for fg in factorgrid1]
        factorgrid1_years=[(datestring(fg).getyears365()) for fg in factorgrid1]

        # FRTB SA Buckets in days
        vertices_days=frtb_vertices_days
        
        # Get nearest points Tenor1/Tenor2 in days for each given FG1
        tenor1_tenor2=[get_boundary_points(vertices_days,d) for d in factorgrid1_days]
        t1=[item[0] for item in tenor1_tenor2]
        t2=[item[1] for item in tenor1_tenor2]
        
        t1_t2_years=[get_boundary_points(frtb_vertices_years,y) for y in factorgrid1_years]
        t1_years=[item[0] for item in t1_t2_years]
        t2_years=[item[1] for item in t1_t2_years]
       
        # Weights to be assigned to sensitivity when assigning to tenor1/tenor2 
        tenor_weights=np.array([assign_weights(factorgrid1_days[i],tenor1_tenor2[i][0],tenor1_tenor2[i][1]) for i,v in enumerate(factorgrid1_days)])
        
        # Assign input sensitivities to nearest points t1/t2
        sensi_t1=tenor_weights[:,0]*df_bucket['Derived Sensitivity']
        sensi_t2=tenor_weights[:,1]*df_bucket['Derived Sensitivity']    
    
        risk_weights_t1=np.minimum(60**(1/2)*sa.rw_sigma/(math.sqrt(10)),np.ones(len(df_bucket['Derived Sensitivity'])))
        risk_weights_t2=np.minimum(60**(1/2)*sa.rw_sigma/(math.sqrt(10)),np.ones(len(df_bucket['Derived Sensitivity'])))
        
        #print(factorgrid1,factorgrid1_days,tenor1_tenor2, tenor_weights, sensi_t1, sensi_t2,risk_weights_t1,risk_weights_t2)
        
        # Weighted sensi for Tenor1/Tenor2 factors
        weighted_sensi_t1=sensi_t1*np.array(risk_weights_t1)
        weighted_sensi_t2=sensi_t2*np.array(risk_weights_t2)
     
        ######################################
        # New dataframes to handle Tenor1/Tenor2
        df1['t']=pd.Series(t1, index=df_bucket.index)
        df1['t_years']=pd.Series(t1_years, index=df_bucket.index)
        df1['Sensi_t']=pd.Series(sensi_t1, index=df_bucket.index)
        df1['Risk Weight_t']=pd.Series(risk_weights_t1, index=df_bucket.index)
        df1['Weighted Sensi']=pd.Series(weighted_sensi_t1, index=df_bucket.index)
        
        df2['t']=pd.Series(t2, index=df_bucket.index)
        df2['t_years']=pd.Series(t2_years, index=df_bucket.index)        
        df2['Sensi_t']=pd.Series(sensi_t2, index=df_bucket.index)
        df2['Risk Weight_t']=pd.Series(risk_weights_t2, index=df_bucket.index)
        df2['Weighted Sensi']=pd.Series(weighted_sensi_t2, index=df_bucket.index)
        ##############################
        
        ####################################
        # Dataframe to combine df1/df2 before calculation done
        df3=df1.copy()
        df3=df3.append(df2, ignore_index=True)
        df3=df3.groupby(['FactorStructure', 'FactorCategory', 'LocalIssuerID', 'Risk Weight_t','t', 'FXRate','t_years'], as_index=False)['Sensi_t','Weighted Sensi'].sum()
        df3=df3[df3['t']!=0]
        df3['Weighted Sensi USD']=df3['Weighted Sensi']/df3['FXRate']
        df3['Sensi USD']=df3['Sensi_t']/df3['FXRate']
        ###############################################################################################
        ###############################################################################################
    
        df_params_equities_bucket=df_params_equities[df_params_equities['FRTB Bucket Equity']==bucket]
    
        df_corr_sptspt_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Spot-Spot)']]
        df_corr_reprep_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Repo-Repo)']]
        df_corr_sptrep_diffissuer_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Spot-Repo Different Issuer)']]
        df_corr_sptrep_sameissuer_bucket=df_params_equities_bucket[['FRTB Bucket Equity','Rho (Spot-Repo Same Issuer)']]
             
        corr_matrix_delta=sa.equity_delta_corr_matrix_bucket(df_bucket['FactorCategory'],df3['LocalIssuerID'],
                                            sa.getcorr(df_corr_sptspt_bucket.iloc[0][1],corr_high_low_med), 
                                            sa.getcorr(df_corr_reprep_bucket.iloc[0,1],corr_high_low_med),
                                            sa.getcorr(df_corr_sptrep_sameissuer_bucket.iloc[0,1],corr_high_low_med),
                                            sa.getcorr(df_corr_sptrep_diffissuer_bucket.iloc[0,1],corr_high_low_med))
    
    
        maturity_corr_matrix=sa.corr_matrix_maturity(factorgrid1_years, corr_high_low_med)
        corr_matrix_vega=sa.corr_matrix_vega(corr_matrix_delta,maturity_corr_matrix)
        
        if bucket==11:
            kb=sa.K_b_equity_vega_other(df3['Weighted Sensi USD'])
        else:    
            kb=sa.K_b_equity_vega(np.array(df3['Weighted Sensi USD']),corr_matrix_vega)
            #kb=sa.K_b_equity_vega(np.array(df_bucket['Weighted Sensi USD']),corr_matrix_vega)
    
    return [df_bucket,kb,df3['Weighted Sensi USD'].tolist(),df3['Factor Structure'].tolist(), df_bucket['Risk Weight'].tolist(),df3['Sensi USD'].tolist(),
            df3['Grid'].tolist(),df_bucket['FRTB Bucket Equity'].tolist()]
    


# Calculates the vega risk charge for given position
def calc_position_equity_vega_risk_charge(position_id,corr_high_low_med):
    df=get_position_df(position_id,df_input_mapped)
    df=filter_for_risk_class(df,'Equity')
    df=filter_for_vega(df)   
    
    if df.empty:       
        return
    else:
        buckets=df['FRTB Bucket Equity'].dropna().unique().tolist()
          
        
        output=[calc_position_equity_vega_bucket_risk_position(position_id,bucket,corr_high_low_med) for bucket in buckets]
        
        bucket_risk_positions=np.array([item[1] for item in output])
        bucket_weighted_sensi=[item[2] for item in output]
        factors=[item[3] for item in output]
        risk_weights=[item[4] for item in output]
        factor_sensis=[item[5] for item in output]
        factor_tenors=[item[6] for item in output]
        factor_buckets=[item[7] for item in output]
        

        y_bc=sa.corr_matrix_equity_vega_ybc(buckets,corr_high_low_med)
        vega_risk_charge=sa.equity_vega_risk_charge(bucket_risk_positions,y_bc,np.array(bucket_weighted_sensi))  

    
    res={'position_id': position_id, 'weighted_sensi':bucket_weighted_sensi, 'K_b':bucket_risk_positions, 'vega_rc': vega_risk_charge,
             'buckets': buckets, 'factors': factors, 'RWs': risk_weights, 'Sensis': factor_sensis, 'Factor_Buckets': factor_buckets, 
             'Tenors':factor_tenors}
    
    
    return res
    
    



    
# For a given position / bucket, calculate the risk position Kb
def calc_position_girr_delta_bucket_risk_position(position_id,bucket,corr_high_low_med):
        
    frtb_vertices=['0.25y','0.5y','1y','2y','3y','5y','10y','15y','20y','30y']
    frtb_vertices_days=[datestring(v).getdays365() for v in frtb_vertices]
    
    #print(frtb_vertices_days)
    
    df=get_position_df(position_id,df_input_mapped)
    df=filter_for_risk_class(df,'GIRR')
    df=filter_for_delta(df)
    df=filter_girr_bucket(df,bucket)
    
    df=pd.merge(df,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')
    
    if df.empty:
        df1=0
        df2=0
        df3=0
        kb_med=0
        ws_b=0
        corr_matrix_girr=0
        factors=0
        factor_grids=0
        s_b=0
        rw=0
    else:
        df1=df.copy()
        df2=df.copy()
        
        # Get FG1 and convert to days365
        factorgrid1=np.array(df['FactorGrid1'])    
        factorgrid1_days=[(datestring(fg).getdays365()) for fg in factorgrid1]
    
    
        # Factor categories, needed to get the risk weight
        factor_categories=np.array(df['FactorCategory'])
    
        # Get the regulatory curve vertices in days
        calc=girrcalc()                           
        # FRTB SA Buckets in days
        vertices_days=frtb_vertices_days
        
        # Get nearest points Tenor1/Tenor2 in days for each given FG1
        # t1/t2 should be the frtb grids in days
        tenor1_tenor2=[get_boundary_points(vertices_days,d) for d in factorgrid1_days]
        t1=[item[0] for item in tenor1_tenor2]
        t2=[item[1] for item in tenor1_tenor2]
        
        # Weights to be assigned to sensitivity when assigning to tenor1/tenor2 
        tenor_weights=np.array([assign_weights(factorgrid1_days[i],tenor1_tenor2[i][0],tenor1_tenor2[i][1]) for i,v in enumerate(factorgrid1_days)])
        
        # Assign input sensitivities to nearest points t1/t2
        sensi_t1=tenor_weights[:,0]*df['Derived Sensitivity']
        sensi_t2=tenor_weights[:,1]*df['Derived Sensitivity']    
    
        # Risk weights for Tenor1/Tenor2, as per regulation
        risk_weights_t1=np.array([calc.getriskweight(factor_categories[index],item) for index,item in enumerate(t1)])
        risk_weights_t2=np.array([calc.getriskweight(factor_categories[index],item) for index,item in enumerate(t2)])
    
        #print(t1,t2,risk_weights_t1,risk_weights_t2)
    
        # Weighted sensi for Tenor1/Tenor2 factors
        weighted_sensi_t1=sensi_t1*np.array(risk_weights_t1)
        weighted_sensi_t2=sensi_t2*np.array(risk_weights_t2)
     
        ######################################
        # New dataframes to handle Tenor1/Tenor2
        
        df1['t']=pd.Series(t1, index=df.index)
        df1['Sensi_t']=pd.Series(sensi_t1, index=df.index)
        df1['Risk Weight_t']=pd.Series(risk_weights_t1, index=df.index)
        df1['Weighted Sensi']=pd.Series(weighted_sensi_t1, index=df.index)
        
        df2['t']=pd.Series(t2, index=df.index)
        df2['Sensi_t']=pd.Series(sensi_t2, index=df.index)
        df2['Risk Weight_t']=pd.Series(risk_weights_t2, index=df.index)
        df2['Weighted Sensi']=pd.Series(weighted_sensi_t2, index=df.index)
        ##############################
        
        ####################################
        # Dataframe to combine df1/df2 before calculation done
        df3=df1.copy()
        df3=df3.append(df2, ignore_index=True)
        df3=df3.groupby(['FactorStructure', 'FactorCategory', 'Risk Weight_t','CCY1' ,'t', 'FXRate'], as_index=False)['Sensi_t','Weighted Sensi'].sum()
        df3=df3[df3['t']!=0]
        ###########################################
        
        df3['Weighted Sensi USD']=df3['Weighted Sensi']/df3['FXRate']
        df3['Sensi USD']=df3['Sensi_t']/df3['FXRate']
        
        #df3['Abs Weighted Sensi USD']=df3['Abs Weighted Sensi USD']/df3['FXRate']
        
        # Corr matrix to be used for factors in the same bucket (currency)
        corr_matrix_girr=sa.corr_matrix_girr(df3.FactorStructure,df3.FactorCategory, df3.t,corr_high_low_med)
        k_b=sa.K_b_girr_delta(df3['Weighted Sensi USD'],corr_matrix_girr)    
        rw=df3['Risk Weight_t'].tolist()
        s_b=df3['Sensi USD'].tolist()
        
        ws_b=df3['Weighted Sensi USD'].tolist()
        factors=df3['FactorStructure'].tolist()
        factor_grids=df3['t'].tolist()
        #print(kb_med)
        

    #return [df1,df2,df3,kb_med,ws_b,corr_matrix_girr]
    return [df1,df2,df3,k_b,ws_b,factors, factor_grids, s_b,rw,df]

# Calculates the delta risk charge for given position
def calc_position_girr_delta_risk_charge(position_id,corr_high_low_med):
    
    df=get_position_df(position_id,df_input_mapped)
    df=filter_for_risk_class(df,'GIRR')
    df=filter_for_delta(df)
    
    if df.empty:
        #return
        bucket_weighted_sensi=[]
        bucket_sensi=[]
        risk_weights=[]
        bucket_risk_positions=0
        delta_risk_charge=0
        buckets=[]
        factors=[]
        factor_grids=[]
    else:
    
        # Get all buckets of the position
        buckets=df['Bucket GIRR'].unique().tolist()
        n_buckets=len(buckets)
        output=[calc_position_girr_delta_bucket_risk_position(position_id,bucket,corr_high_low_med) for bucket in buckets]
        
        # Bucket Risk positions
        bucket_risk_positions=np.array([item[3] for item in output])
        #print(bucket_risk_positions,bucket_risk_positions_2)
        
        # Correlation matrix across buckets
        y_bc=0.5*np.matlib.identity(n_buckets)
        
        # Stressed Corr matrix
        y_bc=sa.get_stress_corr_matrix(0.5*np.matlib.identity(n_buckets),corr_high_low_med)
               
        bucket_weighted_sensi=[item[4] for item in output]
        #print(bucket_weighted_sensi,bucket_weighted_sensi_2)
        
        bucket_sensi=[item[7] for item in output]
        #print(bucket_sensi,bucket_sensi_2)
        
        risk_weights=[item[8] for item in output]
        factors=[item[5] for item in output]
        factor_grids=[item[6] for item in output]
        
#        delta_risk_charge=sa.delta_vega_risk_charge(bucket_risk_positions,y_bc,np.array(bucket_weighted_sensi))
        delta_risk_charge=sa.girr_delta_risk_charge(bucket_risk_positions,y_bc,np.array(bucket_weighted_sensi))
    
    return [position_id,bucket_weighted_sensi,bucket_risk_positions,delta_risk_charge,buckets,factors, factor_grids, bucket_sensi, risk_weights]


# For a given position / bucket, calculate the risk position Kb
def calc_position_girr_curvature_bucket_risk_position(position_id,bucket,corr_high_low_med):
    
    # Filter the input data
    df=get_position_df(position_id,df_input_mapped)
    df=filter_for_risk_class(df,'GIRR')
    df=filter_for_delta(df)
    #df=filter_for_curvature(df)
    df_bucket=filter_girr_bucket(df,bucket)
    # Bring in FX Rates
    df_bucket=pd.merge(df_bucket,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')
    # Only want parallels (for now)
    df_bucket=df_bucket[df_bucket['Grid']=='P']
    df_bucket=df_bucket[df_bucket['Curvature']==True]    

    #
    if df_bucket.empty:
        kb_med=0
        factors=[]
    else:
    
        df_bucket.insert(len(df_bucket.columns),'Curvature Risk Weight',0)
        df_bucket.loc[df_bucket['FactorCategory']!='Currency_Basis',['Curvature Risk Weight']]=0.017
        df_bucket=df_bucket.groupby(['Bucket GIRR','FactorStructure', 'FactorCategory', 'Curvature Risk Weight','FXRate','SR.X','SR.Y'], as_index=False)['Derived Sensitivity'].sum()
        df_bucket.insert(len(df_bucket.columns),'V_up',0)
        df_bucket.insert(len(df_bucket.columns),'V_down',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_up',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_down',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_up USD',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_down USD',0)
        df_bucket.insert(len(df_bucket.columns),'CVR',0)
        
        all_x=list(df_bucket['SR.X'].str.split(' '))
        all_x=convertlistofliststofloat(all_x)
        
        all_V_x=list(df_bucket['SR.Y'].str.split(' '))
        all_V_x=convertlistofliststofloat(all_V_x)
        
        risk_weights=df_bucket['Curvature Risk Weight']
        sensitivities=df_bucket['Derived Sensitivity']
        
        V_up=np.empty(len(all_V_x))
        V_down=np.empty(len(all_V_x))
        
        CVR_up=np.empty(len(all_V_x))
        CVR_down=np.empty(len(all_V_x))    
        CVR=np.empty(len(all_V_x))
    
        for i,v in enumerate(all_x):
        
            ## To be removed/?? When Shock Record arrays have different lengths
            if len(all_x[i])!=len(all_V_x[i]):
                CVR=0            
            else:                            
                V=interpolate.interp1d(np.array(all_x[i]),np.array(all_V_x[i]), kind='linear', fill_value='extrapolate')
                cvr_rw=risk_weights[i]
                
                x_up=cvr_rw
                x_down=-cvr_rw
                
                V_up[i]=V(x_up)
                V_down[i]=V(x_down)
                
                #CVR_up[i]=V_up[i]-cvr_rw*sensitivities[i]
                #CVR_down[i]=V_down[i]+cvr_rw*sensitivities[i]
               
                CVR_up[i]=sa.CVR_up(V_up[i],cvr_rw,sensitivities[i])
                CVR_down[i]=sa.CVR_down(V_down[i],cvr_rw,sensitivities[i])
                
                CVR=-min(CVR_up[i],CVR_down[i])
                            
        df_bucket['V_up']=V_up
        df_bucket['V_down']=V_down
        df_bucket['CVR_up']=CVR_up
        df_bucket['CVR_down']=CVR_down
        df_bucket['CVR']=CVR

        df_bucket['V_up USD']=df_bucket['V_up']/df_bucket['FXRate']
        df_bucket['V_down USD']=df_bucket['V_down']/df_bucket['FXRate']        
        df_bucket['CVR_up USD']=df_bucket['CVR_up']/df_bucket['FXRate']
        df_bucket['CVR_down USD']=df_bucket['CVR_down']/df_bucket['FXRate']

        CVR_up_total=df_bucket['CVR_up USD'].sum()
        CVR_down_total=df_bucket['CVR_down USD'].sum()
        CVR_total=[-min(CVR_up_total,CVR_down_total)]

        #k_b_up=sa.K_b_girr_curvature_up(CVR_up,)

#        CVR_USD=-min(CVR_up,CVR_down)
        #factors=df_bucket['FactorStructure']
        k_b=CVR_total
    
    return [df_bucket,k_b,CVR_total,CVR_up_total,CVR_down_total,df_bucket['FactorStructure'].tolist(),  df_bucket['V_up USD'].tolist(), df_bucket['V_down USD'].tolist(),
            df_bucket['CVR_up USD'].tolist(),df_bucket['CVR_down USD'].tolist()]


# Calculates the delta risk charge for given position

def calc_position_girr_curvature_risk_charge(position_id,corr_high_low_med):
    
    # Filter the input data
    df=get_position_df(position_id,df_input_mapped)
    #print(df.Grid) 
    df=filter_for_risk_class(df,'GIRR')
    df=filter_for_delta(df)
    df=df[df['Grid']=='P']
    df=df[df['Curvature']==True]    
    
    if df.empty:
        
        bucket_curvature_sensi=[]
        bucket_risk_positions=0
        #out=0
        curvature_risk_charge=0
        buckets=[]
        factors=[]
        factors_V_up=[]
        factors_V_down=[]
        factors_CVR_up=[]
        factors_CVR_down=[]

        #factor_grids=[]
    else:
    
        # Get all buckets of the position
        
        
        buckets=df['Bucket GIRR'].unique().tolist()
        n_buckets=len(buckets)
        #print(buckets)

        output=[calc_position_girr_curvature_bucket_risk_position(position_id,bucket,corr_high_low_med) for bucket in buckets]

        # Bucket Risk positions
        
        bucket_risk_positions=np.array([item[1] for item in output])
        #print(bucket_risk_positions)
        
        
        #bucket_curvature_sensi=[calc_position_girr_curvature_bucket_risk_position(position_id,bucket)[2] for bucket in buckets]
        bucket_curvature_sensi=[item[2] for item in output]        
        bucket_curvature_up=[item[3] for item in output]
        bucket_curvature_down=[item[4] for item in output]

        #print(bucket_curvature_sensi)
        
        #factors=[calc_position_girr_delta_bucket_risk_position(position_id,bucket)[5] for bucket in buckets]
        factors=[item[5] for item in output]
        factors_V_up=[item[6] for item in output]
        factors_V_down=[item[7] for item in output]        
        factors_CVR_up=[item[8] for item in output]
        factors_CVR_down=[item[9] for item in output]

        # Correlation matrix across buckets
        y_bc=sa.corr_matrix_bucket(sa.getcorr(0.5*0.5,corr_high_low_med), n_buckets)
        

        curvature_risk_charge=sa.delta_vega_risk_charge(bucket_risk_positions,y_bc,np.array(bucket_curvature_sensi))
        
    
    return [position_id,bucket_curvature_sensi,bucket_risk_positions,curvature_risk_charge,buckets,factors,
            factors_V_up,factors_V_down,factors_CVR_up,factors_CVR_down,df]




def calc_position_girr_vega_bucket_risk_position(position_id,bucket,corr_high_low_med):
    
    df=get_position_df(position_id, df_input_mapped) 
    df=filter_for_risk_class(df,'GIRR')
    df=filter_for_vega(df)


    df_bucket=df[df['Bucket GIRR']==bucket]
    df_bucket=pd.merge(df_bucket,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')

    df_bucket['Risk Weight']=(df_bucket['Measure']=='Vega')    
    df_bucket['Risk Weight']=np.minimum(60**(1/2)*sa.rw_sigma/(math.sqrt(10)),np.ones(len(df_bucket['Risk Weight'])))

    #calc=girrcalc()                           
   
    if df_bucket.empty:
        return
    else:    
        df1=df_bucket.copy()
        df2=df_bucket.copy()
        df3=df_bucket.copy()
        df4=df_bucket.copy()
                
        # Get FG1 and convert to days365
        factorgrid1=np.array(df_bucket['FactorGrid1'])    
        factorgrid1_days=[(datestring(fg).getdays365()) for fg in factorgrid1]
        factorgrid1_years=[(datestring(fg).getyears365()) for fg in factorgrid1]
        
        # Unlerlying maturity in days/years
        factorgrid2=np.array(df_bucket['FactorGrid2'])
        factorgrid2_days=[(datestring(fg).getdays365()) for fg in factorgrid2]
        factorgrid2_years=[(datestring(fg).getyears365()) for fg in factorgrid2]
        
        frtb_vertices=['0.5y','1y','3y','5y','10y']    
        frtb_vertices_days=[datestring(v).getdays365() for v in frtb_vertices]
        frtb_vertices_years=[datestring(v).getyears365() for v in frtb_vertices]
        
        fg1_above_max=[above_list_maximum(frtb_vertices_days,fg1_day) for fg1_day in factorgrid1_days]
        fg1_below_min=[below_list_minimum(frtb_vertices_days,fg1_day) for fg1_day in factorgrid1_days]
        
        fg2_above_max=[above_list_maximum(frtb_vertices_days,fg2_day) for fg2_day in factorgrid2_days]
        fg2_below_min=[below_list_minimum(frtb_vertices_days,fg2_day) for fg2_day in factorgrid2_days]
    
        for i,v in enumerate(factorgrid1_days):
            if fg1_above_max[i]==True:
                factorgrid1_days[i]=max(frtb_vertices_days)
                factorgrid1_years[i]=max(frtb_vertices_years)
            else:
                factorgrid1_days[i]=factorgrid1_days[i]
                factorgrid1_years[i]=factorgrid1_years[i]
        for i,v in enumerate(factorgrid1_days):
            if fg1_below_min[i]==True:
                factorgrid1_days[i]=min(frtb_vertices_days)
                factorgrid1_years[i]=min(frtb_vertices_years)
            else:
                factorgrid1_days[i]=factorgrid1_days[i]
                factorgrid1_years[i]=factorgrid1_years[i]
        for i,v in enumerate(factorgrid2_days):
            if fg2_above_max[i]==True:
                factorgrid2_days[i]=max(frtb_vertices_days)
                factorgrid2_years[i]=max(frtb_vertices_years)
            else:
                factorgrid2_days[i]=factorgrid2_days[i]
                factorgrid2_years[i]=factorgrid2_years[i]
        for i,v in enumerate(factorgrid2_days):
            if fg2_below_min[i]==True:
                factorgrid2_days[i]=min(frtb_vertices_days)
                factorgrid2_years[i]=min(frtb_vertices_years)
            else:
                factorgrid2_days[i]=factorgrid2_days[i]
                factorgrid2_years[i]=factorgrid2_years[i]
        
        
        # Get nearest points Tenor1/Tenor2 in days for FG1
        #fg1_tenor1_tenor2=[get_bound_points(frtb_vertices_days,d) for d in factorgrid1_days]
        fg1_t1_t2=[get_boundary_points(frtb_vertices_days,d) for d in factorgrid1_days]
        fg1_t1=[item[0] for item in fg1_t1_t2]
        fg1_t2=[item[1] for item in fg1_t1_t2]
        
        
        fg1_t1_t2_years=[get_boundary_points(frtb_vertices_years,y) for y in factorgrid1_years]
        fg1_t1_years=[item[0] for item in fg1_t1_t2_years]
        fg1_t2_years=[item[1] for item in fg1_t1_t2_years]
        
        # Get nearest points Tenor1/Tenor2 in days for FG2
        #fg2_tenor1_tenor2=[get_bound_points(frtb_vertices_days,d) for d in factorgrid2_days]
        fg2_t1_t2=[get_boundary_points(frtb_vertices_days,d) for d in factorgrid2_days]
        fg2_t1=[item[0] for item in fg2_t1_t2]
        fg2_t2=[item[1] for item in fg2_t1_t2]
                
        fg2_t1_t2_years=[get_boundary_points(frtb_vertices_years,y) for y in factorgrid2_years]
        fg2_t1_years=[item[0] for item in fg2_t1_t2_years]
        fg2_t2_years=[item[1] for item in fg2_t1_t2_years]
        
       # print(fg1_t1,fg1_t2,fg2_t1,fg2_t2)
        #[factorgrid1_days for item in min(frtb_vertices_days) if 
        
        #[w_x1_y1,w_x1_y2,w_x2_y1,w_x2_y2]
        
        tenor_weights=np.array([assign_weights_2d(value1,value2,fg1_t1[i],fg1_t2[i],fg2_t1[i],fg2_t2[i]) for i, (value1, value2) in enumerate(zip(factorgrid1_days, factorgrid2_days))])
        
        tenor_weights_fg1_t1_fg2_t1=tenor_weights[:,0]
        tenor_weights_fg1_t1_fg2_t2=tenor_weights[:,1]
        tenor_weights_fg1_t2_fg2_t1=tenor_weights[:,2]
        tenor_weights_fg1_t2_fg2_t2=tenor_weights[:,3]
            
        # Assign input sensitivities to nearest points t1/t2
        sensi_fg1_t1_fg2_t1=tenor_weights_fg1_t1_fg2_t1*df_bucket['Derived Sensitivity']
        sensi_fg1_t2_fg2_t1=tenor_weights_fg1_t2_fg2_t1*df_bucket['Derived Sensitivity']
        sensi_fg1_t1_fg2_t2=tenor_weights_fg1_t1_fg2_t2*df_bucket['Derived Sensitivity']
        sensi_fg1_t2_fg2_t2=tenor_weights_fg1_t2_fg2_t2*df_bucket['Derived Sensitivity']
        
        # Weighted sensi for Tenor1/Tenor2 factors
        weighted_sensi_fg1_t1_fg2_t1=sensi_fg1_t1_fg2_t1*np.array(df_bucket['Risk Weight'])
        weighted_sensi_fg1_t2_fg2_t1=sensi_fg1_t2_fg2_t1*np.array(df_bucket['Risk Weight'])
        weighted_sensi_fg1_t1_fg2_t2=sensi_fg1_t1_fg2_t2*np.array(df_bucket['Risk Weight'])
        weighted_sensi_fg1_t2_fg2_t2=sensi_fg1_t2_fg2_t2*np.array(df_bucket['Risk Weight'])
        
        
        #print(weighted_sensi_fg1_t1,weighted_sensi_fg1_t2,weighted_sensi_fg2_t1,weighted_sensi_fg2_t2)
        
        ######################################
        # New dataframes to handle Tenor1/Tenor2
        
        df1['fg1_t']=pd.Series(fg1_t1, index=df_bucket.index)
        df1['fg2_t']=pd.Series(fg2_t1, index=df_bucket.index)
        df1['fg1_y']=pd.Series(fg1_t1_years, index=df_bucket.index)
        df1['fg2_y']=pd.Series(fg2_t1_years, index=df_bucket.index)
        df1['Sensi']=pd.Series(sensi_fg1_t1_fg2_t1, index=df_bucket.index)
        df1['Tenor Weight']=pd.Series(tenor_weights_fg1_t1_fg2_t1, index=df_bucket.index)
        df1['Weighted Sensi']=pd.Series(weighted_sensi_fg1_t1_fg2_t1, index=df_bucket.index)
        
        df2['fg1_t']=pd.Series(fg1_t2, index=df_bucket.index)
        df2['fg2_t']=pd.Series(fg2_t1, index=df_bucket.index)
        df2['fg1_y']=pd.Series(fg1_t2_years, index=df_bucket.index)
        df2['fg2_y']=pd.Series(fg2_t1_years, index=df_bucket.index)
        df2['Sensi']=pd.Series(sensi_fg1_t2_fg2_t1, index=df_bucket.index)
        df2['Tenor Weight']=pd.Series(tenor_weights_fg1_t2_fg2_t1, index=df_bucket.index)
        df2['Weighted Sensi']=pd.Series(weighted_sensi_fg1_t2_fg2_t1, index=df_bucket.index)
        
        df3['fg1_t']=pd.Series(fg1_t1, index=df_bucket.index)
        df3['fg2_t']=pd.Series(fg2_t2, index=df_bucket.index)
        df3['fg1_y']=pd.Series(fg1_t1_years, index=df_bucket.index)
        df3['fg2_y']=pd.Series(fg2_t2_years, index=df_bucket.index)
        df3['Sensi']=pd.Series(sensi_fg1_t1_fg2_t2, index=df_bucket.index)
        df3['Tenor Weight']=pd.Series(tenor_weights_fg1_t1_fg2_t2, index=df_bucket.index)
        df3['Weighted Sensi']=pd.Series(weighted_sensi_fg1_t1_fg2_t2, index=df_bucket.index)
        
        df4['fg1_t']=pd.Series(fg1_t2, index=df_bucket.index)
        df4['fg2_t']=pd.Series(fg2_t2, index=df_bucket.index)
        df4['fg1_y']=pd.Series(fg1_t2_years, index=df_bucket.index)
        df4['fg2_y']=pd.Series(fg2_t2_years, index=df_bucket.index)
        df4['Sensi']=pd.Series(sensi_fg1_t2_fg2_t2, index=df_bucket.index)
        df4['Tenor Weight']=pd.Series(tenor_weights_fg1_t2_fg2_t2, index=df_bucket.index)
        df4['Weighted Sensi']=pd.Series(weighted_sensi_fg1_t2_fg2_t2, index=df_bucket.index)
        ##############################
    
        ####################################
        # Dataframe to combine df1/df2 before calculation done
        df_calc=df1.copy()
        df_calc=df_calc.append(df2, ignore_index=True)
        df_calc=df_calc.append(df3, ignore_index=True)
        df_calc=df_calc.append(df4, ignore_index=True)
        df_calc=df_calc.groupby(['FactorStructure', 'FactorCategory', 'Tenor Weight', 'Risk Weight','CCY1','fg1_t','fg2_t','fg1_y', 'fg2_y', 'FXRate'], as_index=False)['Sensi','Weighted Sensi'].sum()
        df_calc['Weighted Sensi USD']=df_calc['Weighted Sensi']/df_calc['FXRate']
        df_calc['Sensi USD']=df_calc['Sensi']/df_calc['FXRate']
        
        
        risk_weights=df_calc['Risk Weight'].tolist()
        ws_b=df_calc['Weighted Sensi USD'].tolist()
        sensi_b=df_calc['Sensi USD'].tolist()
        factors=df_calc['FactorStructure'].tolist()   
 
        ul_maturity_corr_matrix=sa.corr_matrix_maturity(df_calc['fg1_y'],corr_high_low_med)
        opt_maturity_corr_matrix=sa.corr_matrix_maturity(df_calc['fg2_y'],corr_high_low_med)
        corr_matrix_vega_girr=sa.corr_matrix_vega(opt_maturity_corr_matrix,ul_maturity_corr_matrix)

        k_b=sa.K_b_girr_vega(df_calc['Weighted Sensi USD'],corr_matrix_vega_girr)
         
    #return [df_bucket,kb_med,ws_b,factors,factorgrid1_years,factorgrid2_years]
    #return [df_calc, k_b,ws_b,factors,sensi_b,risk_weights]
    return [df_calc, k_b,ws_b,factors,df_calc['fg1_y'].tolist(),df_calc['fg2_y'].tolist(),sensi_b,risk_weights]




# Calculates the delta risk charge for given position
def calc_position_girr_vega_risk_charge(position_id,corr_high_low_med):

    # Filter the input data
    df=get_position_df(position_id,df_input_mapped)
    df=filter_for_risk_class(df,'GIRR')
    df=filter_for_vega(df)
    
    #print(df)
    #df_results=pd.DataFrame(columns=['PostionID','Bucket GIRR'])    
    #pd.DataFrame(a, columns=['Bucket','K_b'], index=['Bucket'])
    
    if df.empty:
        bucket_weighted_sensi=[]
        bucket_sensi=[]
        bucket_risk_positions=0
        risk_weights=[]
        #out=0
        vega_risk_charge=0
        buckets=[]
        factors=[]
        factor_grid1=[]
        factor_grid2=[]
    else:
         
        buckets=df['Bucket GIRR'].dropna().unique().tolist()
        n_buckets=len(buckets)
        #print(buckets)
               
        output=[calc_position_girr_vega_bucket_risk_position(position_id,bucket, corr_high_low_med) for bucket in buckets]
        
        bucket_risk_positions=np.array([item[1] for item in output])        
        bucket_sensi=[item[6] for item in output]
        risk_weights=[item[7] for item in output]        
        bucket_weighted_sensi=[item[2] for item in output]            
        factors=[item[3] for item in output]
        factor_grid1=[item[4] for item in output]
        factor_grid2=[item[5] for item in output]

        y_bc=sa.corr_matrix_bucket(sa.getcorr(0.50,corr_high_low_med), n_buckets)
        vega_risk_charge=sa.girr_vega_risk_charge(bucket_risk_positions,y_bc,np.array(bucket_weighted_sensi))  
    
    
    return [position_id,bucket_weighted_sensi,bucket_risk_positions,vega_risk_charge,buckets,factors,factor_grid1,factor_grid2,risk_weights, bucket_sensi]



# For a given position / bucket, calculate the risk position Kb
def calc_position_csr_nonsec_delta_risk_position(position_id,bucket,corr_high_low_med):
    
    # Filter the input data
    df=get_position_df(position_id,df_input_mapped)    
    df=filter_for_risk_class(df,'CSR Non Sec')
    df=filter_for_delta(df)
    
    df_bucket=filter_csr_nonsec_bucket(df,bucket)
    df_bucket=pd.merge(df_bucket,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')
    
    
    if df_bucket.empty:
        return
        
    else:
        
        df_bucket=pd.merge(df_bucket, df_params_csr_nonsec, on='Bucket CSR NonSec', how='left')

        df1=df_bucket.copy()
        df2=df_bucket.copy()
        
        # Get FG1 and convert to days365
        factorgrid1=np.array(df_bucket['FactorGrid1'])    
        factorgrid1_days=[(datestring(fg).getdays365()) for fg in factorgrid1]
    
        vertices=['0.5y','1y','3y','5y','10y']
        vertices_days=[(datestring(v).getdays365()) for v in vertices]
                
        # Get nearest points Tenor1/Tenor2 in days for each given FG1
        #tenor1_tenor2=[get_bound_points(vertices_days,d) for d in factorgrid1_days]
        tenor1_tenor2=[get_boundary_points(vertices_days,d) for d in factorgrid1_days]
        t1=[item[0] for item in tenor1_tenor2]
        t2=[item[1] for item in tenor1_tenor2]
        
        #print(factorgrid1,tenor1_tenor2,t1,t2)      
        # Weights to be assigned to sensitivity when assigning to tenor1/tenor2 
        tenor_weights=np.array([assign_weights(factorgrid1_days[i],tenor1_tenor2[i][0],tenor1_tenor2[i][1]) for i,v in enumerate(factorgrid1_days)])
        
        
        # Assign input sensitivities to nearest points t1/t2
        sensi_t1=tenor_weights[:,0]*df_bucket['Derived Sensitivity']
        sensi_t2=tenor_weights[:,1]*df_bucket['Derived Sensitivity']    
     
        risk_weights_t1=np.array(df1['Risk Weight'])
        risk_weights_t2=np.array(df1['Risk Weight'])
        
        # Weighted sensi for Tenor1/Tenor2 factors
        weighted_sensi_t1=sensi_t1*np.array(risk_weights_t1)
        weighted_sensi_t2=sensi_t2*np.array(risk_weights_t2)
     
        ######################################
        # New dataframes to handle Tenor1/Tenor2
        
        df1['t']=pd.Series(t1, index=df_bucket.index)
        df1['Sensi']=pd.Series(sensi_t1, index=df_bucket.index)
        df1['Risk Weight']=pd.Series(risk_weights_t1, index=df_bucket.index)
        df1['Weighted Sensi']=pd.Series(weighted_sensi_t1, index=df_bucket.index)
        
        df2['t']=pd.Series(t2, index=df_bucket.index)
        df2['Sensi']=pd.Series(sensi_t2, index=df_bucket.index)
        df2['Risk Weight']=pd.Series(risk_weights_t2, index=df_bucket.index)
        df2['Weighted Sensi']=pd.Series(weighted_sensi_t2, index=df_bucket.index)
        ##############################
        
        ####################################
        # Dataframe to combine df1/df2 before calculation done
        df3=df1.copy()
        
        df3=df3.append(df2, ignore_index=True)
        #df3=df3.groupby(['Bucket CSR NonSec','FactorStructure', 'LocalIssuerID', 'FactorCategory', 'Risk Weight','CCY1' ,'t','FXRate'], as_index=False)['Sensi','Weighted Sensi'].sum()
        df3=df3.groupby(['Bucket CSR NonSec','FactorStructure', 'issuer.LocalIssuerID', 'FactorCategory', 'Risk Weight','CCY1' ,'t','FXRate'], as_index=False)['Sensi','Weighted Sensi'].sum()
        #print(df3)
        ###########################################
        
        df3['Weighted Sensi USD']=df3['Weighted Sensi']/df3['FXRate']
        df3['Abs Weighted Sensi USD']=abs(df3['Weighted Sensi USD'])
        df3['Sensi USD']=df3['Sensi']/df3['FXRate']
        df3['Abs Weighted Sensi USD']=abs(df3['Weighted Sensi USD'])
        #n=len(df3['LocalIssuerID'].index)
        n=len(df3['issuer.LocalIssuerID'].index)
    
        #corr_matrix_issuers=sa.corr_matrix_name(np.array(df3[['LocalIssuerID']]))        
        
        corr_matrix_issuers=sa.corr_matrix_name(np.array(df3[['issuer.LocalIssuerID']]),corr_high_low_med)        
        corr_matrix_tenors=sa.corr_matrix_tenor(np.array(df3[['t']]),corr_high_low_med)        
        corr_matrix_basis=sa.corr_matrix_basis(np.array(df3[['FactorCategory']]),corr_high_low_med)
        
        # Corr matrix to be used for factors in the same bucket (currency)
        corr_matrix_csr_nonsec=corr_matrix_issuers*corr_matrix_tenors*corr_matrix_basis
        
        if bucket==16:
    ##            #'Other' bucket    
            k_b=sa.K_b_csrnonsec_delta_other(df3['Abs Weighted Sensi USD'])
        else:    
            k_b=sa.K_b_csrnonsec_delta(df3['Weighted Sensi USD'],corr_matrix_csr_nonsec)
            
            
        # (list of) weighted sensitivities for the bucket 
        ws_b=df3['Weighted Sensi USD'].tolist()
        factors=df3['FactorStructure'].tolist()
        factor_grids=df3['t'].tolist()
        #print(kb_med)

    #return [df1,df2,df3,kb_med,ws_b,corr_matrix_girr]
    return [df1,df2,df3,k_b,ws_b,factors, factor_grids, df3['Risk Weight'].tolist(), df3['Sensi USD'].tolist(),df3['Bucket CSR NonSec'].tolist()]


# Calculates the delta risk charge for given position
def calc_position_csr_nonsec_delta_risk_charge(position_id,corr_high_low_med):
    
    df=get_position_df(position_id,df_input_mapped)
    df=filter_for_risk_class(df,'CSR Non Sec')
    df=filter_for_delta(df)
    
    
    if df.empty:    
        return

    else:
    
        # Get all buckets of the position
        buckets=df['Bucket CSR NonSec'].unique().tolist()
        #print(buckets)
        
        n_buckets=len(buckets)
        index_bucket16=[i for i,x in enumerate(buckets) if x==16]    
    
        frtb_sectors=[df_params_csr_nonsec.loc[df_params_csr_nonsec['Bucket CSR NonSec']==bucket,['Credit Sector']].iloc[0] for bucket in buckets]
        frtb_credit_quality=[df_params_csr_nonsec.loc[df_params_csr_nonsec['Bucket CSR NonSec']==bucket,['Credit Quality']].iloc[0] for bucket in buckets]
        
        buckets_without16=[buckets[i] for i,x in enumerate(buckets) if x!=16]    
        frtb_sectors_without16=[frtb_sectors[i] for i,x in enumerate(buckets) if x!=16]    
        frtb_credit_quality_without16=[frtb_credit_quality[i] for i,x in enumerate(buckets) if x!=16]
        
        output=[calc_position_csr_nonsec_delta_risk_position(position_id,bucket,corr_high_low_med) for bucket in buckets]
        
        # Bucket Risk positions for each bucket in the position
        bucket_risk_positions=np.array([item[3] for item in output])
        #print(bucket_risk_positions)
                
        # Bucket weighted sensis for each bucket in the position
        bucket_weighted_sensi=[item[4] for item in output]
        #print(bucket_weighted_sensi)
        
        # Factora for each bucket in the position
        factors=[item[5] for item in output]
        #print(factors)
        
        factor_grids=[item[6] for item in output]       
        risk_weights=[item[7] for item in output]       
        sensis=[item[8] for item in output]       
        factor_buckets=[item[9] for item in output]       
    

        y_bc_sector=sa.corr_matrix_csrnonsec_sectors_ybc(np.array(buckets),corr_high_low_med)
        y_bc_rating=sa.corr_matrix_csrnonsec_ratings_ybc(np.array(buckets),corr_high_low_med)
        
        
        y_bc = y_bc_sector*y_bc_rating
        
        
        delta_risk_charge=sa.delta_vega_risk_charge(np.array(bucket_risk_positions),y_bc,np.array(bucket_weighted_sensi))  
        
    #return [buckets,k_b,s_b,delta_risk_charge]
    return [position_id,bucket_weighted_sensi,bucket_risk_positions,delta_risk_charge,buckets,factors, factor_grids, risk_weights, sensis, factor_buckets]





def calc_position_csr_nonsec_curvature_bucket_risk_position(position_id,bucket, corr_high_low_med):
    
    # Filter the input data
    df=get_position_df(position_id,df_input_mapped)
    df=filter_for_risk_class(df,'CSR Non Sec')
    df=filter_for_delta(df)
    
    df_bucket=filter_csr_nonsec_bucket(df,bucket)
    df_bucket=pd.merge(df_bucket,df_fx_rates[['Ccy','FXRate']],left_on=['PosCcy'], right_on=['Ccy'], how='left')
    
    df_bucket=df_bucket[df_bucket['Grid']=='P']
    df_bucket=df_bucket[df_bucket['Curvature']==True]
        
    if df_bucket.empty:
        return
    else:
        
        df_bucket=pd.merge(df_bucket, df_params_csr_nonsec, on='Bucket CSR NonSec', how='left')
        df_bucket.insert(len(df_bucket.columns),'Curvature Risk Weight',0)
        df_bucket['Curvature Risk Weight']=0.012
        df_bucket=df_bucket.groupby(['Bucket CSR NonSec','FactorStructure', 'LocalIssuerID', 'FactorCategory', 'Curvature Risk Weight','FXRate','SR.X','SR.Y'], as_index=False)['Derived Sensitivity'].sum()
        df_bucket.insert(len(df_bucket.columns),'V_up',0)
        df_bucket.insert(len(df_bucket.columns),'V_down',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_up',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_down',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_up USD',0)
        df_bucket.insert(len(df_bucket.columns),'CVR_down USD',0)
        df_bucket.insert(len(df_bucket.columns),'CVR',0)
        
        all_x=list(df_bucket['SR.X'].str.split(' '))
        all_x=convertlistofliststofloat(all_x)
        
        all_V_x=list(df_bucket['SR.Y'].str.split(' '))
        all_V_x=convertlistofliststofloat(all_V_x)
        
        risk_weights=df_bucket['Curvature Risk Weight']
        sensitivities=df_bucket['Derived Sensitivity']
        
        V_up=np.empty(len(all_V_x))
        V_down=np.empty(len(all_V_x))
        
        CVR_up=np.empty(len(all_V_x))
        CVR_down=np.empty(len(all_V_x))    
        CVR=np.empty(len(all_V_x))
        
        
        for i,v in enumerate(all_x):
        
            ## To be removed/?? When Shock Record arrays have different lengths
            if len(all_x[i])!=len(all_V_x[i]):
                CVR=0            
            else:                            
                V=interpolate.interp1d(np.array(all_x[i]),np.array(all_V_x[i]), kind='linear', fill_value='extrapolate')
                cvr_rw=risk_weights[i]
                
                x_up=cvr_rw
                x_down=-cvr_rw
                
                V_up[i]=V(x_up)
                V_down[i]=V(x_down)
                               
                CVR_up[i]=sa.CVR_up(V_up[i],cvr_rw,sensitivities[i])
                CVR_down[i]=sa.CVR_down(V_down[i],cvr_rw,sensitivities[i])
                
                CVR=-min(CVR_up[i],CVR_down[i])
                            
        df_bucket['V_up']=V_up
        df_bucket['V_down']=V_down
        df_bucket['CVR_up']=CVR_up
        df_bucket['CVR_down']=CVR_down
        df_bucket['CVR']=CVR

        df_bucket['V_up USD']=df_bucket['V_up']/df_bucket['FXRate']
        df_bucket['V_down USD']=df_bucket['V_down']/df_bucket['FXRate']        
        df_bucket['CVR_up USD']=df_bucket['CVR_up']/df_bucket['FXRate']
        df_bucket['CVR_down USD']=df_bucket['CVR_down']/df_bucket['FXRate']

        CVR_up_total=df_bucket['CVR_up USD'].sum()
        CVR_down_total=df_bucket['CVR_down USD'].sum()
        
        corr_matrix_issuers=sa.corr_matrix_name(np.asarray(df_bucket[['LocalIssuerID']]),corr_high_low_med)
        corr_matrix_issuers=corr_matrix_issuers*corr_matrix_issuers

        k_b_up=sa.K_b_csrnonsec_curvature_up(np.asarray(df_bucket['CVR_up USD']), corr_matrix_issuers)        
        k_b_down=sa.K_b_csrnonsec_curvature_down(np.asarray(df_bucket['CVR_down USD']), corr_matrix_issuers)        
        
        K_b=max(k_b_up,k_b_down)
        
        if K_b==k_b_up:
            S_b=sa.S_b_curvature(df_bucket['CVR_up USD'],df_bucket['CVR_down USD'],K_b,k_b_up,k_b_down)
        elif K_b==k_b_down:
            S_b=sa.S_b_curvature(df_bucket['CVR_up USD'],df_bucket['CVR_down USD'],K_b,k_b_up,k_b_down)
        else:
            S_b=0

    return [df_bucket,K_b,k_b_up,k_b_down,S_b]




def calc_position_csr_nonsec_curvature_risk_charge(position_id, corr_high_low_med):    
    # Filter the input data
    df=get_position_df(position_id,df_input_mapped)
    #print(df.Grid) 
    df=filter_for_risk_class(df,'CSR Non Sec')
    df=filter_for_delta(df)
    df=df[df['Grid']=='P']
    df=df[df['Curvature']==True]    
    
    if df.empty:
        return
    else:
        # Get all buckets of the position
        buckets=df['Bucket CSR NonSec'].unique().tolist()
        #n_buckets=len(buckets)
        #print(buckets)

        output=[calc_position_csr_nonsec_curvature_bucket_risk_position(position_id,bucket,corr_high_low_med) for bucket in buckets]

        # Bucket Risk positions
        
        bucket_K_bs=np.array([item[1] for item in output])
        #print(bucket_risk_positions)
        
        
        #bucket_curvature_sensi=[calc_position_girr_curvature_bucket_risk_position(position_id,bucket)[2] for bucket in buckets]
        bucket_S_bs=[item[4] for item in output]        
        #bucket_curvature_up=[item[3] for item in output]
        #bucket_curvature_down=[item[4] for item in output]

        #print(bucket_curvature_sensi)
        
        #factors=[calc_position_girr_delta_bucket_risk_position(position_id,bucket)[5] for bucket in buckets]
        #factors=[item[5] for item in output]
        #factors_V_up=[item[6] for item in output]
        #factors_V_down=[item[7] for item in output]        
        #factors_CVR_up=[item[8] for item in output]
        #factors_CVR_down=[item[9] for item in output]

        y_bc_sector=sa.corr_matrix_csrnonsec_sectors_ybc(np.array(buckets),corr_high_low_med)
        y_bc_rating=sa.corr_matrix_csrnonsec_ratings_ybc(np.array(buckets),corr_high_low_med)
        
        y_bc = y_bc_sector*y_bc_rating
        
        curvature_risk_charge=sa.curvature_risk_charge(bucket_K_bs,y_bc,np.array(bucket_S_bs))

    return [position_id,curvature_risk_charge]




def calc_position_equity_total_risk_charge(position_id,corr_high_low_med):
    
    
    eq_delta_rc=calc_position_equity_delta_risk_charge(position_id,corr_high_low_med)
    eq_vega_rc=calc_position_equity_vega_risk_charge(position_id,corr_high_low_med)
    eq_crv_rc=calc_position_equity_curvature_risk_charge(position_id,corr_high_low_med)
    
    return eq_delta_rc['delta_rc']+eq_vega_rc['vega_rc']+eq_crv_rc['cvr_rc']
    

#def calc_position_total_risk_charge(position_id,corr_high_low_med):
#    eq_delta_rc=calc_position_equity_delta_risk_charge(position_id,corr_high_low_med)    
    
    


run_portfolio=True

#if run_portfolio:
##    for pos in positions:
##        print(pos)
##        #r=calc_position_girr_delta_risk_charge(pos)
##        #r=calc_position_girr_curvature_risk_charge(pos)
##        #r=calc_position_girr_vega_risk_charge(pos)
##        r=calc_position_equity_delta_risk_charge(pos)
##        #r=calc_position_equity_vega_risk_charge(pos)
##        #r=calc_position_equity_curvature_risk_charge(pos)
##        #r=calc_position_csr_nonsec_delta_risk_charge(pos)
##        
##        
##        results.append(r)
#
#else:
#    #pos='008-CCYSWAP-JPY'
#    #pos='008-GOVFUTOP-JPY'
#    pos='528-FXFWD-ZAR'
#    #pos='002-GOVFUT-JPY'
#    #r=calc_position_equity_curvature_risk_charge(pos)
#    #r=calc_position_equity_curvature_bucket_risk_position(pos,8)
#    #r=calc_position_girr_curvature_bucket_risk_position(pos,'JPY')
#    #r=calc_position_girr_curvature_risk_charge(pos)
#    #r=calc_position_girr_vega_risk_charge(pos)
#    r=calc_position_girr_delta_risk_charge(pos)
#    #r=calc_position_girr_delta_bucket_risk_position(pos,'JPY')
#    #r=calc_position_girr_vega_bucket_risk_position(pos,'JPY')
#    results.append(r)
#    
#df_results=pd.DataFrame(data=results)


############################# Results Processing and Exporting ##############
#
#new_data=[]
#resultscoll=[]
#
#for item in results:
#    
#        position_id=item[0]
#        buckets=item[4]
#        bucketweightedsensis=item[1]
#        bucketriskpositions=item[2]
#        #weigted_sensi=item
#        factors=item[5]
#        riskcharge=item[3]
#        
#        #for bws in bucketweightedsensis:
#            #for factor in factors:
#                #print(bws)
#                #print(factor)
#            
#        r=result(position_id,'RiskCharge',riskcharge,'GIRR')
#        
#        #r=result(position_id,'RiskCharge',riskcharge,'GIRR')
#        
#        
#        resultscoll.append(r)
#        #position_id, measure, expected_result, risk_class
#        

######################################################################
#####Results Dataframes for GIRR Delta#################################
#

# Construct Bucket Risk position results in a dataframe
#
#df_results_girr_delta_bucket_risk_position=pd.DataFrame()
#df_results_girr_delta_factor_weighted_sensi=pd.DataFrame()
#df_results_girr_delta_risk_charge=pd.DataFrame()
#
#for item in results:
#    position_id=item[0]
#    weighted_sensis=item[1]
#    bucketriskpositions=item[2]
#    risk_charge=item[3]
#    buckets=item[4]
#    factors=item[5]
#    factor_grids=item[6]
#    sensis=item[7]
#    risk_weights=item[8]
#
#    for bucket in buckets:
#        #print(bucket)
#        i=buckets.index(bucket)
#        add=[position_id,bucket,bucketriskpositions[i]]
#        df_data_add=pd.DataFrame([add],columns=['Position_id', 'Bucket', 'Risk Position'])        
#        df_results_girr_delta_bucket_risk_position=df_results_girr_delta_bucket_risk_position.append(df_data_add)
#        
#
#    weighted_sensis=reduce(lambda x,y: x+y,weighted_sensis,[])
#    sensis=reduce(lambda x,y: x+y,sensis,[])
#    risk_weights=reduce(lambda x,y: x+y,risk_weights,[])
#    factors=reduce(lambda x,y: x+y,factors,[])
#    factor_grids=reduce(lambda x,y: x+y,factor_grids,[])
#    
#    
#    for i,v in enumerate(factors):
#        factor=factors[i]
#        factor_grid=factor_grids[i]
#        weighted_sensi=weighted_sensis[i]
#        sensi=sensis[i]
#        risk_weight=risk_weights[i]
#        add=[position_id,factor, factor_grid, sensi, risk_weight, weighted_sensi]
#        df_data_add=pd.DataFrame([add],columns=['Position_id', 'Factor', 't', 'Sensi', 'Risk Weight', 'Weighted Sensi'])        
#        df_results_girr_delta_factor_weighted_sensi=df_results_girr_delta_factor_weighted_sensi.append(df_data_add)
#    
#    add=[position_id,risk_charge]
#    df_data_add=pd.DataFrame([add],columns=['PositionID', 'Delta Risk Charge'])        
#    df_results_girr_delta_risk_charge=df_results_girr_delta_risk_charge.append(df_data_add)
#


########################################################################
#########Results Dataframes for GIRR Curvature#################################

def run_girr_curvature_calcs(produce_output):
    """ 
    """
    results=[]
    
    for pos in positions:
        print(pos)
        r=calc_position_girr_curvature_risk_charge(pos)    
        results.append(r)

    ## Construct Bucket Risk position results in a dataframe
    df_results_girr_curvature_bucket_risk_position=pd.DataFrame()
    df_results_girr_curvature_factor_curvature=pd.DataFrame()
    df_results_girr_curvature_risk_charge=pd.DataFrame()
        
    for item in results:
        position_id=item[0]
        bucket_curvature_sensi=item[1]
        bucket_risk_positions=item[2]
        risk_charge=item[3]
        buckets=item[4]
        factors=item[5]
        V_up=item[6]
        V_down=item[7]
        CVR_up=item[8]
        CVR_down=item[9]
        #CVR=item[10]
        
        
        for bucket in buckets:
            i=buckets.index(bucket)
            add=[position_id,bucket,bucket_risk_positions[i]]
            df_data_add=pd.DataFrame([add],columns=['Position_id', 'Bucket', 'Risk Position'])        
            df_results_girr_curvature_bucket_risk_position=df_results_girr_curvature_bucket_risk_position.append(df_data_add)
            
    
        bucket_curvature_sensi=reduce(lambda x,y: x+y,bucket_curvature_sensi,[])
        factors=reduce(lambda x,y: x+y,factors,[])
        V_up=reduce(lambda x,y: x+y,V_up,[])
        V_down=reduce(lambda x,y: x+y,V_down,[])
        CVR_up=reduce(lambda x,y: x+y,CVR_up,[])
        CVR_down=reduce(lambda x,y: x+y,CVR_down,[])
        
        for i,v in enumerate(factors):
        
            add=[position_id,factors[i],V_up[i],V_down[i], CVR_up[i],CVR_down[i]]
            df_data_add=pd.DataFrame([add],columns=['Position_id', 'Factor', 'V_up', 'V_down','CVR_up', 'CVR_down'])        
            df_results_girr_curvature_factor_curvature=df_results_girr_curvature_factor_curvature.append(df_data_add)
        
        add=[position_id,risk_charge]
        df_data_add=pd.DataFrame([add],columns=['PositionID', 'Curvature Risk Charge'])        
        df_results_girr_curvature_risk_charge=df_results_girr_curvature_risk_charge.append(df_data_add)

    
    if produce_output:
        outputresults_girr_curvature_csv(df_results_girr_curvature_risk_charge,df_results_girr_curvature_factor_curvature,df_results_girr_curvature_bucket_risk_position)


def run_girr_vega_calcs(produce_output):
    
    results=[]
    
    for pos in positions:
        print(pos)
        r=calc_position_girr_vega_risk_charge(pos)    
        results.append(r)
        #
    ###########################################################################
    ##########Results Dataframes for GIRR Vega   #################################
    
    # Construct Bucket Risk position results in a dataframe
    if run_portfolio:
            
        df_results_girr_vega_bucket_risk_position=pd.DataFrame()
        df_results_girr_vega_factor_weighted_sensi=pd.DataFrame()
        df_results_girr_vega_risk_charge=pd.DataFrame()
        
        for item in results:
            position_id=item[0]
            weighted_sensis=item[1]
            sensis=item[9]
            risk_weights=item[8]
            fg1=item[6]
            fg2=item[7]
            #print(fg1,fg2)
            bucketriskpositions=item[2]
            risk_charge=item[3]
            buckets=item[4]
            factors=item[5]
            
            weighted_sensis=reduce(lambda x,y: x+y,weighted_sensis,[])
            factors=reduce(lambda x,y: x+y,factors,[])
            fg1=reduce(lambda x,y: x+y,fg1,[])
            fg2=reduce(lambda x,y: x+y,fg2,[])
            sensis=reduce(lambda x,y: x+y,sensis,[])
            risk_weights=reduce(lambda x,y: x+y,risk_weights,[])
            #print(factors,fg1,fg2)
            #print(factors[0],fg1[0])
            
            
            for bucket in buckets:
                i=buckets.index(bucket)
                add=[position_id,bucket,bucketriskpositions[i]]
                df_data_add=pd.DataFrame([add],columns=['Position ID', 'Bucket', 'Risk Position'])        
                df_results_girr_vega_bucket_risk_position=df_results_girr_vega_bucket_risk_position.append(df_data_add)
            for i,v in enumerate(factors):
                f=factors[i]
                weighted_sensi=weighted_sensis[i]
                sensi=sensis[i]
                grid1=fg1[i]
                grid2=fg2[i]
                rw=risk_weights[i]
                add=[position_id,f, grid1,grid2, sensi,rw, weighted_sensi]
                df_data_add=pd.DataFrame([add],columns=['Position ID', 'Factor', 'FactorGrid1', 'FactorGrid2', 'Sensi','Risk Weight','Weighted Sensi'])        
                df_results_girr_vega_factor_weighted_sensi=df_results_girr_vega_factor_weighted_sensi.append(df_data_add)
                
            add=[position_id,risk_charge]
            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Vega Risk Charge'])        
            df_results_girr_vega_risk_charge=df_results_girr_vega_risk_charge.append(df_data_add)
        
    
    if produce_output:
        outputresults_girr_vega_csv(df_results_girr_vega_risk_charge,df_results_girr_vega_factor_weighted_sensi,df_results_girr_vega_bucket_risk_position)


###########################################################################

##########Results Dataframes for CSR NonSec Delta#################################
#
#
## Construct Bucket Risk position results in a dataframe

def run_csr_nonsec_delta_calcs(produce_output):
    
    results=[]
    
    for pos in positions:
        print(pos)
        r=calc_position_csr_nonsec_delta_risk_charge(pos)    
        results.append(r)


    df_results_csr_nonsec_delta_bucket_risk_position=pd.DataFrame()
    df_results_csr_nonsec_delta_factor_weighted_sensi=pd.DataFrame()
    df_results_csr_nonsec_delta_risk_charge=pd.DataFrame()
    
    for item in results:
        position_id=item[0]
        weighted_sensis=item[1]
        bucketriskpositions=item[2]
        risk_charge=item[3]
        buckets=item[4]
        factors=item[5]
        factor_grids=item[6]
        risk_weights=item[7]
        sensis=item[8]
        factor_buckets=item[9]
        
        for bucket in buckets:
            i=buckets.index(bucket)
            add=[position_id,bucket,bucketriskpositions[i]]
            df_data_add=pd.DataFrame([add],columns=['Position_id', 'Bucket', 'Risk Position'])        
            df_results_csr_nonsec_delta_bucket_risk_position=df_results_csr_nonsec_delta_bucket_risk_position.append(df_data_add)
            
    
        weighted_sensis=reduce(lambda x,y: x+y,weighted_sensis,[])
        factors=reduce(lambda x,y: x+y,factors,[])
        factor_grids=reduce(lambda x,y: x+y,factor_grids,[])
        risk_weights=reduce(lambda x,y: x+y,risk_weights,[])
        sensis=reduce(lambda x,y: x+y,sensis,[])
        factor_buckets=reduce(lambda x,y: x+y,factor_buckets,[])
        
        for i,v in enumerate(factors):
            factor=factors[i]
            factor_grid=factor_grids[i]
            weighted_sensi=weighted_sensis[i]
            add=[position_id,factor, factor_grid, weighted_sensi, risk_weights[i], sensis[i],factor_buckets[i]]
            df_data_add=pd.DataFrame([add],columns=['Position_id', 'Factor', 't', 'Weighted Sensi', 'Risk Weight', 'Sensi', 'Bucket'])        
            df_results_csr_nonsec_delta_factor_weighted_sensi=df_results_csr_nonsec_delta_factor_weighted_sensi.append(df_data_add)
        
        add=[position_id,risk_charge]
        df_data_add=pd.DataFrame([add],columns=['PositionID', 'Delta Risk Charge'])        
        df_results_csr_nonsec_delta_risk_charge=df_results_csr_nonsec_delta_risk_charge.append(df_data_add)

    if produce_output:
        outputresults_csr_nonsec_delta_csv(df_results_csr_nonsec_delta_risk_charge,df_results_csr_nonsec_delta_factor_weighted_sensi,df_results_csr_nonsec_delta_bucket_risk_position)

#[position_id,bucket_weighted_sensi,bucket_risk_positions,delta_risk_charge,buckets,factors, factor_grids]

############################################################################
###########Results Dataframes for Equity Vega   #################################


#[position_id,bucket_weighted_sensi,bucket_risk_positions,vega_risk_charge,buckets,factors, risk_weights, factor_sensis, factor_tenors]
## Construct Bucket Risk position results in a dataframe

def run_equity_vega_calcs(produce_output, corr_high_low_med):
    
    results=[]
    
    for pos in positions:
        print(pos)
        r=calc_position_equity_vega_risk_charge(pos, corr_high_low_med)    
        results.append(r)

    
    df_results_eq_vega_bucket_risk_position=pd.DataFrame()
    df_results_eq_vega_factor_weighted_sensi=pd.DataFrame()
    df_results_eq_vega_risk_charge=pd.DataFrame()
    
    for item in results:
        if item is None:
            continue
        #position_id=item[0]
        position_id=item['position_id']
        weighted_sensis=item['weighted_sensi']
        bucketriskpositions=item['K_b']
        risk_charge=item['vega_rc']
        buckets=item['buckets']
        factors=item['factors']
        risk_weights=item['RWs']
        sensis=item['Sensis']
        tenors=item['Tenors']
        factor_buckets=item['Factor_Buckets']
 
        
        weighted_sensis=reduce(lambda x,y: x+y,weighted_sensis,[])
        factors=reduce(lambda x,y: x+y,factors,[])
        risk_weights=reduce(lambda x,y: x+y,risk_weights,[])
        sensis=reduce(lambda x,y: x+y,sensis,[])
        tenors=reduce(lambda x,y: x+y,tenors,[])
        factor_buckets=reduce(lambda x,y: x+y,factor_buckets,[])
        
        for bucket in buckets:
            i=buckets.index(bucket)
            add=[position_id,bucket,bucketriskpositions[i]]
            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Bucket', 'Risk Position'])        
            df_results_eq_vega_bucket_risk_position=df_results_eq_vega_bucket_risk_position.append(df_data_add)
        for i,v in enumerate(factors):
            f=factors[i]
            weighted_sensi=weighted_sensis[i]
            add=[position_id,f, weighted_sensi, risk_weights[i], sensis[i], tenors[i],factor_buckets[i]]
            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Factor', 'Weighted Sensi', 'Risk Weight', 'Sensi', 'Tenor', 'Bucket'])        
            df_results_eq_vega_factor_weighted_sensi=df_results_eq_vega_factor_weighted_sensi.append(df_data_add)
            
        add=[position_id,risk_charge]
        df_data_add=pd.DataFrame([add],columns=['Position ID', 'Vega Risk Charge'])        
        df_results_eq_vega_risk_charge=df_results_eq_vega_risk_charge.append(df_data_add)
    
    if produce_output:
        outputresults_eq_vega_csv(df_results_eq_vega_risk_charge,df_results_eq_vega_factor_weighted_sensi,df_results_eq_vega_bucket_risk_position)

    return results

#############################################################################
############Results Dataframes for FX Delta   #################################
def run_fx_curvature_calcs(produce_output):
    
    results=[]
    
    for pos in positions:
        print(pos)
        r=calc_position_fx_curvature_risk_charge(pos)    
        results.append(r)

### Construct Bucket Risk position results in a dataframe
#    df_results_fx_delta_bucket_risk_position=pd.DataFrame()
    df_results_fx_curvature_factor_weighted_sensi=pd.DataFrame()
    df_results_fx_curvature_risk_charge=pd.DataFrame()
    
    for item in results:
        position_id=item[0]
        curvature_sensis=item[1]
        #bucketriskpositions=item[2]
        risk_charge=item[2]
        #buckets=item[4]
        factors=item[3]
        risk_weights=item[4]
        factors_V_up=item[5]
        factors_V_down=item[6]
        factors_CVR_up=item[7]
        factors_CVR_down=item[8]
        
        #factor_buckets=item[8]
        
        #print(position_id,factors)
        
#        for bucket in buckets:
#            i=buckets.index(bucket)
#            add=[position_id,bucket,bucketriskpositions[i]]
#            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Bucket', 'Risk Position'])        
#            df_results_fx_delta_bucket_risk_position=df_results_fx_delta_bucket_risk_position.append(df_data_add)
        for i,v in enumerate(factors):
            #f=factors[i]
            #weighted_sensi=weighted_sensis[i]
            add=[position_id,factors[i], curvature_sensis[i], risk_weights[i],factors_V_up[i],factors_V_down[i],factors_CVR_up[i],factors_CVR_down[i]]
            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Factor', 'Curvature Sensi', 'Risk Weight', 'V_up', 'V_down', 'CVR_up', 'CVR_down'])        
            df_results_fx_curvature_factor_weighted_sensi=df_results_fx_curvature_factor_weighted_sensi.append(df_data_add)
            
        add=[position_id,risk_charge]
        df_data_add=pd.DataFrame([add],columns=['Position ID', 'Curvature Risk Charge'])        
        df_results_fx_curvature_risk_charge=df_results_fx_curvature_risk_charge.append(df_data_add)

    if produce_output:
        outputresults_fx_curvature_csv(df_results_fx_curvature_risk_charge,df_results_fx_curvature_factor_weighted_sensi)


#############################################################################
############Results Dataframes for FX Delta   #################################
def run_fx_delta_calcs(produce_output):
    
    results=[]
    
    for pos in positions:
        print(pos)
        r=calc_position_fx_delta_risk_charge(pos)    
        results.append(r)

### Construct Bucket Risk position results in a dataframe
#    df_results_fx_delta_bucket_risk_position=pd.DataFrame()
    df_results_fx_delta_factor_weighted_sensi=pd.DataFrame()
    df_results_fx_delta_risk_charge=pd.DataFrame()
    
    for item in results:
        position_id=item[0]
        weighted_sensis=item[1]
        #bucketriskpositions=item[2]
        risk_charge=item[2]
        #buckets=item[4]
        factors=item[3]
        risk_weights=item[4]
        sensis=item[5]
        #factor_buckets=item[8]
        
        print(position_id,factors)
        #weighted_sensis=reduce(lambda x,y: x+y,weighted_sensis,[])
        #factors=reduce(lambda x,y: x+y,factors,[])
        #risk_weights=reduce(lambda x,y: x+y,risk_weights,[])
        #sensis=reduce(lambda x,y: x+y,sensis,[])
        #factor_buckets=reduce(lambda x,y: x+y,factor_buckets,[])
        
#        for bucket in buckets:
#            i=buckets.index(bucket)
#            add=[position_id,bucket,bucketriskpositions[i]]
#            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Bucket', 'Risk Position'])        
#            df_results_fx_delta_bucket_risk_position=df_results_fx_delta_bucket_risk_position.append(df_data_add)
        for i,v in enumerate(factors):
            f=factors[i]
            weighted_sensi=weighted_sensis[i]
            add=[position_id,f, weighted_sensi, sensis[i], risk_weights[i]]
            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Factor', 'Weighted Sensi', 'Sensi', 'Risk Weight'])        
            df_results_fx_delta_factor_weighted_sensi=df_results_fx_delta_factor_weighted_sensi.append(df_data_add)
            
        add=[position_id,risk_charge]
        df_data_add=pd.DataFrame([add],columns=['Position ID', 'Delta Risk Charge'])        
        df_results_fx_delta_risk_charge=df_results_fx_delta_risk_charge.append(df_data_add)

    if produce_output:
        outputresults_fx_delta_csv(df_results_fx_delta_risk_charge,df_results_fx_delta_factor_weighted_sensi)



def run_fx_vega_calcs(produce_output):
    
    results=[]
    
    for pos in positions:
        print(pos)
        r=calc_position_fx_vega_risk_charge(pos)    
        results.append(r)

### Construct Bucket Risk position results in a dataframe
#    df_results_fx_delta_bucket_risk_position=pd.DataFrame()
    df_results_fx_vega_factor_weighted_sensi=pd.DataFrame()
    df_results_fx_vega_risk_charge=pd.DataFrame()
    
    #[position_id,weighted_sensi,vega_risk_charge,factors, risk_weights,factor_sensi]
    
    for item in results:
        position_id=item[0]
        weighted_sensis=item[1]
        #bucketriskpositions=item[2]
        risk_charge=item[2]
        #buckets=item[4]
        factors=item[3]
        risk_weights=item[4]
        sensis=item[5]
        factor_tenors=item[6]
        #factor_buckets=item[8]
        
        print(position_id,factors)
        #weighted_sensis=reduce(lambda x,y: x+y,weighted_sensis,[])
        #factors=reduce(lambda x,y: x+y,factors,[])
        #risk_weights=reduce(lambda x,y: x+y,risk_weights,[])
        #sensis=reduce(lambda x,y: x+y,sensis,[])
        #factor_buckets=reduce(lambda x,y: x+y,factor_buckets,[])
        
#        for bucket in buckets:
#            i=buckets.index(bucket)
#            add=[position_id,bucket,bucketriskpositions[i]]
#            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Bucket', 'Risk Position'])        
#            df_results_fx_delta_bucket_risk_position=df_results_fx_delta_bucket_risk_position.append(df_data_add)
        for i,v in enumerate(factors):
            #f=factors[i]
            weighted_sensi=weighted_sensis[i]
            add=[position_id,factors[i], weighted_sensi, sensis[i], risk_weights[i],factor_tenors[i]]
            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Factor', 'Weighted Sensi', 'Sensi', 'Risk Weight', 't'])        
            df_results_fx_vega_factor_weighted_sensi=df_results_fx_vega_factor_weighted_sensi.append(df_data_add)
            
        add=[position_id,risk_charge]
        df_data_add=pd.DataFrame([add],columns=['Position ID', 'Delta Risk Charge'])        
        df_results_fx_vega_risk_charge=df_results_fx_vega_risk_charge.append(df_data_add)

    if produce_output:
        outputresults_fx_vega_csv(df_results_fx_vega_risk_charge,df_results_fx_vega_factor_weighted_sensi)


#
##############################################################################
#############Results Dataframes for Equity Delta   #################################

def run_equity_delta_calcs(produce_output, corr_high_low_med):
    
    results=[]
    
    for pos in positions:
        print(pos)
        r=calc_position_equity_delta_risk_charge(pos,corr_high_low_med)    
        results.append(r)
        
    df_results_eq_delta_bucket_risk_position=pd.DataFrame()
    df_results_eq_delta_factor_weighted_sensi=pd.DataFrame()
    df_results_eq_delta_risk_charge=pd.DataFrame()
    
    for item in results:
        if item is None:
            continue
        
        position_id=item['position_id']
        weighted_sensis=item['weighted_sensi']
        bucketriskpositions=item['K_b']
        risk_charge=item['delta_rc']
        buckets=item['Factor_Buckets']
        factors=item['factors']
        risk_weights=item['RWs']
        sensis=item['Sensis']
        factor_buckets=item['Factor_Buckets']
        factor_sectors=item['Sectors']
        factor_mkt_cap=item['Market Cap']
        factor_cty=item['Country']
        factor_econ=item['Economy']
        
        
        weighted_sensis=reduce(lambda x,y: x+y,weighted_sensis,[])
        factors=reduce(lambda x,y: x+y,factors,[])
        risk_weights=reduce(lambda x,y: x+y,risk_weights,[])
        sensis=reduce(lambda x,y: x+y,sensis,[])
        factor_buckets=reduce(lambda x,y: x+y,factor_buckets,[])
        factor_sectors=reduce(lambda x,y: x+y,factor_sectors,[])
        factor_mkt_cap=reduce(lambda x,y: x+y,factor_mkt_cap,[])
        factor_cty=reduce(lambda x,y: x+y,factor_cty,[])
        factor_econ=reduce(lambda x,y: x+y,factor_econ,[])
        
        
        for bucket in buckets:
            i=buckets.index(bucket)
            add=[position_id,bucket,bucketriskpositions[i]]
            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Bucket', 'Risk Position'])        
            df_results_eq_delta_bucket_risk_position=df_results_eq_delta_bucket_risk_position.append(df_data_add)
        for i,v in enumerate(factors):
            f=factors[i]
            weighted_sensi=weighted_sensis[i]
            add=[position_id,f, weighted_sensi, sensis[i], risk_weights[i],factor_buckets[i],factor_sectors[i],factor_mkt_cap[i],factor_cty[i],factor_econ[i]]
            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Factor', 'Weighted Sensi', 'Sensi', 'Risk Weight', 'Bucket', 'Sector', 'Mkt Cap $', 'Country', 'Economy'])        
            df_results_eq_delta_factor_weighted_sensi=df_results_eq_delta_factor_weighted_sensi.append(df_data_add)
            
        add=[position_id,risk_charge]
        df_data_add=pd.DataFrame([add],columns=['Position ID', 'Delta Risk Charge'])        
        df_results_eq_delta_risk_charge=df_results_eq_delta_risk_charge.append(df_data_add)
    

    if produce_output:
        outputresults_eq_delta_csv(df_results_eq_delta_risk_charge,df_results_eq_delta_factor_weighted_sensi,df_results_eq_delta_bucket_risk_position)
#    
    return results
        
#############################################################################
#############Results Dataframes for Equity Curvature   #################################
##[position_id,bucket_curvature_sensi,bucket_risk_positions,curvature_risk_charge,buckets,factors,risk_weights,factors_V_up,factors_V_down,
##            factors_CVR_up,factors_CVR_down]
##
## Construct Bucket Risk position results in a dataframe

def run_equity_curvature_calcs(produce_output, corr_high_low_med):
    
    results=[]
    
    for pos in positions:
        print(pos)
        r=calc_position_equity_curvature_risk_charge(pos,corr_high_low_med) 
        results.append(r)
    
    df_results_eq_curvature_bucket_risk_position=pd.DataFrame()
    df_results_eq_curvature_factor_weighted_sensi=pd.DataFrame()
    df_results_eq_curvature_risk_charge=pd.DataFrame()
    
    for item in results:
        if item is None:
            continue
        
        position_id=item['position_id']
        weighted_sensis=item['CVR_k']
        bucketriskpositions=item['K_b']
        risk_charge=item['cvr_rc']
        buckets=item['buckets']
        factors=item['factors']
        risk_weights=item['RWs']
        V_up=item['V_up']
        V_down=item['V_down']
        CVR_up=item['CVR_up']
        CVR_down=item['CVR_down']
        factor_buckets=item['Factor_Buckets']
        #print(position_id, V_up)
        
        weighted_sensis=reduce(lambda x,y: x+y,weighted_sensis,[])
        factors=reduce(lambda x,y: x+y,factors,[])
        V_up=reduce(lambda x,y: x+y,V_up,[])
        V_down=reduce(lambda x,y: x+y,V_down,[])
        CVR_up=reduce(lambda x,y: x+y,CVR_up,[])
        CVR_down=reduce(lambda x,y: x+y,CVR_down,[])
        factor_buckets=reduce(lambda x,y: x+y,factor_buckets,[])
        
        
        for bucket in buckets:
            i=buckets.index(bucket)
            add=[position_id,bucket,bucketriskpositions[i]]
            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Bucket', 'Risk Position'])        
            df_results_eq_curvature_bucket_risk_position=df_results_eq_curvature_bucket_risk_position.append(df_data_add)
        for i,v in enumerate(factors):
            f=factors[i]
            weighted_sensi=weighted_sensis[i]
            add=[position_id,f, weighted_sensi,V_up[i],V_down[i],CVR_up[i],CVR_down[i], factor_buckets[i]]
            df_data_add=pd.DataFrame([add],columns=['Position ID', 'Factor', 'CVR_k', 'V_up','V_down','CVR_up','CVR_down', 'Bucket'])        
            df_results_eq_curvature_factor_weighted_sensi=df_results_eq_curvature_factor_weighted_sensi.append(df_data_add)
            
        add=[position_id,risk_charge]
        df_data_add=pd.DataFrame([add],columns=['Position ID', 'Curvature Risk Charge'])        
        df_results_eq_curvature_risk_charge=df_results_eq_curvature_risk_charge.append(df_data_add)
    
    if produce_output:
        outputresults_eq_curvature_csv(df_results_eq_curvature_risk_charge,df_results_eq_curvature_factor_weighted_sensi,df_results_eq_curvature_bucket_risk_position)

    return results

############################################################################################
# Outputting results
########################################################################################
def outputresults_csr_nonsec_delta_csv(df_RC, df_WS, df_BRP):
    df_RC.to_csv('SBA CSR NonSec - Delta Risk Charge.csv')
    df_WS.to_csv('SBA CSR NonSec - Delta Weighted Sensitivities.csv')
    df_BRP.to_csv('SBA CSR NonSec - Delta Risk Position.csv')
def outputresults_girr_delta_csv():
    df_results_girr_delta_risk_charge.to_csv('SBA GIRR - Delta Risk Charge.csv')
    df_results_girr_delta_factor_weighted_sensi.to_csv('SBA GIRR - Delta Weighted Sensitivities.csv')
    df_results_girr_delta_bucket_risk_position.to_csv('SBA GIRR - Delta Risk Position.csv')
def outputresults_girr_curvature_csv(df_RC, df_FC, df_BRP):
    df_RC.to_csv('SBA GIRR - Curvature Risk Charge.csv')
    df_FC.to_csv('SBA GIRR - Curvature Sensitivities.csv')
    df_BRP.to_csv('SBA GIRR - Curvature Risk Position.csv')
def outputresults_girr_vega_csv(df_RC, df_WS, df_BRP):
    df_RC.to_csv('SBA GIRR - Vega Risk Charge.csv')
    df_WS.to_csv('SBA GIRR - Vega Sensitivities.csv')
    df_BRP.to_csv('SBA GIRR - Vega Risk Position.csv')
def outputresults_eq_vega_csv(df_RC, df_WS, df_BRP):
    df_RC.to_csv('SBA Equity - Vega Risk Charge.csv')
    df_WS.to_csv('SBA Equity - Vega Weighted Sensitivities.csv')
    df_BRP.to_csv('SBA Equity - Vega Risk Position.csv')
def outputresults_fx_delta_csv(df_RC, df_WS):
    df_RC.to_csv('SBA FX - Delta Risk Charge.csv')
    df_WS.to_csv('SBA FX - Delta Weighted Sensitivities.csv')
def outputresults_fx_vega_csv(df_RC, df_WS):
    df_RC.to_csv('SBA FX - Vega Risk Charge.csv')
    df_WS.to_csv('SBA FX - Vega Weighted Sensitivities.csv')
def outputresults_fx_curvature_csv(df_RC, df_WS):
    df_RC.to_csv('SBA FX - Curvature Risk Charge.csv')
    df_WS.to_csv('SBA FX - Curvature Sensitivities.csv')
def outputresults_eq_delta_csv(df_RC, df_WS, df_BRP):
    df_RC.to_csv('SBA Equity - Delta Risk Charge.csv')
    df_WS.to_csv('SBA Equity - Delta Weighted Sensitivities.csv')
    df_BRP.to_csv('SBA Equity - Delta Risk Position.csv')
def outputresults_eq_curvature_csv(df_RC, df_WS, df_BRP):
    df_RC.to_csv('SBA Equity - Curvature Risk Charge.csv')
    df_WS.to_csv('SBA Equity - Curvature Sensitivities.csv')
    df_BRP.to_csv('SBA Equity - Curvature Risk Position.csv')

####################################################################################################

#def outputresults_excel():
#    writer = pd.ExcelWriter('Equity_Delta.xlsx')
#    df_results_eq_delta_risk_charge.to_excel(writer,'Risk_Charge')
#    df_results_eq_delta_factor_weighted_sensi.to_excel(writer,'Weighted_Sensi')
#    df_results_eq_delta_bucket_risk_position.to_excel(writer,'Bucket_Risk_Position')
#    
#    writer.save()

#def outputresults_excel():
#    writer = pd.ExcelWriter('GIRR_Delta.xlsx')
#    df_results_risk_charge.to_excel(writer,'Risk_Charge')
#    df_results_factor_weighted_sensi.to_excel(writer,'Weighted_Sensi')
#    df_results_bucket_risk_position.to_excel(writer,'Bucket_Risk_Position')
#    writer.save()

df_input_mapped.to_clipboard()
