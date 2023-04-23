
# coding: utf-8

# This is for Houselisting and Housing Census data - 2011

# http://www.censusindia.gov.in/2011census/hlo/HLO_Tables.html

# This combines ds for all 6 states : data preprocessing step.

# This is all about different asset indicators
# In[2]:

import pandas as pd
import glob


# In[55]:

list=glob.glob('../rawdata/*/*.xlsx')
dsl=[]
for path in list:
    dsl.append(pd.read_excel(path,skiprows=6,converters={1:lambda x:str(x),3:lambda x:str(x),5: lambda x:str(x),
                                                         7: lambda x: str(x), 8:lambda x:str(x)}))
#merge all
ds=pd.concat(dsl)

# In[57]:

ds_out=pd.DataFrame(ds.iloc[:,:10],copy=True)
ds_out.columns=['state-code','state','district-code','district-name','teh-code','teh-name','vill-code','ward-no','area-name','rural/urban']


# In[58]:

ds_out['res-good']=ds[12]
ds_out['res-bad']=ds[14]
ds_out['rooms-under-3']=ds[49]+ds[50]+ds[51]
ds_out['rooms-over-3']=ds[52]+ds[53]+ds[54]+ds[55]
ds_out['household-size-under-5']=ds[56]+ds[57]+ds[58]+ds[59]
ds_out['household-size-over-5']=ds[60]+ds[61]+ds[62]
#ownership status
ds_out['owned']=ds[63]
ds_out['not-owned']=ds[64]+ds[65]
#water facilites
ds_out['water-treated']=ds[72]+ds[74]+ds[77] #tapwater from treated source + covered well + tubewell
ds_out['water-untreated']=ds[73]+ds[75] #tapwater from untreated source + uncovered well
ds_out['water-natural']=ds[76]+ds[78]+ds[79]+ds[80]+ds[81]
#Lighting
ds_out['electric-like']=ds[85]+ds[87]
ds_out['oil-like']=ds[86]+ds[88]+ds[89]
ds_out['no-lighting']=ds[90]
#Asset Ownership
ds_out['electronics']=(ds[128]+ds[129]+ds[130]+ds[131])/3 #radio/transistor+tv+laptop-net+laptop-without-net
ds_out['has-phone']=ds[132]+ds[133]+ds[134] #landline+mobile+both
ds_out['transport-cycle']=ds[135]
ds_out['transport-motorized']=(ds[136]+ds[137])/2 #scooter + car
ds_out['no-assets']=ds[139] #% of households with none of the assets available
#Financial services
ds_out['banking-services-availability']=ds[127]
#Cooking fuel
ds_out['cook-fuel-natural']=ds[109]+ds[110]+ds[111]+ds[112]+ds[117]
ds_out['cook-fuel-processed']=ds[113]+ds[114]+ds[115]+ds[116]
ds_out['no-cooking']=ds[118]
#Bathroom
ds_out['bathroom-within']=ds[103]+ds[104]
ds_out['bathroom-outside']=ds[105]
#permanent house
ds_out['permanent-house']=ds[140]
ds_out['non-permanent-house']=ds[141]+ds[142] #semi permanent and temporary
#village id creation
ds_out['villageid']=ds[1]+ds[3]+ds[5]+ds[7]

# Some rows in the dataset have same villagecode.
# {VillageCode:000000} - means summary of that district.
# {TehsilCode:99999} - area not under any subdistrict
# These are to removed as they have no ward wise boundaries.

# In[59]:

ds_out=ds_out.drop_duplicates(subset='villageid',keep=False)


# In[ ]:
ds_out.set_index('villageid',inplace=True)

ds.to_csv('HL_6states_raw.csv',encoding='utf-8')
ds_out.to_csv('HL_6states_indic.csv',encoding='utf-8')

