#!/usr/bin/env python
# coding: utf-8

# # DEEM SKELETON

# In[ ]:


from bisect import bisect_left
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")
#RMSE 
from datetime import datetime


def DEEM(Sub_Daily, Sub_Daily_Training,Daily_Extreme_Training, Trials, Corr_Stop):
    '''
    Parameters
    --------------
    Sub_Daily : DataFrame
        This is the raw subdaily data you aim to estimate the maximum and minimum temperatures from.
    
    Sub_Daily_Training : DataFrame
        A list with the date and temp as 2 columns and index going from 0,1...X. All values are subdaily so they have hours
        associated with them also time is in 24 hour format.
        
    Daily_Extreme_Training : DataFrame
        A list with the date and temp as 3 columns and index going from 0,1...X. All values are daily with max and min
        associated with them also time is in 24 hour format.
     
    Trials : Integer
        The number of trails you want to run the estimation training over.
        
    Returns
    -------------
    Max : DataFrame
        X trials of Maximum temperature estimates
        
    MaxCorr : DataFrame
        X trails of the Spearman Rank Correlation assigned to the Maximum Temperature associated with the subdaily
    
    Min : DataFrame
        X trials of Minimum temperature estimates
        
    MinCorr : DataFrame
        X trails of the Spearman Rank Correlation assigned to the Minimum Temperature associated with the subdaily


    '''
    
    
    
    # Part 1: Split the Sub_Daily Training into individual hours ane combine
    Sub_Max, Sub_Min, Hours_Avaliable = Sub_Daily_Splitter(Sub_Daily_Training)

    # Part 2: Concat the Maximum and Minimum Data to the subdaily data
    Sub_Ext_Max, Sub_Ext_Min = concat_des_to_sub(Sub_Max, Sub_Min, Hours_Avaliable, Daily_Extreme_Training)

    #Now Every Single Available hour and max and min is ready to be used.
    #Part 3: Split into each respective Month and add all together so its like Month_Hour_Mx/Mn
    Monthly_Split_Dic = Month_Splitter(Hours_Avaliable,Sub_Ext_Max, Sub_Ext_Min)

    #Include 24 in the hours avalaible, this is to get it back to 0
    Hours_Avaliable_Inc_24 = Hours_Avaliable.copy()
    Hours_Avaliable_Inc_24.append(24)
    
    #PART 4 Is to fix up the Historical Data so it is closest to the every hour hour mark where data is avaliable
    Sub_Daily = Closest_Hour(Sub_Daily, Hours_Avaliable_Inc_24)

    #PART 5 Is to sample by the length of the number of datapoints for that month and max or min
    #Now I need to select 600 points and trail it 1000 times for each single thing in the dictionary and label the hour 0 as hour 0 run 1]
    #and PRO Max Run 1
    Sampled = Sampler_Trainer(Monthly_Split_Dic,Trials)

    #Part 6
    #Now to apply the regression anaylsis onto the data I have provide
    Linear_Analysis = Linear_Regression_Equations(Trials, Hours_Avaliable, Sampled)

    #Part 7    
    #Get the data into their respective max and min with the hours matching the regression data, look at the explabations
    #above in Part 2 and Part 7 for more information
    Max_Data = Max_Sub(Sub_Daily)
    Min_Data= Min_Sub(Sub_Daily)

    #Part 8 Temperature Estimation
    Full_Temperature_Estimation= Tmax_Tmin_All_Data_Est(Trials, Max_Data, Min_Data, Linear_Analysis)

    #Part 9 The Best Temperature Estimation
    Temperature_Estimation = Absolute_Estimation(Full_Temperature_Estimation, Trials, Corr_Stop)

    #Part 10. Adding all into DataFrames (not dictionaries)
    Max, MaxCorr, Min, MinCorr= Cleansing_Data(Temperature_Estimation)

    
    
    return(Max.reset_index(), MaxCorr.reset_index(), Min.reset_index(), MinCorr.reset_index())




# # PART 1 SUB DAILY SPLITTER FUNCTION

# In[ ]:


def Sub_Daily_Splitter(Data):
    '''
    Parameters
    --------------
    
    Data : DataFrame
        A list with the date and temp as 2 columns and index going from 0,1...X. All values are subdaily so they have hours
        associated with them.
        
    Return
    ------------
    Sub_Max : Dictionary/DataFrames
        The respective hours and the shifts to fit the regression and tmax calculation like the BOM has done is
    Sub_Min : Dictionary/DataFrames
        The respective hours and the shifts to fit the regression and tmin calculation like the BOM has done is
    Hours_Avaliable : Array
        All the hours that have at least 10 years worth of data
        
    '''
    #Set datetime to date
    Data_Col = Data.columns
    Data = Data.set_index(Data_Col[0])
    
    #We need the hours in 24 hour format as a list.
    Every_Hour = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    
    #Now to create the dictionaries necessary for the splitting.
    Sub_Hourly_Dic = {}
    # Hours_Avaliable is different to Every_Hour as it finds hours that have data with at least 10 years worth of data.
    Hours_Avaliable = []
    
    #Begin the for loop
    for HOUR in Every_Hour:
              
        #Locate all the data for that hour
        Single_Hour_Data = pd.concat([Data[Data.index.hour==HOUR]],axis =0)
        
        #Now to check if the data has at least 10 years worth of Data
        if (len(Single_Hour_Data) >= 3600):
            #If it is then append it into the dictionary
            
            #reset the index to fix the datetime
            Single_Hour_Data = Single_Hour_Data.reset_index()
            #Make sure that datetime is still on the date we used
            Single_Hour_Data[Data_Col[0]] = pd.to_datetime(Single_Hour_Data[Data_Col[0]]).dt.date
            #Set Index back to date, but maybe not if we need to make it for Max and Min
            Single_Hour_Data = Single_Hour_Data.set_index(Data_Col[0]).dropna()
            
            #Add to Dictionaries and columns
            Sub_Hourly_Dic["Hour" +"_"+ str(HOUR)] = Single_Hour_Data
            
            #This becomes useful in the next section where we get information relative to max and min temperatures.
            Hours_Avaliable.append(HOUR)
        
        
    #Now to split the data into the respective Max and Min dictionaries.

    
    '''
    This is a bit of explanation about the choices I make with what horus I choose for this.
    So in my previous versions of creating this function I had to choose the times of when I can locate
    the Tmax and Tmin from. Now what noticed was firstly on the day (+0) the Tmax was generally found between 
    12pm+0 to 6pm+0. So this meant the likelyhood that the max was between 9am+0 to 9am+1. This actually aligns with what 
    times the Tmax is found between 9am+0 to 9am+1. This means that for this we need to shift the hour values of 0am+1 to 8am+1
    to be used on this particular day.
    
    Now there is a trickier part. It is the tmin. Like the tmax, the tmin is calulcated by the BOM from 9am-1 to 9am+0. However
    within my findings it turns out that the correlation at 9am-1 is much lower then at 9am+0, furthermore the afetrnoon of the 
    prvious day has a higher correlation to what the min will be the next day than the correlation of the day in focus.
    This has me belive that the min for the day in focus is influenced much more highly by the temperatures of the previous day
    then the temperature of the day in focus which I will go into further discussion later and read about on papers becasue this 
    is an interesting debate. But to estimate temperature more of this will be explained and explored.
    
    For now, I came to the conculsion that the tmin will be estimated by the 10am-1 to 10am+0 to account for the 9am+0 higher 
    correlation. For tmax it will be like the BOM standard 9am+0 to 9am+1
    '''
    
    #Create the dictionaries for max and min
    Sub_Max = {}
    Sub_Min = {}
    
    #Now for loop with the hours we do have
    for HOURS in Hours_Avaliable:
        #Lets shift the hours
        #Since we know the key was "Hour_HOURS"
        #Extract the DataFrame for that specific hour
        Hourly_Data =  Sub_Hourly_Dic.get('Hour_{}'.format(HOURS))
        
        #MAX
        #Remember 0+0 to 8+1
        if (HOURS in range(0,9)):
            #Shift it negative one which means everything is pushed up, so tomorrows temp is now are todays hour.
            Shift_Max = Hourly_Data.shift(-1, axis = 0).dropna()
            #Append it to max dictionary
            Sub_Max["Hour" +"_"+ str(HOURS)+"+1"] =Shift_Max
        else:
            Shift_Max = Hourly_Data
            #Append it to max dictionary
            Sub_Max["Hour" +"_"+ str(HOURS)+"+0"] = Shift_Max
            
            
        #Min
    
        #Remember 9-1 to 8+0 : IN LINE WITH BOM STANDARDS, SHOULDNT MESS WITH IT

        if (HOURS in range(10,23)):
            #Shift it positive one which means everything is pushed down, so yesterdays temp is now are todays temp.
            Shift_Min = Hourly_Data.shift(1, axis = 0).dropna()
            #Append it to min dictionary
            Sub_Min["Hour" +"_"+ str(HOURS)+"-1"] =Shift_Min
        else:
            Shift_Min = Hourly_Data
            #Append it to max dictionary
            Sub_Min["Hour" +"_"+ str(HOURS)+"+0"] = Shift_Min
    
    
    
    return(Sub_Max, Sub_Min, Hours_Avaliable)
    


# # PART 2 CONCATINATION DEs To Sub-dailys

# In[ ]:


def concat_des_to_sub(Sub_Max,Sub_Min,Hours_Avaliable, DE_values):
    '''
    Parameters
    --------------
    Sub_Max : Dictionary/DataFrames
        The respective hours and the shifts to fit the regression and tmax calculation like the BOM has done is
    Sub_Min : Dictionary/DataFrames
        The respective hours and the shifts to fit the regression and tmin calculation like the BOM has done is
    Hours_Avaliable : Array
        All the hours that have at least 10 years worth of data
    DE_values : DataFrame
        A list with the date and temp as 3 columns and index going from 0,1...X. All values are daily with max and min
        associated with them also time is in 24 hour format.
    
    Return
    ------------
    Sub_Mx : Dictionary/DataFrame
        A dictionary of many dataframes that associate the Tmax with the subdaily values of that day
    
    Sub_Mn : Dictionary/DataFrame
        A dictionary of many dataframes that associate the Tmin with the subdaily values of that day
    '''
    DE_values_col = DE_values.columns
    DE_values = DE_values.set_index(DE_values_col[0])
    
    
    #Create the and Min dictionaries
    Sub_Mx = {}
    Sub_Mn = {}
    
    #Extract the max and min keys
    Keys_Mx = list(Sub_Max)
    Keys_Mn = list(Sub_Min)

    #Go with Tmax
    for i in range(len(Keys_Mx)):
        #Extract the subdaily data for that hour
        Mx_Sub = Sub_Max.get(Keys_Mx[i])
        #Combine with Tmax where datetime si the joiner
        Combined_Train_Mx = pd.merge(left = Mx_Sub, 
                                        right  =DE_values[DE_values_col[1]],
                                        left_index=True,right_index=True  )
        #Rename to Max
        Combined_Train_Mx = Combined_Train_Mx.rename(columns={DE_values_col[1]:'Max'})
        #Append to dictioanry 
        Sub_Mx["Hour" +"_"+ str(Hours_Avaliable[i])] = Combined_Train_Mx
        
    #Min follow similar as Max
    for j in range(len(Keys_Mn)):
        #Extract the subdaily data for that hour
        Mn_Sub = Sub_Min.get(Keys_Mn[j])
        #Combine with Tmax where datetime si the joiner
        Combined_Train_Mn = pd.merge(left = Mn_Sub, 
                                        right  =DE_values[DE_values_col[2]],
                                        left_index=True,right_index=True  )
        Combined_Train_Mn = Combined_Train_Mn.rename(columns={DE_values_col[2]:'Min'})
        #Append to dictioanry 
        Sub_Mn["Hour" +"_"+ str(Hours_Avaliable[j])] = Combined_Train_Mn
        
        
    return(Sub_Mx,Sub_Mn)


# # PART 3 Split into months

# In[ ]:


def Month_Splitter(Hours_Avaliable,Sub_Ext_Max, Sub_Ext_Min):
    '''
    Parameters
    --------------
    Hours_Avaliable : Array
        All the hours that have at least 10 years worth of data
    Sub_Mx : Dictionary/DataFrame
        A dictionary of many dataframes that associate the Tmax with the subdaily values of that day
    Sub_Mn : Dictionary/DataFrame
        A dictionary of many dataframes that associate the Tmin with the subdaily values of that day
        
    Return
    ------------
    Monthly_Split_Dic : Dictionary/DataFrame
        A dictionary that has the data splkit into month and hours
    
    '''
    #Lets get all the monthly arrays sorted   
    Month_Number = [1,2,3,4,5,6,7,8,9,10,11,12]
    Month_Name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    #Now lets create a monthly dictionary that the subdaily and its associated tmax/tmin can go in
    Monthly_Split_Dic = {}
    
    #Lets begin the loop to extract the subdaily find the hours and append it all together
    for i in Hours_Avaliable:
        #Extract Max and Min DataFrames, as of the other function we know what the key is for the dictionary
        Max_Data = Sub_Ext_Max.get('Hour_{}'.format(i))
        Min_Data = Sub_Ext_Min.get('Hour_{}'.format(i))
        
        #Extract the Month Number and extract the data for that month
        for q in range(len(Month_Number)):
            #Get the data for the month only
            Month_Max_Data = pd.concat([Max_Data[Max_Data.index.month==Month_Number[q]],], axis = 0)
            Month_Min_Data = pd.concat([Min_Data[Min_Data.index.month==Month_Number[q]],], axis = 0)
            #Add to Dictionary
            Monthly_Split_Dic[Month_Name[q] +"_"+ str(i) + "_"+"Mx"] = Month_Max_Data
            Monthly_Split_Dic[Month_Name[q] +"_"+ str(i) +"_"+ "Mn"] = Month_Min_Data
  

    return(Monthly_Split_Dic)


# # Part 4 Closest Hour Functions

# In[ ]:


def Closest_Hour(Data, hours): 
    '''
    Parameters
    -------------
    Data: DataFrame
        The sub_daily data we are aiming to estimate the Tmax and Tmin from wtih two columns, one with datetime and the 
        other with DataFrame
    
    hours:
    The hours that are avaliable to use to get the data from this Data dataset as close as possible to the trained hours
    as some hours may not be able to be used.
    
    Returns
    -----------------
    Dataset : DataFrame
        A dataset that has the closest hour to one of the avalaible hours in the dataset.
        
    '''
    #Get a new array
    closest_hour= []
    
    #We want to match to the hour closest to the 3

    for i in range(len(Data)):
        #Extract the single day
        Individual_Day = Data.loc[i]
    
        #Extract hour
        Individual_Hour = Individual_Day['date'].hour
        
        #Take the closest hour
        Closest_Ind_Hour = take_closest(hours, Individual_Hour)
        
        #If closest hour is 24, make sure it takes the closest hour on either side with the 23 and lower being favoured
        if (Closest_Ind_Hour == 24):
            Left_Check= abs(24 - hours[len(hours)-2])
            Right_Check = abs(hours[0]-0)
            
            
            if Left_Check > Right_Check:
                Closest_Ind_Hour = hours[0]
            else:
                Closest_Ind_Hour = hours[len(hours)-2]
            
        
        #Append the closest hour 
        closest_hour.append(Closest_Ind_Hour)
    
    #Add it as a series then combine to make it a dataframe
    CL = pd.Series(closest_hour, name = 'Closest Hour')
    Dataset = pd.merge(left = Data,right  =CL,left_index=True,right_index=True  )
    return(Dataset)


# In[17]:


def take_closest(myList, myNumber): 
    """
    Parameter
    --------------
    
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    
    myList: 
        The values that the data can be closest to
    
    myNumber:
        The raw value that will then be converted to the Closest Hour
        
    Returns
    ---------------
    after/before : Integer
        Value that the hour can be closest to
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before



# # PART 5 Training By Bootstrapping Sampling

# In[ ]:


def Sampler_Trainer(Data,Trials):
    '''
    Parameters
    -------------
    Data : DataFrame/Dictionary
        A dictionary that has the data splkit into month and hours
        
    Trials : Integer
        The number of trails you want to run the estimation training over.

    Returns
    -----------
    Samples : DataFrame/Dictionary
        Using the observations and the training data we can have created a dictionary of DataFrames
        that have trialed that have been sampled by the lenght of the data avalaible for that month.
        
    
    '''
    
    #Now I need to select random samples from the length of the data for each month, hour and mx or mn
    #then do this 1000 times and label them in the columns Hour 0 as Hour 0 Run 1 and Max as Max Run 1.
    
    
    #Create the dictionary that all the data will be inputed to
    Samples = {}
    
    #Get the entire key column
    Keys = list(Data)
    
    #Now extract the DataFrame from the dictionary for the Key
    for keys_used in Keys:
        #Extract and drop NaNs
        Ind_DF = Data.get(keys_used).dropna()
        #Now sample by the length fo the DataFrame and this is done for the first run only 
        Run1_Data = Ind_DF.sample(n=int(len(Ind_DF)),replace=True)
        #Drop the date column with Index is 0 to Samples-1
        Run1_Data = Run1_Data.reset_index(drop = True)

        #Get the columns names
        Col = Run1_Data.columns
        
        #Now change column name to make it run 1 etc
        Run1_Data= Run1_Data.rename(columns={Col[0]:Col[0] + ' ' +  'Run 1'})
        Run_Data= Run1_Data.rename(columns={Col[1]:Col[1] + ' ' +  'Run 1'})
        
        #Now develope the for loop but the trials is based off the lenght of Data
        for rns in range(2,Trials+1):
            #This is the now the random sampling for 1000 different samples of 600 
            Individual_Run = Ind_DF.sample(n=int(len(Ind_DF)),replace=True)
            #Drop the date column
            Individual_Run = Individual_Run.reset_index(drop = True)
            
            #Get the columns names
            Col = Individual_Run.columns
        
            #Now change column name to make it run 1 etc
            Individual_Run= Individual_Run.rename(columns={Col[0]:Col[0] + ' ' +  'Run {}'.format(rns)})
            Individual_Run= Individual_Run.rename(columns={Col[1]:Col[1] + ' ' +  'Run {}'.format(rns)})
        
            #Concate with RUNS
            Run_Data = pd.concat([Run_Data, Individual_Run],axis=1)
            
        #Now add this to a new dictionary
        Samples[keys_used + "_" + "Samp"] = Run_Data
        
    return(Samples)


# # PART 6 The Regression Equations Functions

# In[ ]:


def Linear_Regression_Equations(Trials, hours, Data):
    '''
    Parameters
    --------------
    Trials : Integer
        The number of trails you want to run the estimation training over.
        
    hours : array
    
    Data : DataFrame/Dictionary
        Using the observations and the training data we can have created a dictionary of DataFrames
        that have trialed that have been sampled by the lenght of the data avalaible for that month.
    
    Returns
    --------------
    Regressed_Trial : Dictionary/DataFrames
        For each trial, the set of equations for each sub-daily time, Trained DEs and Month will be generated for application 
        to estimate the DEs of the inputted data for estimation.
    '''
    
    #Create dictionaries
    Regressed_Trial = {}
    
    #Define the month names
    Month_Name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    #Being the for loop by extracting the month name
    for month_num in range(0,12):
        #Extract the month name_
        Month_Str =  Month_Name[month_num]
        #This is useful in the key as it is Aug_9_Mn_Samp where month_hour_mx/mn_samp
        
        #Now using the trials lets extract the trials within the particular dictionary
        for trial_number in range(1,Trials+1):
            #Now this is all the arrays that will be appended to at the end that include the linear
            #regression line components, A and B and the Correlation by the spearman r
            AMx_Total = []
            BMx_Total = []
            CORRMx_Total = []
            Time = []
            AMn_Total = []
            BMn_Total = []
            CORRMn_Total = []
            
            #Now for loop to extract the data and get the regression
            for i in hours:
                #---MAX---#
                #Extract the maximum data
                Mxt = Data.get('{}_{}_Mx_Samp'.format(Month_Str,i))
                #Get the linear formula and the correlation of the data
                AMx, BMx, corrMx = linear_regression_polyfit(Mxt['temp Run {}'.format(trial_number)],Mxt['Max Run {}'.format(trial_number)])
                #Append it all
                AMx_Total.append(AMx)
                BMx_Total.append(BMx)
                CORRMx_Total.append(corrMx)
                #Repeat for min
                #---MIN---#
                Mnt = Data.get('{}_{}_Mn_Samp'.format(Month_Str,i))
                AMn, BMn, corrMn = linear_regression_polyfit(Mnt['temp Run {}'.format(trial_number)],Mnt['Min Run {}'.format(trial_number)])
                Time.append(int(i)) 
                AMn_Total.append(AMn)
                BMn_Total.append(BMn)
                CORRMn_Total.append(corrMn)

            #Add it all into a dataframe
            Time = pd.Series(Time,name = 'Hours')
            
            AMX = pd.Series(AMx_Total,name = 'A')
            BMX = pd.Series(BMx_Total,name = 'B')
            corrMX = pd.Series(CORRMx_Total,name = 'Correlation')
            ItemsMX = pd.concat([Time,AMX,BMX,corrMX],axis = 1)
            
            AMN = pd.Series(AMn_Total,name = 'A')
            BMN = pd.Series(BMn_Total,name = 'B')
            corrMN = pd.Series(CORRMn_Total,name = 'Correlation')
            ItemsMN = pd.concat([Time,AMN,BMN,corrMN],axis = 1)
            
            Regressed_Trial["{}".format(Month_Str) + "_" + 'Trial'+ "_" + str(trial_number) + "_" + "Mx"] = ItemsMX
            Regressed_Trial["{}".format(Month_Str) + "_" + 'Trial'+ "_" + str(trial_number) + "_" + "Mn"] = ItemsMN
    return(Regressed_Trial)




#Now develop the linear regression equation
def linear_regression_polyfit(x,y):
    #Find the linear Relationship
    A, B = np.polyfit(x, y, 1)
    #Find the correlation                  
    corr, _ = spearmanr(x, y)
    return(A,B,corr)



# # PART 7 Shifting Sub-dailies for Max and Min

# In[ ]:


def Max_Sub(Data):
    '''
    Parameters
    -------------
    Data : DataFrame
        The unshifted inputted sub-hourly data as it does not match the Max time period, of [9am+0, 9am+1)
    Returns
    -------------
    Sub_Daily_Data : DataFrame
        The shifted inputted sub-hourly data to match the Tmax time period, of [9am+0, 9am+1)
    '''
    Sub_Daily_Data = Data.copy()
    #Get the estimation sorted
    # Shift hours 0 to 8 to the previous day's hours for maximum regression of 9am+0 to 8am+1
    Sub_Daily_Data['date'] = pd.to_datetime(Sub_Daily_Data['date'])
    Sub_Daily_Data.loc[Sub_Daily_Data['date'].dt.hour < 9, 'date'] = Sub_Daily_Data['date'] - pd.offsets.Day(1)
    Sub_Daily_Data['date'] = Sub_Daily_Data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return(Sub_Daily_Data)


def Min_Sub(Data):
    '''
    Parameters
    -------------
    Data : DataFrame
        The unshifted inputted sub-hourly data as it does not match the Min time period, of [10am-1, 10am+0)
    Returns
    -------------
    Sub_Daily_Data : DataFrame
        The shifted inputted sub-hourly data to match the Tmax time period, of [10am-1, 10am+0)
    '''
    Sub_Daily_Data = Data.copy()
    # Shift hours 10 to 23 to the tomorrows day's hours for minimum regression of 10am-1 to 9am+0
    Sub_Daily_Data['date'] = pd.to_datetime(Sub_Daily_Data['date'])
    Sub_Daily_Data.loc[Sub_Daily_Data['date'].dt.hour > 9, 'date'] = Sub_Daily_Data['date'] + pd.offsets.Day(1)
    Sub_Daily_Data['date'] = Sub_Daily_Data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')


    return(Sub_Daily_Data)


# # PART 8 Estimating DEs for all inputted sub-daily times Functions

# In[ ]:


def Tmax_Tmin_All_Data_Est(Trials, Historical_Max, Historical_Min, Linear):
    '''
    Parameters
    --------------
    Trials : Integer
        Number of trials that have been used in this estimation
    
    Historical : DataFrame
        The dataset we will estimate the tmax and tmin temperatures from this already should be in a good format
    
    Linear : Dictionary/DataFrame
        The dictionary with all the linear regressed data for each trial ready to be applied onto the 
    

    Returns
    --------
    All_Data_Est : Dictionary/DataFrames
        All inputted sub-daily data have been associated with a DE estimation value and a spearman rank correlation 
        for all X trials
    
    '''
    
    #Now we begin with the final dictionary
    #Write a all data didctionary
    
    All_Data_Est = {}
    
    #Columns for data 
    Historical_Max_Col = Historical_Max.columns
    Historical_Min_Col = Historical_Min.columns
    
    #Lets begin with the for loop for the trials of the linear
    for T in range(1,Trials+1):
        print(T)
        #Set all the arrays for the information to be added into it
        Est_Max = []
        Est_Min = []
        Max_Corr = []
        Min_Corr = []
        
        
        #Now lets begin with Mx
        for indexed in range(len(Historical_Max)):
            #Extract the initial data
            Day_Data_Max = Historical_Max.loc[indexed]
            
            
            
            #Extract these values : closest hour, month, temp
            Month_V_Max = datetime.strptime(Day_Data_Max[Historical_Max_Col[0]], '%Y-%m-%d %H:%M:%S').month
            Hour_Max = Day_Data_Max[Historical_Max_Col[2]]
            Temperature_Max = Day_Data_Max[Historical_Max_Col[1]]
            
            #Now using another function we can sift through the trial and month to find the estimation,
            Mx_Temp, Corr_Mx = The_Estimator(Month_V_Max, Hour_Max, Temperature_Max, Linear, T, True)
        
            Est_Max.append(Mx_Temp)
            Max_Corr.append(Corr_Mx)
        
        
        #Add the data to the dates again
        Est_Max = pd.Series(Est_Max,name = 'Max Temp Estimation')
        Max_Corr = pd.Series(Max_Corr,name = 'Correlation Max T')
        
        Dataset_Max = pd.concat([Historical_Max, Est_Max, Max_Corr],axis=1)
        
        #Now lets begin with Mx
        for indexed in range(len(Historical_Min)):
            #Extract the initial data
            Day_Data_Min = Historical_Min.loc[indexed]
            
            #Extract these values : closest hour, month, temp
            Month_V_Min = datetime.strptime(Day_Data_Min[Historical_Min_Col[0]], '%Y-%m-%d %H:%M:%S').month
            Hour_Min = Day_Data_Min[Historical_Min_Col[2]]
            Temperature_Min = Day_Data_Min[Historical_Min_Col[1]]
            
            #Now using another function we can sift through the trial and month to find the estimation,
            Mn_Temp, Corr_Mn = The_Estimator(Month_V_Min, Hour_Min, Temperature_Min, Linear, T, False)
        
            Est_Min.append(Mn_Temp)
            Min_Corr.append(Corr_Mn)
        
        
        #Add the data to the dates again
        Est_Min = pd.Series(Est_Min,name = 'Min Temp Estimation')
        Min_Corr = pd.Series(Min_Corr,name = 'Correlation Min T')
        
        Dataset_Min = pd.concat([Historical_Min, Est_Min, Min_Corr],axis=1)
        
        
        #Add to a Trial Dictionary
        All_Data_Est['Trial' + '_' + str(T) + "_Mx"] = Dataset_Max
        All_Data_Est['Trial' + '_' + str(T) + "_Mn"] = Dataset_Min
            

        
    return(All_Data_Est)

def The_Estimator(MONTH, Hour, Temp, DATA_4_EST, Trial_Number, Max):
    
    '''
    Parameters
    --------------
    MONTH : Integer
        Month which the inputted sub-daily value is taken from
        
    Hour : Integer
        Hour when the inputted sub-daily temperature was taken from

    Temp : Decimal
        Temperature of the inputted sub-daily data
        
    DATA_4_EST : Dictionary
        The X trial sampled bootstrapping regression equations for which are used to extracted the MONTH, Hour, Temp 
        and the specific Trial_Number from
    
    Trial_Number : Integer
        The trial used to find the linear regression equation that the DE for that MONTH, Hour and Temp
        will be used
    
    Max : True/False
        Determine whether the function utilises the Max or Min DE estimation
    
    
    Returns
    --------
    Est_Max : Decimal
        The estimated Tmax from the linear regression equation for that Hour and MONTH and sub-daily temperature
    
    Corr_Max : Decimal
        The associated correlation from the sub-daily vs tmax for that Hour and MONTH
    
    Est_Min : Decimal
        The estimated Tmin from the linear regression equation for that Hour and MONTH and sub-daily temperature
    
    Corr_Min : Decimal
        The associated correlation from the sub-daily vs tmin for that Hour and MONTH

    '''
    
    Month_Name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    
    if (Max == True):
        #Estimate the Max Temp
        #Extract data from the linear regression dictionary
        Info = DATA_4_EST.get('{}_Trial_{}_Mx'.format(Month_Name[MONTH-1],Trial_Number))
        Info = Info.set_index('Hours')
        Info = Info.loc[int(Hour)]
        #Estimate the Max based off this information
        Est_Max = Info['A']*Temp + (Info['B'])
        #Find the Corr for this day and hour
        Corr_Max =  Info['Correlation']
        return(Est_Max,Corr_Max)
    else:
        #Estimate the Min Temp
        #Extract data from the linear regression dictionary
        Info = DATA_4_EST.get('{}_Trial_{}_Mn'.format(Month_Name[MONTH-1],Trial_Number))
        Info = Info.set_index('Hours')
        Info = Info.loc[int(Hour)]        #Estimate the Max based off this information
        Est_Min = Info['A']*Temp + (Info['B'])
        #Find the Corr for this day and hour
        Corr_Min =  Info["Correlation"]
        return(Est_Min,Corr_Min)


# # PART 9 FINDING THE APPROPIATE DEs

# In[ ]:


def Absolute_Estimation(Estimated_Data, Trials, Corr_Stop):
    
    '''
    Parameters
    --------------
    Estimated_Data : Dictionary/DataFrames
        The inputted sub-daily data associated with a DE estimation value and a spearman rank correlation 
        for all X trials
    
    Trials : Integer
        The number of Trials used in DEEM (X)
        
    Corr_Stop : Value
        This is for the choice model to choose a threshold that stops the chance for low correlated DEs to be used 
        as the final DEs.
    
    Returns
    --------------
    Est_Daily_Extremes : Dictionary/DataFrames
        The entirety of the estimated DEs with their associated spearmam rank correlations for all X trials
    
    '''
    #We need a new dictionary for the finalised estimation
    Est_Daily_Extremes = {}
    

    #Lets begin by using a for loop that extracts that Trail number and the indivudal max and min estimations
    for T in range(1,Trials+1):
        print(T)

        #Extract the data
        Max_Data = Estimated_Data.get('Trial_{}_Mx'.format(T))
        Min_Data = Estimated_Data.get('Trial_{}_Mn'.format(T))
    
        #Lets extract the columns as well, this will be useful.
        #Get Columns
        Max_C = Max_Data.columns
        Min_C = Min_Data.columns
    
        #Make the data datetime 
        #Convert date to datetime
        Max_Data[Max_C[0]] = pd.to_datetime(Max_Data[Max_C[0]])
        Min_Data[Min_C[0]] = pd.to_datetime(Min_Data[Min_C[0]])
    
    
        #Delete the hour out of the date
        Max_Data[Max_C[0]] = Max_Data[Max_C[0]].dt.date 
        Min_Data[Min_C[0]] = Min_Data[Min_C[0]].dt.date 
        
    
    
        #Now we want to see only the individual dates only
        Unique_dates_Max = Max_Data[[Max_C[0]]].drop_duplicates()
        Unique_dates_Max =  Unique_dates_Max.reset_index(drop = True)
        Unique_dates_Min = Min_Data[[Min_C[0]]].drop_duplicates()
        Unique_dates_Min =  Unique_dates_Min.reset_index(drop = True)
        
        #Redo datetime because for some reason when removing the hour it resets the date
        Max_Data[Max_C[0]] = pd.to_datetime(Max_Data[Max_C[0]])
        Min_Data[Min_C[0]] = pd.to_datetime(Min_Data[Min_C[0]])
    
        #Now we have the necessary data for estimated for a single day.
        #Now define the vectors for the Max, Min, Max_Corr, and Min_Corr
        Tmax = []
        Corr_Max = []
        Tmin = []
        Corr_Min = []
        Dates_Mx = []
        Dates_Mn = []
        #Now go through the max and min and choose the best value either in a simple or complex case
        #Max
        for i in range(len(Unique_dates_Max)):
            #Get the individual date
            loc_date_Mx = Max_Data.loc[Max_Data[Max_C[0]] == '{}-{}-{}'.format(Unique_dates_Max[Max_C[0]][i].year,Unique_dates_Max[Max_C[0]][i].month,Unique_dates_Max[Max_C[0]][i].day)]
            #iT is in its length, the 1 length data is remaining with the index as the row
            #from here we will then select either complex or simple and then go into another function.
            Max_Est,Max_Corr = Choice_Model(loc_date_Mx, True, Corr_Stop)
        
            Tmax.append(Max_Est)
            Corr_Max.append(Max_Corr)
    
    
        #Min
        for i in range(len(Unique_dates_Min)):
            #Get the individual date
            loc_date_Mn = Min_Data.loc[Min_Data[Min_C[0]] == '{}-{}-{}'.format(Unique_dates_Min[Min_C[0]][i].year,Unique_dates_Min[Min_C[0]][i].month,Unique_dates_Min[Min_C[0]][i].day)]
            #iT is in its length, the 1 length data is remaining with the index as the row
            #from here we will then select either complex or simple and then go into another function.
            Min_Est,Min_Corr = Choice_Model(loc_date_Mn, False, Corr_Stop)
            Tmin.append(Min_Est)
            Corr_Min.append(Min_Corr)
        
        Tmax_A = pd.Series(Tmax,name = 'Max Temp Estimation')
        Tmin_A = pd.Series(Tmin,name = 'Min Temp Estimation')
        Corr_Max_A = pd.Series(Corr_Max,name = 'Correlation Max T')
        Corr_Min_A = pd.Series(Corr_Min,name = 'Correlation Min T')
        Estimated_Temp_Max = pd.concat([Unique_dates_Max, Tmax_A,Corr_Max_A],axis=1)
        Estimated_Temp_Min = pd.concat([Unique_dates_Min, Tmin_A,Corr_Min_A],axis=1)
        
        #Now add it to 1 single DataFrame
        Estimated_Merge = pd.merge(Estimated_Temp_Max, Estimated_Temp_Min, on=Max_C[0], how='outer')
        
        # Create a date range with missing dates
        start_date = str(Estimated_Merge[Max_C[0]][0])
        end_date = str(Estimated_Merge[Max_C[0]][len(Estimated_Merge)-1])
        date_range = pd.date_range(start=start_date, end=end_date)
        # Create a DataFrame with the missing dates
        missing_dates_df = pd.DataFrame({Max_C[0]: date_range})
        
        #Now add all together so its one continue daily plot
        Estimated_Merge[Max_C[0]] = pd.to_datetime(Estimated_Merge[Max_C[0]])
        
            
        # Merge the original DataFrame with the missing dates DataFrame
        Daily_Extremes_Est = pd.merge(missing_dates_df, Estimated_Merge, on=Max_C[0], how='outer')
        
        #set date as index
        Daily_Extremes_Est = Daily_Extremes_Est.set_index(Max_C[0])
        #Add to Dictionary
        Est_Daily_Extremes['Trial'+ "`_" + str(T)] = Daily_Extremes_Est
    return(Est_Daily_Extremes)


# # PART 9.5 CHOICE MODEL

# In[ ]:


def Choice_Model(data, Max, Corr_Stop):
    '''
    Parameters
    -----------------
    data : DataFrame
        A dataframe of all the DEs estimated for each day of which will be used for selecting the appropiate 
        DE.
    
    Max : True/False
        Determine if the function is looking at Maximum or Minimum temperature
    
    Corr_Stop : Value
        A threshold that reduces the chance for low correlated DEs to be used as the final DEs
    
    
    '''

    '''
    Lets work on the Max only version, this should be like in temp est v4 
    
    So like complex est 2 it has 3 modes, the lenth 1, 2 or more
    
    the first two options are the same as complex v2 but if there are 3 or more then thats when we start diverging 
    from the estimation complex variant 1
    
    More weight on the initial correlation and temp estimation
    
    '''

    #Single Case: Length of 1
    if (len(data) == 1):
        '''2.'''
        #This finds the values required for a length of 1 data
        if(Max ==True):
            data = data.reset_index(drop = True)
            Estimated_Max = data['Max Temp Estimation'].loc[0]
            Correlation_Max = data['Correlation Max T'].loc[0]
            return(Estimated_Max,Correlation_Max)
        else:
            data = data.reset_index(drop = True)
            Estimated_Min = data['Min Temp Estimation'].loc[0]
            Correlation_Min = data['Correlation Min T'].loc[0]
           
                
            return(Estimated_Min,Correlation_Min)

    #The only 2 datavalues choices
    elif (len(data) == 2):
        #Begin with Max
        if(Max == True):
            #Gets the value of the highest correlation first
            Highest_Correlation =  data.loc[data['Correlation Max T'] == data['Correlation Max T'].max()]
            Highest_Correlation = Highest_Correlation.reset_index(drop = True)
                
            #Now lets check if maximum estimation is the highest maximum value of the day
            Estimated_Temp = Highest_Correlation['Max Temp Estimation'].loc[0]
            #Lets see if the highest temperature is hiher then the estimated temp
            Highest_Actual_Temperature = data.loc[data['temp'] == data['temp'].max()]
            Highest_Actual_Temperature = Highest_Actual_Temperature.reset_index(drop =True)
            if(Estimated_Temp > Highest_Actual_Temperature['temp'].loc[0]):
                #Keep estimated temp
                return(Estimated_Temp, Highest_Correlation['Correlation Max T'].loc[0])
            else:
                #Choose the estimated highest actual temperature
                return(Highest_Actual_Temperature['Max Temp Estimation'].loc[0],Highest_Actual_Temperature['Correlation Max T'].loc[0])

                
        else:
            #Gets the value of the highesr correlation first
            Highest_Correlation =  data.loc[data['Correlation Min T'] == data['Correlation Min T'].max()]
            Highest_Correlation = Highest_Correlation.reset_index(drop = True)
            #Now lets check if minimum estimation is lower then the lowest minimum value of the day
            Estimated_Temp = Highest_Correlation['Min Temp Estimation'].loc[0]
            #Lets see if the highest temperature is hiher then the estimated temp
            Lowest_Actual_Temperature = data.loc[data['temp'] == data['temp'].min()]
            Lowest_Actual_Temperature = Lowest_Actual_Temperature.reset_index(drop =True)    
                
            if(Estimated_Temp < Lowest_Actual_Temperature['temp'].loc[0]):
                #Keep estimated temp
                return(Estimated_Temp, Highest_Correlation['Correlation Min T'].loc[0])
            else:
                #Choose the estimated lowest actual temperature
                return(Lowest_Actual_Temperature['Min Temp Estimation'].loc[0],Lowest_Actual_Temperature['Correlation Min T'].loc[0])
    
    
    
    
    
    else:
        #This is for 3 or more variables
        '''
        So the criteria for this one is:
        1. We begin by choosing the temperature of highest correlation known
        2. From this value we will then check whether at least 1 observational max is above or min is below the 
        estimated value        
        3.a If None are keep the value of the highest correlated temp
        3.b If there are extract those values
        4. Using 3.b check whether the correlation of any are above 0.85 then choose the highest one and keep that
        only
        
        '''
        #Decide is chosing max or min
        if(Max == True):
            #Gets the value of the highest correlation first
            Highest_Correlation =  data.loc[data['Correlation Max T'] == data['Correlation Max T'].max()]
            Highest_Correlation = Highest_Correlation.reset_index(drop = True)

            #Now lets check if there are more values that are hotter observationally then this
            Estimated_Temp = Highest_Correlation['Max Temp Estimation'].loc[0]

            #Lets see if the highest temperature is hiher then the estimated temp
            Highest_Actual_Temperature = data.loc[data['temp'] == data['temp'].max()]
            Highest_Actual_Temperature = Highest_Actual_Temperature.reset_index(drop = True) ##

            #Now do the checking function or step 2
            if(Estimated_Temp < Highest_Actual_Temperature['temp'].loc[0]):
                #Extract all values that have temperatures higher then this
                Highest_Actual_Temperature = data.loc[data['temp'] >= Estimated_Temp]
                #Drop any with a correlation of less then 0.85
                Highest_Actual_Temperature = Highest_Actual_Temperature.loc[Highest_Actual_Temperature['Correlation Max T'] >= Corr_Stop]
                
                Highest_Actual_Temperature = Highest_Actual_Temperature.reset_index(drop = True) ##
                
                #Check if there is more then  one varibale 
                if (len(Highest_Actual_Temperature) >= 1):
                    #Choose the highest correlated value
                    Highest_Correlation_Temp =  Highest_Actual_Temperature.loc[Highest_Actual_Temperature['Correlation Max T'] == Highest_Actual_Temperature['Correlation Max T'].max()]##hIGHEST CORRE TO HIGHEST ACTU
                    Highest_Correlation_Temp = Highest_Correlation_Temp.reset_index(drop = True)
                    return(Highest_Correlation_Temp['Max Temp Estimation'].loc[0],Highest_Correlation_Temp['Correlation Max T'].loc[0])
                else: 
                    #Return the estimated one from before
                    return(Estimated_Temp,Highest_Correlation['Correlation Max T'].loc[0])

            else:
                return(Estimated_Temp,Highest_Correlation['Correlation Max T'].loc[0])

        else:
            #Gets the value of the highest correlation first
            Highest_Correlation =  data.loc[data['Correlation Min T'] == data['Correlation Min T'].max()]
            Highest_Correlation = Highest_Correlation.reset_index(drop = True)

            #Now lets check if there are more values that are hotter observationally then this
            Estimated_Temp = Highest_Correlation['Min Temp Estimation'].loc[0]

            #Lets see if the highest temperature is hiher then the estimated temp
            Lowest_Actual_Temperature = data.loc[data['temp'] == data['temp'].min()]
            Lowest_Actual_Temperature = Lowest_Actual_Temperature.reset_index(drop = True) ##

           #Now do the checking function or step 2
            if(Estimated_Temp > Lowest_Actual_Temperature['temp'].loc[0]):
                #Extract all values that have temperatures higher then this
                Lowest_Actual_Temperature = data.loc[data['temp'] >= Estimated_Temp]
                #Drop any with a correlation of less then 0.85
                Lowest_Actual_Temperature = Lowest_Actual_Temperature.loc[Lowest_Actual_Temperature['Correlation Min T'] >= Corr_Stop]
                
                Lowest_Actual_Temperature = Lowest_Actual_Temperature.reset_index(drop = True) ##
                
                #Check if there is more then  one varibale 
                if (len(Lowest_Actual_Temperature) >= 1):
                    #Choose the highest correlated value
                    Highest_Correlation_Temp =  Lowest_Actual_Temperature.loc[Lowest_Actual_Temperature['Correlation Min T'] == Lowest_Actual_Temperature['Correlation Min T'].max()]##hIGHEST CORRE TO HIGHEST ACTU
                    Highest_Correlation_Temp = Highest_Correlation_Temp.reset_index(drop = True)
                    return(Highest_Correlation_Temp['Min Temp Estimation'].loc[0],Highest_Correlation_Temp['Correlation Min T'].loc[0])
                else: 
                    #Return the estimated one from before
                    return(Estimated_Temp,Highest_Correlation['Correlation Min T'].loc[0])
    
            else:
                return(Estimated_Temp,Highest_Correlation['Correlation Min T'].loc[0])


# # PART 10 Cleaning the Newly Estimated DEs

# In[ ]:


def Cleansing_Data(data):
    '''
    Parameters
    ----------------
    data : DataFrame
        The esitmated DEs in a DataFrame with associated correlations
        
    Returns 
    ---------------
    Max_All : DataFrame
        Tmax estimation DataFrame for all X trials with confidence intervals, mean and median
    
    CorrMax_All : DataFrame
        Tmax spearman rank correlation for all X trails with confidence intervals, mean and median
    
    Min_All : DataFrame
        Tmin estimation DataFrame for all X trials with confidence intervals, mean and median
    
    CorrMin_All : DataFrame
        Tmin spearman rank correlation for all X trails with confidence intervals, mean and median
    
    '''
    
    
    
    '''
    It is the dictionaries of all the trials and this will be just cleaned up with all relevent information to covnert
    them into 4 dataframes

    
    '''
    
        #DataFrames
    Max_DF = pd.DataFrame()
    Min_DF = pd.DataFrame()
    CorrMax_DF = pd.DataFrame()
    CorrMin_DF = pd.DataFrame()

    for key, df in data.items():
        #Extract the trial number
        trial_number = key.split('_')[1]

        #Change Name of each column to Something Simple with trial Number
        df.columns = ['Max_' + trial_number, 'MaxCorr_' + trial_number,
                      'Min_' + trial_number, 'MinCorr_' + trial_number]

        #Combine the Trials
        Max_DF = pd.concat([Max_DF, df[df.columns[0]]], axis=1)
        Max_DF = Max_DF.reset_index()
        Max_DF = Max_DF.rename(columns = {'index':'date'})
        Max_DF = Max_DF.set_index('date')
        CorrMax_DF = pd.concat([CorrMax_DF, df[df.columns[1]]], axis=1)
        CorrMax_DF = CorrMax_DF.reset_index()
        CorrMax_DF = CorrMax_DF.rename(columns = {'index':'date'})
        CorrMax_DF = CorrMax_DF.set_index('date')
        Min_DF = pd.concat([Min_DF, df[df.columns[2]]], axis=1)
        Min_DF = Min_DF.reset_index()
        Min_DF = Min_DF.rename(columns = {'index':'date'})
        Min_DF = Min_DF.set_index('date')        
        CorrMin_DF = pd.concat([CorrMin_DF, df[df.columns[3]]], axis=1)
        CorrMin_DF = CorrMin_DF.reset_index()
        CorrMin_DF = CorrMin_DF.rename(columns = {'index':'date'})
        CorrMin_DF = CorrMin_DF.set_index('date')      




    #Add Median, Mean, Conf60, Conf90, and Range

    #----------------MAX-----------------------#
    # Calculate the median across Trials
    Data = Max_DF
    median_values = Data.median(axis=1).round(2)
    median_values.name = 'Max Median'
    # Calculate the mean across Trials
    mean_values = Data.mean(axis=1).round(2)
    mean_values.name = 'Max Mean'
    # Calculate the 60% confidence interval
    confidence_60 = Data.quantile(q=[0.2, 0.8], axis=1).T.round(2)
    confidence_60.columns = ['Max Lower CI (60%)', 'Max Upper CI (60%)']
    # Calculate the 90% confidence interval
    confidence_90 = Data.quantile(q=[0.05, 0.95], axis=1).T.round(2)
    confidence_90.columns = ['Max Lower CI (90%)', 'Max Upper CI (90%)']
    # Calculate the full range
    Range = Data.apply(lambda row: np.ptp(row), axis=1).round(2)
    Range.name = 'Max Full Range'

    Max_All = pd.concat([mean_values,median_values,confidence_60,confidence_90,Range,Data.round(2)],axis=1)

    #----------------CORRMAX-----------------------#
    # Calculate the median across Trials
    Data = CorrMax_DF
    median_values = Data.median(axis=1).round(4)
    median_values.name = 'CorrMax Median'
    # Calculate the mean across Trials
    mean_values = Data.mean(axis=1).round(4)
    mean_values.name = 'CorrMax Mean'
    # Calculate the 60% confidence interval
    confidence_60 = Data.quantile(q=[0.2, 0.8], axis=1).T.round(4)
    confidence_60.columns = ['CorrMax Lower CI (60%)', 'CorrMax Upper CI (60%)']
    # Calculate the 90% confidence interval
    confidence_90 = Data.quantile(q=[0.05, 0.95], axis=1).T.round(4)
    confidence_90.columns = ['CorrMax Lower CI (90%)', 'CorrMax Upper CI (90%)']
    # Calculate the full range
    Range = Data.apply(lambda row: np.ptp(row), axis=1).round(4)
    Range.name = 'CorrMax Full Range'

    CorrMax_All = pd.concat([mean_values,median_values,confidence_60,confidence_90,Range,Data.round(4)],axis=1)



    #----------------MIN-----------------------#
    # Calculate the median across Trials
    Data = Min_DF
    median_values = Data.median(axis=1).round(2)
    median_values.name = 'Min Median'
    # Calculate the mean across Trials
    mean_values = Data.mean(axis=1).round(2)
    mean_values.name = 'Min Mean'
    # Calculate the 60% confidence interval
    confidence_60 = Data.quantile(q=[0.2, 0.8], axis=1).T.round(2)
    confidence_60.columns = ['Min Lower CI (60%)', 'Min Upper CI (60%)']
    # Calculate the 90% confidence interval
    confidence_90 = Data.quantile(q=[0.05, 0.95], axis=1).T.round(2)
    confidence_90.columns = ['Min Lower CI (90%)', 'Min Upper CI (90%)']
    # Calculate the full range
    Range = Data.apply(lambda row: np.ptp(row), axis=1).round(2)
    Range.name = 'Min Full Range'

    Min_All = pd.concat([mean_values,median_values,confidence_60,confidence_90,Range,Data.round(2)],axis=1)

    #----------------CORRMAX-----------------------#
    # Calculate the median across Trials
    Data = CorrMin_DF
    median_values = Data.median(axis=1).round(4)
    median_values.name = 'CorrMin Median'
    # Calculate the mean across Trials
    mean_values = Data.mean(axis=1).round(4)
    mean_values.name = 'CorrMin Mean'
    # Calculate the 60% confidence interval
    confidence_60 = Data.quantile(q=[0.2, 0.8], axis=1).T.round(4)
    confidence_60.columns = ['CorrMin Lower CI (60%)', 'CorrMin Upper CI (60%)']
    # Calculate the 90% confidence interval
    confidence_90 = Data.quantile(q=[0.05, 0.95], axis=1).T.round(4)
    confidence_90.columns = ['CorrMin Lower CI (90%)', 'CorrMin Upper CI (90%)']
    # Calculate the full range
    Range = Data.apply(lambda row: np.ptp(row), axis=1).round(4)
    Range.name = 'CorrMin Full Range'

    CorrMin_All = pd.concat([mean_values,median_values,confidence_60,confidence_90,Range,Data.round(4)],axis=1)

    return(Max_All, CorrMax_All, Min_All, CorrMin_All)


# In[ ]:




