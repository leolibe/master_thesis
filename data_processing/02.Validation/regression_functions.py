#%%
import numpy as np
import pandas as pd
import copy

import matplotlib.pyplot as plt
from datetime import datetime

from scipy.optimize import curve_fit
import natural_cubic_spline_stack_overflow as ncs # taken from stack overflow

import math as m
import pandas as pd

'''
Regression functions
This file gather the functions to test the following regressions on dataset:
- polynomial regression degree 1,2 and 3
- logistic regression
- ncs regression
'''
#%%

def polynomial_deg123_reg(reg_years:np.array, reg_values:np.array, reg_predictor_years:np.array)->tuple:
    ''' 
    This function performs a polynomial regression of degree 1, 2 and 3 on the data provided.
    Arguments: 
    - reg_years: np.array of years for the training data
    - reg_values: np.array of values for the training data
    - reg_predictor_years: np.array of years for the prediction data
    '''
    #plt.figure(figsize=(16,10))
    inputs = reg_years
    outputs = reg_values
    pred_inputs = reg_predictor_years

    #set the colors of the different plots corresponding to the polynomial regression of a degree between 1 and 3
    colors = ['','g', 'r', 'b']

    #initialize lists to store the inputs and outputs of the polynomial regression
    pred_inputs_list = []
    pred_outputs_list = []

    #perform regression for different degrees:
    for degree in [1,2,3]:
        #find polynomial
        polynomial = np.poly1d(np.polyfit(reg_years, reg_values, degree))
        print(f' the polynomial our fit created is: \n{polynomial}.')
        #apply to extended values
        pred_outputs = polynomial(reg_predictor_years)
    
        #plot the regression corresponding to the degree
        #plt.plot(pred_inputs, pred_outputs, color = colors[degree], lw = 3, label = f'polynomial regression (degree {degree}) values')
        pred_inputs_list.append(pred_inputs)
        pred_outputs_list.append(pred_outputs)
    return pred_inputs_list, pred_outputs_list
#%%
#define logistic function and noise function
def logistic(x:np.ndarray,ti:float,tau:float,C0:float,C1:float) -> np.ndarray:
    """
    General logistic function.
    Arguments:
    - x: np.ndarray of observation points (time)
    - ti: inflection time
    - tau: transition time coefficient
    - C0: start value
    - C1: end value

    Returns:
    - np.ndarray with len(x) number of points
    """
    return (C1 - C0)/(1 + np.exp(-(x - ti) / tau)) + C0  


def noise(start: int, stop: int, lo_time_deltas: list, lo_deviations: list) -> np.ndarray:
    '''
    Generates noise for timeseries for a set of time deltas and 
    deviations.This works by setting random deviations at certain 
    intervals and interpolating the points in between.The noise can then
    simply be added to the smooth timeseries curve to generate the final
    timeseries.
    
    Arguments:
    - start: beginning of the timeseries
    - stop: end of the timeseries
    - lo_time_deltas: list of time deltas which set the points at which 
                      noise trends are set
    - lo_deviations: the respective standard deviation from which the 
                     deviation for each point is drawn.

    Returns:
    - np.ndarray with stop-start+1 values of noise, averaging around 0
    '''
    no_time = stop-start +1 #number of discrete time instances
    final_points = np.zeros(no_time)

    for (time_delta, deviation) in zip(lo_time_deltas, lo_deviations):
        no_points = int((no_time-1)/time_delta)+2 #1 more than necessary to extend series
        end_time = start + (no_points-1)*time_delta
        macro_points = np.random.normal(0, deviation, no_points) 
        macro_point_x = np.linspace(start, end_time,no_points)
        macro_point_x = np.delete(macro_point_x, -1) #delete the extra point here


        extended_macro_points = [macro_points[0]]
        for index, macro_point in enumerate(macro_points[1:]):
            connection = np.linspace(macro_points[index], macro_point, time_delta+1, endpoint=True)
            extended_macro_points.extend(connection[1:])
        extended_macro_points = np.array(extended_macro_points[0:no_time])
        macro_points = np.delete(macro_points, -1)

        final_points = np.add(final_points, extended_macro_points)

    return final_points
#%%
#we start by setting some true values for a timeseries with a (some) function:
weight = 0
def true_function(x):
    '''Simple polynomial that looks nice for our values'''
    x = x-2000
    y =  -(x-25)**4/(7*10**3)   + ((x-22)**3)/100 - (x**2)/10  + x*2+ 200 + weight*np.sin(x)
    #y =  (x/3)** 2 + 60 + weight*np.sin(x)
    return y
start = 2000
duration = 50

#%%
def logistic_reg( reg_years:np.array, reg_values:np.array, reg_predictor_years:np.array, col:str, low_Co, low_C1,high_tau, high_Co, high_C1 ):
    '''
    This function performs a logistic regression on the data set and plots the results in the color col.
    Arguments:
    - reg_years: the years of the data set known
    - reg_values: the values of the data set known
    - reg_predictor_years: the years of the data set to predict
    - col: the color of the plot
    - low_Co: the lower bound of the parameter Co
    - low_C1: the lower bound of the parameter C1
    - high_tau: the higher bound of the parameter tau (tau the transition time coefficient)
    - high_Co: the higher bound of the parameter Co (the start value)
    - high_C1: the higher bound of the parameter C1 (the end value)
    '''
    #plt.figure(figsize=(16,10))
    inputs = reg_years
    outputs = reg_values
    pred_inputs = reg_predictor_years

    # it might be necessary to adjust the bounds argument, 
    # determining the extreme acceptable value for the parameters of the logistic function.
    # Bounds are set as ([low_ti, low_tau, low_Co, low_C1],[high_ti,high_tau, high_Co, high_C1 ])
    popt, pcov = curve_fit(logistic, inputs, outputs, bounds = ([min(inputs), 0, low_Co, low_C1], [max(inputs), high_tau, high_Co, high_C1]))
    pred_outputs = logistic(pred_inputs, *popt)
    print(f'The optimal choice of parameters for the logistic function, given the sample data, is {popt} (ti, tau, C0, C1).')
    #plt.plot(pred_inputs, pred_outputs, color = col, lw = 3, label = 'logistic regression values')
    return pred_inputs, pred_outputs
#%%
def natural_cubic_line_reg(reg_years, reg_values, reg_predictor_years, col):
    '''
    This function performs a natural cubic spline regression on the data set and plots the results in the color col.
    Arguments:
    - reg_years: the years of the data set known
    - reg_values: the values of the data set known
    - reg_predictor_years: the years of the data set to predict
    - col: the color of the plot
    '''
    #plt.figure(figsize=(16,10))
    reg_years_ncs = reg_years
    reg_values_ncs = reg_values
    reg_predictor_years_ncs = reg_predictor_years

    inputs_ncs = reg_years_ncs
    outputs_ncs = reg_values_ncs
    pred_inputs = reg_predictor_years_ncs

    # we can either choose the knots manually, or supply a number of knots
    # - see second graphic on top.
    # knots at the 2nd and 2nd to last points, and at 20%, 40%, 60% and 80%
    knots = [inputs_ncs[1], inputs_ncs[int(0.2*len(inputs_ncs))],inputs_ncs[int(0.4*len(inputs_ncs))],
            inputs_ncs[int(0.6*len(inputs_ncs))], inputs_ncs[int(0.8*len(inputs_ncs))], inputs_ncs[-2]]


    # just for showing the different cubic fits
    sections = []
    for i, knot in enumerate(knots[1:]):
        index_first = np.where(inputs_ncs == knots[i])[0][0]
        index_second = np.where(inputs_ncs == knot)[0][0]
        section_years = inputs_ncs[index_first:index_second]
        section_values = outputs_ncs[index_first:index_second]
        sections.append([section_years, section_values])

    # setting up the actual model (training)
    ncs_model = ncs.get_natural_cubic_spline_model(inputs_ncs, outputs_ncs, minval=min(inputs_ncs), 
                                                   maxval=max(inputs_ncs), knots = knots)
    # predicting of the single curve by the model
    pred_outputs = ncs_model.predict(pred_inputs)
    print(f'The ncs_model')

    #plot the knots:
    #or knot in knots:
        #plt.plot([knot, knot], [min(pred_outputs), max(pred_outputs)], lw = 0.5, color = 'darkgreen', 
                 #alpha = 0.4)

    for section in sections:
        [x,y] = section
        polynomial = np.poly1d(np.polyfit(x, y, 3))
        pol_outputs = polynomial(x)
        #plt.plot(x, pol_outputs, '--', color = 'blue', lw = 1.5, label = 'sectional cubic fit')
    #plt.plot(pred_inputs, pred_outputs, color = col, lw = 3, alpha = 0.8,
         #label = 'ncs regression values')
    #plt.xlabel('Years')
    #plt.ylabel('Values')
    return pred_inputs, pred_outputs
#%%
#To compare with the initial dataset

#calculate stock and implement stock driven model here
'''''
plt.figure(figsize=(16,10))

pred_inputs_poly, pred_outputs_poly = polynomial_deg123_reg(year, stock, pred_years)
pred_inputs_log, pred_outputs_log = logistic_reg(year,stock, pred_years, 'grey',0,0.5e8,10,1e8,1e8)
pred_inputs_ncs, pred_outputs_ncs = natural_cubic_line_reg(year, stock, pred_years, 'violet')
plt.plot(year, stock, 's', color = 'black', markersize = 2, label = f'original values')


#add plot and labels here
plt.legend()
plt.title('stock development')
plt.show()
'''