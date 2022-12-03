import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
from sklearn.metrics import r2_score
import random
import progressbar

def change_coord(data,names):
    "Changes the concentration vs time values to the rate vs time values for more precise fitting"
    for dataset in data:
        for name in names:
            rate_list = []
            for i in range(len(dataset[name])-1):
                rate_list.append((dataset[name][i+1]-dataset[name][i])/(dataset['time'][i+1]-dataset['time'][i])) 
            dataset[f'{name}_rate'] = rate_list
    return data

def get_r2(real,simul, names):
    total = 0
    for name in names:
        total += r2_score(real[name], simul[name]) + r2_score(real[f'{name}_rate'], simul[f'{name}_rate'])
    return total

def fit_sim(diff, data, names, rates, iterations =1000):
    max_score = len(data)*len(names)*2
    bar = progressbar.ProgressBar(max_value=iterations)
    progress = []
    def residual(paras):

        """
        compute the residual between actual data and fitted data
        """
        subject = {rate: paras[rate].value for rate in rates}
        score = 0
        
        for dataset in data:
            sim_dict = {}
            sol = integrate.solve_ivp(diff, [0,dataset['time'][-1]], dataset['init'], args = (subject,), t_eval = dataset['time'])
            sim_dict['time'] = list(sol.t)
            for count,name in enumerate(names):
                sim_dict[name] = list(sol.y[count])
            sim_dict = change_coord([sim_dict],names)
            score += get_r2(dataset,sim_dict[0], names)
        score_list = [max_score-score for i in range(len(subject))]
        
        if len(progress) < iterations:
            progress.append(0)
            bar.update(len(progress))
        
        return score_list



    # measured data (timespan, and noisy data list [0])
    plt.figure()
    

    # set parameters including bounds; you can also fix parameters (use vary=False)
    params = Parameters()
    for rate in rates:
        params.add(rate, value=0.01, min=1e-10, max=1)

    # fit model
    results = minimize(residual, params, method='leastsq', max_nfev = iterations)  # leastsq nelder
    # check results of the fit
    fin_sub = {rate: results.params[rate].value for rate in rates}
    
    score = 0
    final_list = []
    for dataset in data:
        final_dict = {}
        final_sol = integrate.solve_ivp(diff, [0,dataset['time'][-1]], dataset['init'], args = (fin_sub,), t_eval = dataset['time'])
        final_dict['time'] = list(final_sol.t)
        for count,name in enumerate(names):
            final_dict[name] = list(final_sol.y[count])
        final_dict = change_coord([final_dict],names)
        final_list.append(final_dict[0])
        score += get_r2(dataset,final_dict[0],names)


    # plot fitted data
    
    for count, dataset in enumerate(data):
        plt.figure()
        for name in names:
            plt.scatter(dataset['time'], dataset[name], marker = 'o', label = f'real data {name}')
            plt.plot(final_list[count]['time'], final_list[count][name], label = f'sim data {name}')
        plt.legend()
        plt.title(f"plot for T: {dataset['T']}, init: {dataset['init']}")
    # display fitted statistics
    print(f"score is {score} out of {max_score} for T: {data[0]['T']}")
    report_fit(results)
    
    return results.params

def fit_sim_full(diff, data, names, rates,guesses, temperatures, iterations =1000):
    max_score = len(data)*len(names)*2
    R = 8.314
    bar = progressbar.ProgressBar(max_value=iterations)
    progress = [] 
    def residual(paras):
        """
        compute the residual between actual data and fitted data
        """
        base_subject = {}
        for rate in rates:
            base_subject[f'E{rate[1:]}'] = paras[f'E{rate[1:]}'].value
            base_subject[rate] = paras[rate].value
        for rate in rates:
            if base_subject[rate] >1:
                score = [max_score for i in range(len(paras))]
                return score
            
        subjects = {}
        for temp in temperatures:
            subject = {rate: base_subject[rate]*(np.exp((-base_subject[f'E{rate[1:]}']/R)*(1/temp-1/max(temperatures))))
                      for rate in rates}    
            subjects[temp] = subject

        score = 0
        for dataset in data:
            sim_dict = {}
            subject = subjects[dataset['T']]
            sol = integrate.solve_ivp(diff, [0,dataset['time'][-1]], dataset['init'], args = (subject,), t_eval = dataset['time'])
            sim_dict['time'] = list(sol.t)
            for count,name in enumerate(names):
                sim_dict[name] = list(sol.y[count])
            sim_dict = change_coord([sim_dict],names)
            score += get_r2(dataset,sim_dict[0], names)
        score_list = [max_score-score for i in range(len(paras))]
        
        if len(progress) < iterations:
            progress.append(0)
            bar.update(len(progress))
        
        return score_list



    # set parameters including bounds; you can also fix parameters (use vary=False)
    params = Parameters()
    for num, rate in enumerate(rates):
        params.add(f'E{rate[1:]}', value=guesses[0,num], min=1000, max=100000)
        params.add(rate, value=guesses[1,num], min=1e-5, max=1e5)



    # fit model
    results = minimize(residual, params, method='leastsq', max_nfev = iterations)  # leastsq nelder
    
    fin_base_subject = {}
    for rate in rates:
        fin_base_subject[f'E{rate[1:]}'] = results.params[f'E{rate[1:]}'].value
        fin_base_subject[rate] = results.params[rate].value
    for rate in rates:
        if fin_base_subject[rate] >1:
            score = [max_score for i in range(len(paras))]
            return score

    fin_subjects = {}
    for temp in temperatures:
        fin_subject = {rate: fin_base_subject[rate]*(np.exp((-fin_base_subject[f'E{rate[1:]}']/R)*(1/temp-1/max(temperatures))))
                  for rate in rates}    
        fin_subjects[temp] = fin_subject
    
    score = 0
    final_list = []
    for dataset in data:
        final_dict = {}
        fin_subject = fin_subjects[dataset['T']]
        final_sol = integrate.solve_ivp(diff, [0,dataset['time'][-1]], dataset['init'], args = (fin_subject,), t_eval = dataset['time'])
        final_dict['time'] = list(final_sol.t)
        for count,name in enumerate(names):
            final_dict[name] = list(final_sol.y[count])
        final_dict = change_coord([final_dict],names)
        final_list.append(final_dict[0])
        score += get_r2(dataset,final_dict[0], names)
    
    # check results of the fit
    for count, dataset in enumerate(data):
        plt.figure()
        for name in names:
            plt.scatter(dataset['time'], dataset[name], marker = 'o', label = f'real data {name}')
            plt.plot(final_list[count]['time'], final_list[count][name], label = f'sim data {name}')
        plt.legend()
        plt.title(f"plot for T: {dataset['T']}, init: {dataset['init']}")
    print(f"score is {score} out of {max_score}")
    report_fit(results)
    
    # display fitted statistics
    
    return results.params
    

def data_formatting(data, temperatures, initial_conditions = None):
    total_dicts = []
    for i in range(len(data)):
        data_dict = {}
        for column in data[i].columns:
            data_dict[column] = list(data[i][column])
        data_dict['T'] = temperatures[i]
        data_dict['init'] = initial_conditions[i]            
        total_dicts.append(data_dict)
    return total_dicts
        

class fitting_set:
    def __init__(self,system,data,temperatures, initial_conditions, rates):
        self.system = system
        self.temperatures = sorted(set(temperatures))
        self.data = data_formatting(data, temperatures, initial_conditions)
        keys = self.data[0].keys()
        unwanted = ['time', 'T', 'init']
        self.species_names = [key for key in keys if key not in unwanted]
        self.data = change_coord(self.data,self.species_names)
        self.rates = rates
        self.rates_dict = {rate:[] for rate in self.rates}
        self.rates_dict['temp'] = []
        

        
    def fit_sys(self,data_num, iterations):
        fit_sim(self.system, [self.data[data_num]], self.species_names, self.rates, iterations)
        
     

    def fit_sys_all(self,iterations):
        self.max_temp_vals = []
        for temp in self.temperatures:
            indices = []
            for i in range(len(self.data)):
                if self.data[i]['T'] == temp:
                    indices.append(i)
            to_run = [self.data[i] for i in indices]
            fits = fit_sim(self.system, to_run, self.species_names, self.rates, iterations)
            self.rates_dict['temp'].append(temp)
            for rate in self.rates:
                self.rates_dict[rate].append(fits[rate].value)
                if temp == max(self.temperatures):
                    self.max_temp_vals.append(fits[rate].value)
        
    def kin_params_guesses(self):
        self.Eact_list = []
        self.A_list = []
        to_run = dict(self.rates_dict.items())
        del to_run['temp']
        for variable,lis in to_run.items():
            log_list = np.log(lis)
            reciprocal_list = 1/np.array(self.rates_dict['temp'])
            lin_model = np.polyfit(reciprocal_list,log_list,1)
            slope = lin_model[0]
            intercept = lin_model[1]
            self.Eact_list.append(slope*8.314)
            self.A_list.append(np.exp(intercept))
        
    def fit_sys_full(self, iterations):
        self.kin_params_guesses()
        guesses = np.zeros([2,6])
        guesses[0,:] = self.Eact_list
        guesses[0,:] *= -1
        guesses[1,:] = self.max_temp_vals
        for guess in guesses[0,:]:
            if guess <1000:
                guess = 1000
        fits = fit_sim_full(self.system, self.data, self.species_names, self.rates, guesses, self.temperatures, iterations)
        final_data = {}
        R = 8.314
        final_data['rate'] = self.rates
        final_data['E_act'] = []
        final_data[f'T={max(self.temperatures)}'] = []
        for rate in self.rates:
            final_data['E_act'].append(fits[f'E{rate[1:]}'].value)
            final_data[f'T={max(self.temperatures)}'].append(fits[rate].value)
        final_data['Arrhenius constant'] = list(np.array(final_data[f'T={max(self.temperatures)}'])/(np.exp(-np.array(final_data['E_act'])/(max(self.temperatures)*R))))
        for temp in self.temperatures:
            final_data[f'T={temp}'] = list(np.array(final_data[f'T={max(self.temperatures)}'])*(np.exp((-np.array(final_data['E_act'])/R)*(1/temp-1/max(self.temperatures)))))
        final_table = pd.DataFrame(final_data)
        return final_table
                
            
        
        