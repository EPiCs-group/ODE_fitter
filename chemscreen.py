# plot, make loop over values but cut off unnecessary data for simulated datasets
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, Parameter, report_fit
from sklearn.metrics import r2_score
import random
import progressbar

R=8.314

def preprocess_matrix(matrix,ks,reactions,concs,to_run):
    "Changes the matrices to a more easy to handle format for the differential"
    reactions = dict(reactions)
    matrix = dict(matrix)
    for key,dic in matrix.items():
        matrix[key] = dict(dic)
    ks = list(ks)
    concs = list(concs)
    
    for val in matrix.values():
        keylist = []
        for key in val.keys():            
            if key not in to_run.keys():
                keylist.append(key)                
        for pos in keylist:
            del val[pos]
            
    
    keylist = []
    for key in reactions.keys():            
        if key not in to_run.keys():
            keylist.append(key) 
    for pos in keylist:
        del reactions[pos]
        ks[pos-1] = 0

    
    keylist = []
    i=0
    poslist = []
    for key,dic in matrix.items():
        if dic == {}:
            keylist.append(key)
            poslist.append(i)
        i+=1
        
    for key in keylist:
        del matrix[key]
    poslist.reverse()
    for pos in poslist:
        del concs[pos]
        
    return matrix, ks, reactions, concs

def preprocess_reactions(reactions,to_run):
    "Changes the reaction to a more easy to handle format for the differential"
    for reaction in reactions.keys():
        reactions[reaction] = reactions[reaction][to_run[reaction]-1]
    return reactions

def diff(x, concs, ks,reacts,matrix):
    "Executes the differential based on the processed inputs"
    dt_list = []
    for val in (list(matrix.values())):
        tot = 0
        for key,value in val.items():
            react = reacts[key]
            tot += value*eval(react)
        dt_list.append(tot)
    for i in range(len(concs)-len(dt_list)):
        dt_list.append(0)
    return dt_list

class ODE_systems:
    "Class to store ODE systems"
    def __init__(self, reactions, matrix,ks,to_run = None):
        self.reactions = reactions
        self.matrix = matrix
        self.ks = [0.01 for i in range(ks)]
        self.concs = [1 for i in range(len(matrix))]
        if to_run != None:
            self.to_run = to_run
        else:
            self.to_run = None
        
    
    def get_matrices(self):
        "Makes the processed differentials from the user inputs"
        systems_list = []
        if self.to_run != None:
            for run in self.to_run:
                new_matrix, new_ks, new_reactions, new_concs = preprocess_matrix(self.matrix,self.ks,self.reactions,self.concs,run)
                new_reactions = preprocess_reactions(new_reactions,run)
                systems_list.append([dict(new_matrix), list(new_ks), dict(new_reactions), list(new_concs), len(new_ks)])
        else:
            systems_list.append([self.matrix, self.ks, self.reactions, self.concs])
        return systems_list
                                                                                 

def get_A(ktrue, E_act, ref_temp):
    "To derive arrhenius constant from the rate constant at the first input temperature and the activation energy"
    R = 8.314
    E_act = np.array(E_act)
    T= ref_temp
    A = ktrue/(np.exp(-E_act/(ref_temp*R)))
    return A

def get_noise(t,var,stat):
    "Function to add noise to the data"
    length = len(t)
    static_noise = np.random.normal(0,stat,length)
    variable_noise = np.random.normal(0,var,length)
    return static_noise, variable_noise

def make_data(system,timespan ,Ts,inits,ks,kin_params = None,var_noise = 0, stat_noise = 0):
    "Makes a dataset out of the specified kinetic parameters, times, temperatures, and initial conditions, with noise if specified"
    time = timespan[-1]
    
    if kin_params:
        pre_consts = get_A(ks,kin_params,Ts[0])
        
    
    sol_list = []
    noisy_data_lists = []
    for init in inits:
        for T in Ts:
            noisy_data = []
            if kin_params != None:
                rate_consts = pre_consts*np.exp(-np.array(kin_params)/(R*T))
            else:
                rate_consts = ks
            sol = integrate.solve_ivp(system, [0,time], init, t_eval = timespan, args=[rate_consts])
            indexes = []
            
            for i in range(len(sol.y)):
                if sol.y[i][-1] == sol.y[i][0]:
                    indexes.append(i)
            indexes.reverse()
            sol.y = list(sol.y)
            if indexes != []:
                for index in indexes:
                    del sol.y[index]
                sol.y = np.array(sol.y)
            for line in sol.y:
                static_noise,variable_noise = get_noise(timespan,var_noise,stat_noise)
                noisy_data.append(line + static_noise + np.multiply(line,variable_noise))
            data_dict = {"t" : timespan, "y": noisy_data, "T": T, "init" : init}
            noisy_data_lists.append(data_dict)
            

        
    noisy_data_lists = np.array(noisy_data_lists)
    return noisy_data_lists

def change_coord(data,timespan):
    "Changes the concentration vs time values to the rate vs time values for more precise fitting"
    rate_lists = []
    for line in range(len(data)):
        rate_list = []
        for i in range(len(data[line])-1):
            rate_list.append((data[line][i+1]-data[line][i])/(timespan[i+1]-timespan[i]))   
        rate_lists.append(rate_list)

    return rate_lists

def get_r2(real,simul,num_lines,timespan):
    "Calculates the R-squared of compared datasets"
    tot_score = 0
    real_rate, simul_rate = change_coord(real, timespan), change_coord(simul, timespan)
    for i in range(num_lines):
        tot_score += r2_score(real[i],simul[i]) +r2_score(real_rate[i],simul_rate[i])
    return tot_score

def kin_params_guesses(constants, Ts):
    "Plots and returns the guessed activation energy and Arrhenius constant based on rate constants at certain temperatures"
    Eact_list = []
    A_list = []
    for i in range(len(constants[0])):
        log_list = []
        for constant_list in constants:
            log_list.append(np.log(constant_list[i]))
        reciprocal_list = 1/np.array(Ts)
        lin_model = np.polyfit(reciprocal_list,log_list,1)
        slope = lin_model[0]
        intercept = lin_model[1]
        Eact_list.append(slope*8.314)
        A_list.append(np.exp(intercept))
        plt.figure(100)
        plt.title('arrhenius plot')
        plt.scatter(reciprocal_list,log_list, label = f'k{i}')
        plt.plot(reciprocal_list, np.array(reciprocal_list)*slope+intercept, label = 'fit')
        plt.legend()
    return Eact_list, A_list

def fit_sim(system,data, iterations):
    "Main function for optimizing to an optimal fit, does this only at a single temperature"
    matrix,reacts,num_ks,init = system[0],system[2],len(system[1]), system[3]
    if len(data) <= 1:
        init_list = [data[0]['init']]
        T = data[0]['T']
        values_list = [data[0]['y']]
        timespan_list = [data[0]['t']]
        time_list = [data[0]['t'][-1]]
        max_score = len(matrix)*2
    else:
        T = data[0]['T']
        init_list = []
        values_list = []
        timespan_list = []
        time_list = []
        max_score = 0
        for entry in data:
            init_list.append(entry['init'])
            values_list.append(entry['y'])
            timespan_list.append(entry['t'])
            time_list.append(entry['t'][-1])
            max_score += 2*len(matrix)
    
    bar = progressbar.ProgressBar(max_value=iterations)
    progress = []        

    def residual(paras):
        "compute the residual between actual data and fitted data"
        param_list = []
        for param in paras:
            param_list.append(paras[param].value)
            

        
        score = 0
        for i in range(len(data)):
            sol = integrate.solve_ivp(diff, [0,time_list[i]], init_list[i], t_eval = timespan_list[i], args=[param_list,reacts,matrix])
            score += get_r2(values_list[i], sol.y, len(data[0]['y']), timespan_list[i])
        if len(progress) < iterations:
            progress.append(0)
            bar.update(len(progress))

        to_return = [max_score-score for i in range(len(paras))]
        return to_return

    

    params = Parameters()
    for i in range(num_ks):
        params.add(f'k{i}',value=0.01, min=1e-10, max=1)


    results = minimize(residual, params, method='leastsq', max_nfev = iterations)  # leastsq nelder
    
    result_list = []
    for param in results.params:
        result_list.append(results.params[param].value)

    
    final_res = []
    
    for i in range(len(results.params)):
        final_res.append(results.params[f'k{i}'].value)

    
    for j in range(len(data)):
        data_fitted = integrate.solve_ivp(diff, [0,time_list[j]], init_list[j], t_eval = timespan_list[j], args=[result_list,reacts,matrix])
        plt.figure()
        if len(values_list[j])> len(matrix.items()):
            for i in range(len(matrix.items())):
                plt.plot(data_fitted.t,data_fitted.y[i], label =list(matrix.keys())[i][1:-2])
                plt.scatter(timespan_list[j], values_list[j][i], label =list(matrix.keys())[i][1:-2])
        else:
            for i in range(len(values_list[j])):
                plt.scatter(timespan_list[j], values_list[j][i], label =list(matrix.keys())[i][1:-2])
                plt.plot(data_fitted.t,data_fitted.y[i], label =list(matrix.keys())[i][1:-2])
        plt.title(f'single temperature fit plot at temperature: {T} K')
        plt.ylabel('concentration')
        plt.xlabel('time')
        plt.legend()
    sim = max_score-residual(results.params)[0]
    print(f'score for temperature fit of {T} is: {sim} out of {max_score} \n \n fit results:')
    # display fitted statistics
    report_fit(results)
    
    final_res = []
    
    for i in range(len(results.params)):
        final_res.append(results.params[f'k{i}'].value)
    
    return final_res, T

def get_lines(diff, param_list, reacts, matrix,T, data):
    "Helper function for executing the differentials for the activation energy and Arrhenius constant guesses"
    ks_list = []
    half = int(len(param_list)/2)
    for i in range(half):
        ks_list.append(param_list[i]*np.exp(-param_list[half+i]/(R*T)))

    init= data['init']
    T = data['T']
    timespan = data['t']
    time = timespan[-1]
    lines = integrate.solve_ivp(diff, [0,time], init, t_eval = timespan, args=[ks_list,reacts,matrix])

    return lines

def fit_sim_full(system,data, temps, Eacts, As,iterations):
    "Main function for optimizing to an optimal fit, does this at all temperature by guessing activation energies and Arrhenius constants"
    matrix,reacts,num_ks,init = system[0],system[2],len(system[1]), system[3]
    R = 8.314
    max_score = len(data)*len(data[0]['y'])*2
    pre_consts = []
    E_acts = []
    As_names = []
    Eacts_names = []
    for i in range(num_ks):
        As_names.append(f'A{i+1}')
        Eacts_names.append(f'E{i+1}')
    bar = progressbar.ProgressBar(max_value=iterations)
    progress = []
    
    def residual(paras):
        "compute the residual between actual data and fitted data"
        param_list = []
        
        for param in paras:
            param_list.append(paras[param].value)
        
        
        half = int(len(param_list)/2)
        for i in range(half):
            val = param_list[i]*np.exp(-param_list[half+i]/(R*temps[-1]))
            if val >1:
                to_return = [max_score for i in range(len(paras))]
                return to_return
        lines_list = []
        for i in range(len(data)):
            T_current = data[i]['T']
            lines = get_lines(diff,param_list,reacts,matrix,T_current,data[i])
            lines_list.append(lines)


        score = 0
        num = 0
        for run in data:
            score += get_r2(run['y'], lines_list[num].y,len(run['y']), run['t'])
            num += 1
            
        if len(progress) < iterations:
            progress.append(0)
            bar.update(len(progress))
        
        to_return = [max_score-score for i in range(len(paras))]
        return to_return

    params = Parameters()
    for i in range(len(As)):
        params.add(As_names[i],value=As[i], min=1e-5, max=1e5)
    for i in range(len(Eacts)):
        params.add(Eacts_names[i],value=-Eacts[i], min=1000, max=100000)    


    results = minimize(residual, params, method='leastsq', max_nfev = iterations)  # leastsq nelder
    
    result_list = []
    for param in results.params:
        result_list.append(results.params[param].value)

    bar.update(iterations)
    
    for i in range(len(data)):
        if len(data[0]['y']) > len(matrix.items()):
            T_current = data[i]['T']
            mock_run = get_lines(diff,result_list,reacts,matrix,T_current,data[i])
            plt.figure()
            for j in range(len(matrix.items())):
                plt.scatter(data[i]['t'],data[i]['y'][j], label =list(matrix.keys())[j][1:-2])
                plt.plot(mock_run.t, mock_run.y[j], label =list(matrix.keys())[j][1:-2])
            plt.legend()
            plt.title(f'Simultaneous fitting plot for temperature: {T_current} K')
            plt.ylabel('Concentration')
            plt.xlabel('Time')
            plt.savefig(f"C:/Users/natha/OneDrive/Documenten/uni/uni documenten/MEP/MEP_figures/paper/multi_fit{i}.svg",dpi=300)
        else:
            T_current = data[i]['T']
            mock_run = get_lines(diff,result_list,reacts,matrix,T_current,data[i])
            plt.figure()
            for j in range(len(data[0]['y'])):
                plt.scatter(data[i]['t'],data[i]['y'][j], label =list(matrix.keys())[j][1:-2])#, color= colors[j])
                plt.plot(mock_run.t, mock_run.y[j], label =list(matrix.keys())[j][1:-2])#, color= colors[j])
            plt.legend()
            plt.title(f'Simultaneous fitting plot for temperature: {T_current} K')
            plt.ylabel('Concentration')
            plt.xlabel('Time')
            plt.savefig(f"C:/Users/natha/OneDrive/Documenten/uni/uni documenten/MEP/MEP_figures/paper/multi_fit{i}.svg",dpi=300)
            
    sim = max_score-residual(results.params)[0]
    print(f'full fit score is: {sim} out of {max_score}\n \n fit results:')
    
    report_fit(results)
    
    return results.params

class fit_data():
    "Class to execute and store a fitting procedure"
    def __init__(self, data, systems):
        self.data = data
        self.systems = systems.get_matrices()
        
    def fit_one(self,entry_num,system_num, iterations = 1000):
        "for fitting a single run with a single system"
        data = [self.data[entry_num]]
        system = self.systems[system_num]
        fit_sim(system,data,iterations)
        
    def fit_many_entries(self,entries,system_num, iterations = 1000):
        "for multiple runs with a single system"
        for entry in entries:
            data = [self.data[entry]]
            system = self.systems[system_num]
            fit_sim(system,data,iterations)
    
    def fit_many_systems(self,entry_num,systems, iterations = 1000):
        "for fitting one run with multiple systems"
        for system in systems:
            data = [self.data[entry_num]]
            system = self.systems[system]
            fit_sim(system,data,iterations)
    
    def fit_many(self,entries,systems, iterations = 1000):
        "for fitting multiple runs with multiple systems"
        for entry in entries:
            for system in systems:            
                data = [self.data[entry]]
                system = self.systems[system]
                fit_sim(system,data,iterations)
                
    def fit_full(self,system_num, iterations = 1000, iterations_full = 1000):
        "for fitting all data with a single system and for fitting across different temperatures"
        system = self.systems[system_num]
        unique_temps = []
        data_list = []
        params_list = []
        T_list = []
        for entry in self.data:
            unique_temps.append(entry['T'])
        unique_temps = set(unique_temps)
        unique_temps = list(unique_temps)
        unique_temps.sort()
        for temp in unique_temps:
            hits = []
            for entry in self.data:
                if entry['T'] == temp:
                    hits.append(entry)
            data_list.append(hits)
        for pair in data_list:
            params, T = fit_sim(system,pair,iterations)
            params_list.append(params)
            T_list.append(T)
        Eacts, As = kin_params_guesses(params_list,T_list)
        fit_sim_full(system,self.data,T_list,Eacts,As, iterations_full)
        
            