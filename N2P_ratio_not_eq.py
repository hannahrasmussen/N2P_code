#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy as sym
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import phys
from scipy.interpolate import CubicSpline

#constants:
a_value = 1/137    #The fine structure constant; unitless
D = 1           #a parameter that acts as a fraction of the number density of fermions?
dmnp = 1.29332 #difference in mass of a proton and neutron in MeV
f_pi = 131         #MeV, not really sure what this constant means 
gA = 1.27 #some constant that has to do with weak vector-axial coupling
Gf = 1.166*10**-11 #This is the fermi constant in units of MeV^-2
hbar = 6.582*10**-22 #Mev*s
me = .511      #Mass of an electron in MeV
mpi_charged = 139.569  #Mass of a charged pion in MeV
mpi_neutral = 135  #Mass of a neutral pion in MeV
mPL = 1.124*10**22 #Planck mass in MeV
mu = 105.661       #Mass of a muon in MeV
x0 = me/mu


#Cubic splining Luke's code (in case we decide we need it):
#eq_a_array = np.load("Luke-a.npy") #equilibrium a array
#eq_T_array = np.load("Luke-T.npy") #equilibrium temperature array
#eq_eta_array = np.load("Luke-eta.npy") #equilibrium eta array
#cs_eta = CubicSpline(np.flip(eq_T_array),np.flip(eq_eta_array)) #Temp is flipped so it is strictly 
        #increasing as the independent variable, but that means any output from here needs
        #to be flipped back in order to graph properly #oh wait shit doesn't that mean eq_eta_array should be
        #flipped as well to make the indices match?


@nb.jit(nopython=True)
def f_elec(Ee,T,eta): #Ee argument is energy of electron, function returns occupation fraction of electron at this energy & temp
    return 1/(np.e**((Ee/T)-eta)+1)

@nb.jit(nopython=True)
def f_pos(Ee,T,eta): #Ee argument is energy of electron, function returns occupation fraction of positron at this energy & temp
    return 1/(np.e**((Ee/T)+eta)+1)
    
@nb.jit(nopython=True)
def trapezoid(array,dx,x0,xf,a,B): #here dx is bxszE,x0 and xf are actual starting and ending points of array,
                                   #a and B are desired starting and ending points of array 
    total = np.sum(dx*(array[1:-2]+array[2:-1])/2) #first and last boxes aren't included, they're added in specially below
    
    if (len(array)==1):
        total += (B-a)*array[0]
    
    else:
        #linear interpolation:
        array_x0 = (((array[1]-array[0])/dx)*a) + array[0]-(x0*((array[1]-array[0])/dx)) #takes the form of (m*x)+b
        array_xf = (((array[-1]-array[-2])/dx)*B) + array[-1]-(xf*((array[-1]-array[-2])/dx)) #takes the form of (m*x)+b
    
        diff_start = (x0+dx)-a
        diff_end = B-(xf-dx)
        total += (diff_start*(array_x0+array[1]))/2
        total += (diff_end*(array_xf+array[-2])/2)
    
    return total

@nb.jit(nopython=True)
def oldtrapezoid(array,dx): #dx will just be boxsize for our cases currently
    total = np.sum(dx*(array[1:]+array[:-1])/2)
    return total


# In[2]:


eps_values, w_values = np.polynomial.laguerre.laggauss(10)

@nb.jit(nopython=True)
def calc_I1(x):
    return np.sum(w_values*I1(eps_values,x)) 

@nb.jit(nopython=True)
def I1(eps,x): #Energy Density
    numerator = (np.e**eps)*(eps**2)*((eps**2+x**2)**.5)
    denominator = np.e**((eps**2+x**2)**.5)+1
    return numerator/denominator

@nb.jit(nopython=True)
def nH(Tcm,t,ms,angle): #number density of decaying particles
    part1 = D*3*1.20206/(2*np.pi**2)
    part2 = Tcm**3*np.e**(-t/tH(ms,angle))
    return part1*part2


# In[3]:


@nb.jit(nopython=True)
def decay2(ms,angle):  #angle is the mixing angle of vs with active neutrinos
    numerator = 9*(Gf**2)*a_value*(ms**5)*((np.sin(angle))**2)
    denominator = 512*np.pi**4
    gamma = numerator/denominator
    return gamma

@nb.jit(nopython=True)
def decay5(ms,angle): #angle is the mixing angle of the sterile neutrino with the active neutrinos
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    part2 = ms*((ms**2)-(mpi_neutral**2))*(np.sin(angle))**2
    gamma = part1*part2
    return gamma

@nb.jit(nopython=True)
def decay6(ms,angle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+me)**2)*((ms**2) - (mpi_charged-me)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(angle))**2
    gamma = part1*part2
    return 2*gamma #because vs can decay into either pi+ and e- OR pi- and e+

@nb.jit(nopython=True)
def decay7(ms,angle):
    part1 = (Gf**2)*(f_pi**2)/(16*np.pi)
    parentheses = ((ms**2) - (mpi_charged+mu)**2)*((ms**2) - (mpi_charged-mu)**2)
    part2 = ms * ((parentheses)**(1/2)) * (np.sin(angle))**2
    gamma = part1*part2
    return 2*gamma #because vs can decay into either pi+ and u- OR pi- and u+

@nb.jit(nopython=True)
def tH(ms,angle):
    return 1/(decay2(ms,angle)+decay5(ms,angle)+decay6(ms,angle)+decay7(ms,angle))


# In[4]:


@nb.jit(nopython=True)
def dYn_dt(y_array,bxszE,eta,Yn,Yp): #change in neutron proportion
    #nue stands for electron neutrino, n stands for neutron, pos stands for positron, p stands for proton, 
    #elec stands for electron, and anue stands for electron antineutrino
    l_v_n = lambda_nue_n(y_array,bxszE,eta)
    l_po_n = lambda_pos_n(y_array,bxszE,eta)
    l_n = lambda_n(y_array,bxszE,eta)
    l_p_e = lambda_p_elec(y_array,bxszE,eta)
    l_av_p = lambda_anue_p(y_array,bxszE,eta)
    l_av_e_p = lambda_anue_elec_p(y_array,bxszE,eta)
    return -Yn*(l_v_n + l_po_n + l_n) + Yp*(l_p_e + l_av_p + l_av_e_p)


# In[5]:


@nb.jit(nopython=True)
def dYp_dt(y_array,bxszEE,eta,Yn,Yp): #change in proton proportion 
    #nue stands for electron neutrino, n stands for neutron, pos stands for positron, p stands for proton, 
    #elec stands for electron, and anue stands for electron antineutrino
    l_v_n = lambda_nue_n(y_array,bxszE,eta)
    l_po_n = lambda_pos_n(y_array,bxszE,eta)
    l_n = lambda_n(y_array,bxszE,eta)
    l_p_e = lambda_p_elec(y_array,bxszE,eta)
    l_av_p = lambda_anue_p(y_array,bxszE,eta)
    l_av_e_p = lambda_anue_elec_p(y_array,bxszE,eta)
    return Yn*(l_v_n + l_po_n + l_n) - Yp*(l_p_e + l_av_p + l_av_e_p)


# In[6]:


@nb.jit(nopython=True)
def lambda_nue_n(y_array,bxszE,eta): #bxszE is the boxsize of the current energy array
    T = y_array[-2]
    integrand = np.zeros(len(y_array)-3) #integral will go from 0 to "infinity" via Gauss Laguerre
    for i in range (len(integrand)):
        Ev = bxszE*i
        part1 = 1 #usually includes Coulomb and zero temp corrections but we're leaving these out
        part2 = (Ev**2)*(Ev+dmnp)*((Ev+dmnp)**2 - me**2)**(1/2)
        part3 = y_array[i]*(1-f_elec(Ev+dmnp,T,eta)) #here, y_array refers to neutrino population and f_elec refers to electron occupation fraction
        integrand[i] = part1*part2*part3
    integral = trapezoid(integrand,bxszE,0,bxszE*i,0,bxszE*i)
    return (Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)


# In[7]:


@nb.jit(nopython=True)
def lambda_pos_n(y_array,bxszE,eta): #bxszE is the boxsize of the current energy array
    T = y_array[-2]
    if ((len(y_array)-3-int((dmnp+me)/bxszE))<=0):
        return 0
    integrand = np.zeros(len(y_array)-3-int((dmnp+me)/bxszE)) #this should make the integral go from dmnp+me to "infinity"
    for i in range (len(integrand)):
        Ev = bxszE*i + bxszE*int((dmnp+me)/bxszE) #makes sure I start at the box before dmnp+me
        part1 = 1 #usually includes zero temp correction but we're leaving it out
        part2 = (Ev**2)*(Ev-dmnp)*((Ev-dmnp)**2 - me**2)**(1/2)
        part3 = (1-y_array[i+int((dmnp+me)/bxszE)])*(f_pos(Ev-dmnp,T,eta)) #here, y_array refers to antineutrino population and f_pos refers to positron occupation fraction
        integrand[i] = part1*part2*part3
    integrand[0] = 0 #otherwise it will be nan because the first Ev will always be less than dmnp+me, making part2 imaginary
    integral = trapezoid(integrand,bxszE,bxszE*int((dmnp+me)/bxszE),bxszE*i,dmnp+me,bxszE*i)
    return (Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)


# In[8]:


@nb.jit(nopython=True)
def lambda_n(y_array,bxszE,eta): #bxszE is the boxsize of the current energy array
    T = y_array[-2]
    integrand = np.zeros(int((dmnp-me)/bxszE)+1) #integral will go from 0 to dmnp-me
    if (len(integrand)>len(y_array)):
        hold = len(y_array)
    else:
        hold = len(integrand)
    for i in range (hold):
        Ev = bxszE*i
        part1 = 1 #usually includes Coulomb and zero temp correction but we're leaving these out
        part2 = (Ev**2)*(dmnp-Ev)*((dmnp-Ev)**2 - me**2)**(1/2)
        part3 = (1-y_array[i])*(1-f_elec(dmnp-Ev,T,eta)) #here, y_array refers to antineutrino population and f_elec refers to electron occupation fraction
        integrand[i] = part1*part2*part3
    integral = trapezoid(integrand,bxszE,0,bxszE*i,0,dmnp-me)
    return (Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)


# In[9]:


@nb.jit(nopython=True)
def lambda_p_elec(y_array,bxszE,eta): #bxszE is the boxsize of the current energy array
    T = y_array[-2]
    integrand = np.zeros(len(y_array)-3) #integral will go from 0 to "infinity"
    for i in range (len(integrand)):
        Ev = bxszE*i
        part1 = 1 #usually includes Coulomb and zero temp correction but we're leaving these out
        part2 = (Ev**2)*(Ev+dmnp)*((Ev+dmnp)**2 - me**2)**(1/2)
        part3 = (1-y_array[i])*(f_elec(Ev+dmnp,T,eta)) #here, y_array refers to neutrino population and f_elec refers to electron occupation fraction
        integrand[i] = part1*part2*part3
    integral = trapezoid(integrand,bxszE,0,bxszE*i,0,bxszE*i)
    return (Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)


# In[10]:


@nb.jit(nopython=True)
def lambda_anue_p(y_array,bxszE,eta): #bxszE is the boxsize of the current energy array
    T = y_array[-2]
    if ((len(y_array)-3-int((dmnp+me)/bxszE))<=0):
        return 0
    integrand = np.zeros(len(y_array)-3-int((dmnp+me)/bxszE)) #this should make the integral go from dmnp+me to "infinity"
    for i in range (len(integrand)):
        Ev = bxszE*i + bxszE*int((dmnp+me)/bxszE) #makes sure I start at the box before dmnp+me
        part1 = 1 #usually includes zero temp correction but we're leaving it out
        part2 = (Ev**2)*(Ev-dmnp)*((Ev-dmnp)**2 - me**2)**(1/2)
        part3 = (y_array[i+int((dmnp+me)/bxszE)])*(1-f_pos(Ev-dmnp,T,eta)) #here, y_array refers to antineutrino population and f_pos refers to positron occupation fraction
        integrand[i] = part1*part2*part3
    integrand[0] = 0 #otherwise it will be nan because the first Ev will always be less than dmnp+me, making part2 imaginary
    integral = trapezoid(integrand,bxszE,bxszE*int((dmnp+me)/bxszE),bxszE*i,dmnp+me,bxszE*i)
    return (Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)


# In[11]:


@nb.jit(nopython=True)
def lambda_anue_elec_p(y_array,bxszE,eta): #bxszE is the boxsize of the current energy array
    T = y_array[-2]
    integrand = np.zeros(int((dmnp-me)/bxszE)+1) #integral will go from 0 to dmnp-me
    if (len(integrand)>len(y_array)):
        hold = len(y_array)
    else:
        hold = len(integrand)
    for i in range (hold):
        Ev = bxszE*i
        part1 = 1 #usually includes Coulomb and zero temp correction but we're leaving these out
        part2 = (Ev**2)*(dmnp-Ev)*((dmnp-Ev)**2 - me**2)**(1/2)
        part3 = (y_array[i])*(f_elec(dmnp-Ev,T,eta)) #here, y_array refers to antineutrino population and f_elec refers to electron occupation fraction
        integrand[i] = part1*part2*part3
    integral = trapezoid(integrand,bxszE,0,bxszE*i,0,dmnp-me)
    return (Gf**2)*(1 + 3*gA**2)*integral/(2*np.pi**3)


# In[12]:


@nb.jit(nopython=True)
def driver(a_array, T_array, y_mat, e_matrix, ms, mixangle):
    y_matrix = np.transpose(y_mat) #dQda, Temp, and time are y_matrix[-3], y_matrix[-2],
                               #and y_matrix[-1], respectively. Time is in Mev^-1. Pretty sure the way the outputs are
                               #currently designed we need to transpose y_mat
    
    eta_array = np.zeros(len(a_array))
    #eta_array = np.flip(cs_eta(np.flip(T_array))) #<--- in case we decide we need actual values for eta
    
    l_v_n_array = np.zeros(len(a_array))
    l_po_n_array = np.zeros(len(a_array))
    l_n_array = np.zeros(len(a_array))
    l_p_e_array = np.zeros(len(a_array))
    l_av_p_array = np.zeros(len(a_array))
    l_av_e_p_array = np.zeros(len(a_array))
    
    for i in range(len(a_array)):
        e_array = e_matrix[-1]
        boxsize = e_array[1]-e_array[0]
        l_v_n_array[i] = lambda_nue_n(y_matrix[i],boxsize/a_array[i], eta_array[i])/hbar
        l_po_n_array[i] = lambda_pos_n(y_matrix[i],boxsize/a_array[i], eta_array[i])/hbar
        l_n_array[i] = lambda_n(y_matrix[i],boxsize/a_array[i], eta_array[i])/hbar
        l_p_e_array[i] = lambda_p_elec(y_matrix[i],boxsize/a_array[i], eta_array[i])/hbar
        l_av_p_array[i] = lambda_anue_p(y_matrix[i],boxsize/a_array[i], eta_array[i])/hbar
        l_av_e_p_array[i] = lambda_anue_elec_p(y_matrix[i],boxsize/a_array[i], eta_array[i])/hbar

    n_to_p = l_v_n_array + l_po_n_array + l_n_array
    p_to_n = l_p_e_array + l_av_p_array + l_av_e_p_array
    
    Tcm_array = 1/a_array
        
    H = np.zeros(len(T_array))
    for i in range(len(T_array)):
        e_array = e_matrix[i]
        boxsize = e_array[1]-e_array[0]
        H_part1 = 8 * np.pi / (3 * mPL**2)
        H_part3 = T_array[i]**4 * np.pi**2 / 15                  #photons
        H_part4 = 2 * T_array[i]**4 * calc_I1(me/T_array[i]) / np.pi**2   #electrons/positrons
        H_part5 = 7/8 * np.pi**2/30 * 6 * Tcm_array[i]**4        #neutrinos
        H_part6 = ms*nH(Tcm_array[i],y_matrix[i][-1],ms,mixangle)          #sterile neutrinos
        H_part7 = (Tcm_array[i]**4/(2*np.pi**2))*oldtrapezoid(y_matrix[i][:len(e_array)]*e_array**3,boxsize) #products of sterile neutrinos?
        H[i] = np.sqrt(H_part1 * (H_part3 + H_part4 + H_part5 + H_part6 + H_part7)) / hbar
        
    #plt.figure(figsize=(8,6))
    #plt.loglog(T_array,n_to_p,label='$n\Rightarrow p$')
    #plt.loglog(T_array,p_to_n,label='$p\Rightarrow n$')
    #plt.loglog(T_array,H,linestyle='--',label = 'Hubble rate')
    #plt.xlabel('$T$ (MeV)',fontsize=18)
    #plt.ylabel('Rate ($s^{-1}$)',fontsize=18)
    #plt.tick_params(axis="x", labelsize=12)
    #plt.tick_params(axis="y", labelsize=12)
    #plt.xlim(10,10**-2)
    #plt.ylim((10**-10,10**5))
    #plt.legend(loc="upper right", fontsize=14)
    #plt.show()
    
    return n_to_p, p_to_n, H #returns arrays of the neutron to proton rate, the proton to neutron rate, and the Hubble
                             #rate over time