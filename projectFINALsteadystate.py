import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm4bem
from dm4bem import read_epw, sol_rad_tilt_surf


#Matrix A
#lines are flows/resistances and columns are nodes
#the last two lines are the controllers

A = np.zeros([28,21])
A[0, 0] = 1
A[1, 0], A[1, 1] = -1, 1
A[2, 1], A[2, 2] = -1, 1
A[3, 2], A[3, 3] = -1, 1
A[4, 3], A[4, 4] = -1, 1
A[5, 4], A[5, 5] = -1, 1
A[6, 6],A[6, 7] = 1, -1
A[7, 7],A[7, 8] = 1, -1
A[8, 8], A[8, 9] = 1, -1
A[9, 9], A[9, 10] = 1, -1
A[10, 10], A[10, 11] = 1, -1
A[11, 11]= 1
A[12, 6]= 1
A[13, 5] = 1
A[14, 5], A[14, 12] = 1, -1
A[15, 12], A[15, 13] = 1, -1
A[16, 13], A[16, 14] = 1, -1
A[17, 14], A[17, 15] = 1, -1
A[18, 5], A[18, 15] = 1, -1
A[19, 15] = 1
A[20, 15], A[20, 16] = 1, -1
A[21, 16], A[21, 17] = 1, -1
A[22, 17], A[22, 18] = 1, -1
A[23, 18], A[23, 19] = 1, -1
A[24, 19], A[24, 20] = 1, -1
A[25, 20]= 1
A[26, 5]= 1 #Controller 1
A[27, 15]= 1 #Controller 2

print('This is the incidence matrix A:')
print(A)

#Materials information
# lambda = thermal conductivity

#Concrete
lambda_concrete = 1.4 #W/mK
density_concrete = 2300 #kg/m^3
specheat_concrete = 880 #J/kgK
width_concrete = 0.02 #m

#Insulation
lambda_insulation = 0.04 #W/mK
density_insulation = 16 #kg/m^3
specheat_insulation = 1210 #J/kgK
width_insulation = 0.08 #m

#Window
uvalue_window = 1.4 #W/m2K
density_window = 2500 #kg/m^3
specheat_window = 1210#J/kgK
width_window = 0.04 #m
surface_window = 1

#Door
uvalue_door = 0.9 #W/m2K
density_door = 314 #kg/m^3
specheat_door = 2380 #J/kgK
width_door = 0.044 #m
surface_door = 1.66


#Walls
h_out = 10 #W/m^2*K convection coefficient between outdoor air and wall
h_in = 4 #W/m^2*K convection coefficient between indoor air and wall
l = 3 #length of walls 1,4,7
L = 4 #length of walls 2,3,5,6
surface_wall1 = L*3 #surface of walls 1,4,7
surface_wall2 = l*3 #surface of walls 2,3,5,6

#Controllers: if Kp = 0, the controller is off
Kp1 =10**5 
Kp2= 10**8

Tsp1 = 25
Tsp2 = 25


#Matrix G
#Diagonal matrix containing the conductances (inverse of the conduction and convection resistances)
# conduction conductance = (lambda*surface)/width
# convection conductance = (conv.coef.)*surface
g =[h_out*surface_wall2, 
     lambda_insulation*surface_wall2/width_insulation, #insulation of wall 2
     lambda_insulation*surface_wall2/width_insulation,
     lambda_concrete*surface_wall2/width_concrete,
     lambda_concrete*surface_wall2/width_concrete, #resistance 4 (just so i dont get lost)
     h_in*surface_wall2,
     h_in*surface_wall1, #wall 1
     lambda_concrete*surface_wall1/width_concrete,
     lambda_concrete*surface_wall1/width_concrete,
     lambda_insulation*surface_wall1/width_insulation,
     lambda_insulation*surface_wall1/width_insulation,
     h_out*surface_wall1, #res 11
     uvalue_door,
     uvalue_window,
     h_in*surface_wall1, #wall 7
     lambda_concrete*surface_wall1/width_concrete,
     lambda_concrete*surface_wall1/width_concrete,
     h_in*surface_wall1,
     uvalue_door,
     uvalue_window, #res 19
     h_in*surface_wall2, #wall 3
     lambda_concrete*surface_wall2/width_concrete,
     lambda_concrete*surface_wall2/width_concrete,
     lambda_insulation*surface_wall2/width_insulation,
     lambda_insulation*surface_wall2/width_insulation,
     h_out*surface_wall2,
     Kp1, #controllers
     Kp2]

G = np.diag(g)

print('This is the conductance matrix G:')
print(G)

#Thermal Capacities
C_concrete1=density_concrete*specheat_concrete*(width_concrete*surface_wall1)
C_concrete2=density_concrete*specheat_concrete*(width_concrete*surface_wall2)

C_insulation1= density_insulation*specheat_insulation*(width_insulation*surface_wall1)
C_insulation2= density_insulation*specheat_insulation*(width_insulation*surface_wall2)

density_air= 1.2
specheat_air=1000
volume_air= 3*l*L

C_air=density_air*specheat_air*volume_air

# Matrix C
#Diagonal matrix containing the capacities
C=np.zeros([21,21])

C[1,1]=C_insulation2   
C[3,3]=C_concrete2
C[5,5]=C_air
C[8,8]=C_concrete1
C[10,10]=C_insulation1
C[15,15]=C_air
C[17,17]=C_concrete2
C[19,19]=C_insulation2

print('This is the capacity matrix C:')
print(C)


#Vectors
#b: temperature sources, associated to the resistances
b = np.zeros(28)
b[[0,11,12,13,19,25]] = -5 #CONFERIR ISSO AQ acho q é pra ser a temp externa
b[26],b[27]= Tsp1, Tsp2
print(b)
#f: flow-rate sources, associated to the nodes
f = np.zeros(21)

#state-space model

omega = np.linalg.inv(np.transpose(A) @ G @ A)@ ((np.transpose(A) @ G @ b))
print(omega)

error1 = Tsp1 - omega[5]
error2 = Tsp2 - omega[15]

print("The error of controller 1 is:", error1, "the error of controller 2 is:", error2)

q1 = Kp1*error1
q2 = Kp2*error2

print("The flow of controller 1 is:", q1, "the flow of controller 2 is:", q2) #dividir pela area do quarto


