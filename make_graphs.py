import matplotlib.pyplot as plt
import numpy as np


class_const_errors = [0.30529765, 0.3134899, 0.3134899, 0.32004369, 0.31785909, 0.3123976
, 0.30311305, 0.30311305, 0.30420535, 0.3080284, 0.30966685, 0.30966685
, 0.30966685, 0.29983616, 0.29983616, 0.29983616, 0.28618241, 0.28618241
, 0.28836701, 0.28509011, 0.28509011, 0.27525942, 0.25395958, 0.25395958
, 0.25395958, 0.28181322, 0.28181322, 0.25942108, 0.22719825, 0.22719825
, 0.2266521, 0.2266521, 0.2310213, 0.2310213, 0.2310213, 0.2310213
, 0.24085199, 0.20917531, 0.22392135, 0.22392135, 0.22392135, 0.21845986
, 0.21845986, 0.21845986, 0.21518296, 0.21845986, 0.21845986, 0.21845986
, 0.21845986, 0.25395958, 0.27307482, 0.27307482, 0.27580557, 0.27580557
, 0.28782086, 0.3123976, 0.3123976, 0.3123976, 0.31294375, 0.31294375
, 0.31294375, 0.31294375, 0.31294375, 0.31294375, 0.3047515, 0.32058984
, 0.30638995, 0.30638995, 0.30638995, 0.30638995, 0.30966685, 0.3080284
, 0.33752048, 0.34844347, 0.36865101, 0.36865101, 0.3828509, 0.3828509
, 0.3828509, 0.3828509, 0.38230475, 0.38230475, 0.38230475, 0.38230475
, 0.38448935, 0.3850355, 0.3894047, 0.3894047, 0.3894047, 0.3894047
, 0.3894047, 0.3894047, 0.3894047, 0.38995085, 0.38995085, 0.38995085
, 0.38995085, 0.36428181, 0.36428181]

class_asc_errors = [ 0.30529765, 0.3134899, 0.3134899, 0.32004369, 0.31785909, 0.3123976
, 0.30311305, 0.30311305, 0.30311305, 0.31075915, 0.310213, 0.310213
, 0.310213, 0.30202075, 0.30202075, 0.30202075, 0.28563626, 0.28563626
, 0.28345167, 0.27307482, 0.27307482, 0.28290552, 0.25177499, 0.25177499
, 0.25177499, 0.27799017, 0.30529765, 0.3014746, 0.23975969, 0.29109776
, 0.28454397, 0.29601311, 0.3014746, 0.25341344, 0.28891316, 0.25669033
, 0.27580557, 0.22392135, 0.22610595, 0.22610595, 0.22610595, 0.22228291
, 0.24139814, 0.25232114, 0.24303659, 0.24412889, 0.29929001, 0.28672856
, 0.28399782, 0.33151283, 0.33752048, 0.33752048, 0.33478973, 0.33533588
, 0.33533588, 0.33697433, 0.33697433, 0.34134353, 0.33697433, 0.33588203
, 0.33806663, 0.33806663, 0.34407428, 0.34243583, 0.34025123, 0.35062807
, 0.34735117, 0.26597488, 0.32386674, 0.32386674, 0.32495904, 0.32113599
, 0.34025123, 0.35172037, 0.35608957, 0.35608957, 0.36646641, 0.36755871
, 0.35991262, 0.35991262, 0.35663572, 0.35772802, 0.36045877, 0.33806663
, 0.36100492, 0.36100492, 0.36701256, 0.36701256, 0.36919716, 0.36810486
, 0.36919716, 0.37192791, 0.37028946, 0.37793555, 0.3773894, 0.3773894
, 0.38121245, 0.37575096, 0.37465866]

class_desc_errors = [0.30529765, 0.3134899, 0.32495904, 0.32441289, 0.32386674, 0.31785909
, 0.30857455, 0.30857455, 0.30857455, 0.30857455, 0.30966685, 0.30966685
, 0.30966685, 0.30966685, 0.30966685, 0.30966685, 0.27416712, 0.27416712
, 0.27416712, 0.25559803, 0.25559803, 0.25559803, 0.25559803, 0.25559803
, 0.25559803, 0.25559803, 0.25559803, 0.25559803, 0.25559803, 0.25559803
, 0.25559803, 0.25559803, 0.29437466, 0.29437466, 0.29437466, 0.29437466
, 0.29437466, 0.23375205, 0.23375205, 0.23375205, 0.23375205, 0.23375205
, 0.23375205, 0.23375205, 0.23375205, 0.23375205, 0.23375205, 0.23375205
, 0.23375205, 0.23375205, 0.23375205, 0.23375205, 0.23375205, 0.23375205
, 0.26051338, 0.26051338, 0.26051338, 0.26051338, 0.26051338, 0.26051338
, 0.26051338, 0.26051338, 0.26051338, 0.26051338, 0.26051338, 0.26051338
, 0.26051338, 0.26051338, 0.26051338, 0.26051338, 0.26051338, 0.26051338
, 0.26051338, 0.26051338, 0.26051338, 0.26051338, 0.26051338, 0.26051338
, 0.26051338, 0.26051338, 0.21845986, 0.21845986, 0.21845986, 0.21845986
, 0.21845986, 0.21845986, 0.21845986, 0.21845986, 0.21845986, 0.21845986
, 0.21845986, 0.21845986, 0.21845986, 0.21845986, 0.21845986, 0.21845986
, 0.21845986, 0.21845986, 0.21845986]

class_nopert_errors = [0.30529765, 0.28399782, 0.18569088, 0.1463681, 0.11305298, 0.10704533
, 0.10267613, 0.10376843, 0.09721464, 0.10049153, 0.09939924, 0.09939924
, 0.09393774, 0.09393774, 0.09721464, 0.09666849, 0.09448389, 0.09120699
, 0.09393774, 0.09120699, 0.09120699, 0.09120699, 0.09120699, 0.06990715
, 0.09175314, 0.09120699, 0.09120699, 0.09120699, 0.09120699, 0.09120699
, 0.0704533, 0.0704533, 0.06990715, 0.07099945, 0.06881486, 0.0704533
, 0.0704533, 0.0704533, 0.06990715, 0.04096122, 0.04369197, 0.04314582
, 0.04259967, 0.06772256, 0.04369197, 0.04369197, 0.04369197, 0.04423812
, 0.04369197, 0.04423812, 0.04369197, 0.04369197, 0.04369197, 0.04423812
, 0.04806117, 0.04806117, 0.04915347, 0.04806117, 0.04696887, 0.04642272
, 0.04587657, 0.04806117, 0.04860732, 0.05079192, 0.05024577, 0.04969962
, 0.04751502, 0.04751502, 0.04696887, 0.04751502, 0.04751502, 0.04751502
, 0.04751502, 0.04751502, 0.04751502, 0.04696887, 0.04751502, 0.04751502
, 0.04696887, 0.04696887, 0.04696887, 0.04696887, 0.04696887, 0.04587657
, 0.04696887, 0.04751502, 0.04751502, 0.04751502, 0.04696887, 0.04860732
, 0.04915347, 0.05024577, 0.04969962, 0.05024577, 0.04915347, 0.04915347
, 0.04969962, 0.05133807, 0.05133807]

time_pts = np.arange(1, 100)

fig, ax = plt.subplots()
ax.plot(time_pts, class_const_errors, '-', label='constant')
ax.plot(time_pts, class_asc_errors, '-', label='INC')
ax.plot(time_pts, class_desc_errors, '-', label='DEC')
ax.plot(time_pts, class_nopert_errors, '-', label='no perturbation')

plt.ylim(ymax=0.5)
legend = ax.legend(loc='upper left', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')

plt.show()