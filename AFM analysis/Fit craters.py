# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:29:11 2025

@author: Nilesh Dalla
"""

import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import seaborn as sns
from scipy.optimize import leastsq
from plotly.offline import plot
import plotly.graph_objs as go
from time import sleep


dpi=150
save_data=0

plt.rcParams.update({
    "font.size": 14,

    # Labels
    "axes.labelsize": 16,
    "axes.labelweight": "bold",

    # Title
    "axes.titlesize": 18,
    "axes.titleweight": "bold",

    # Tick labels
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.weight": "bold",   # general fallback
})


# Circle residuals
def calc_R(x, z, xc, zc):
    return np.sqrt((x - xc)**2 + (z - zc)**2)

def residuals_circle(params, x, z):
    xc, zc, R = params
    return calc_R(x, z, xc, zc) - R

def gaussian(x, A, x0, sigma, offset):
    return -A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords
    return A * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))
    ) + offset

filename='SampleT1_after2etchtreatment_crater_5,4_polydegbgrem.txt' #Crater5,5Laseroptic17_500ppm_only3degpolybackrem
data_file = "S:\\Topics\\Open_cavity\\Analysis\\Concave_shape_analysis\\Modified_Nilesh\\Sample T1 2025-08-06\\"+filename


# filename='sample _T1_crater_4-4_3polybgremoved.txt' #Crater5,5Laseroptic17_500ppm_only3degpolybackrem
# data_file = "S:\\Topics\\Open_cavity\\Analysis\\Concave_shape_analysis\\Modified_Nilesh\\Sample T1 2025-07-31\\"+filename


region_radius = 3  # in micrometers for fitting area


# Read manually skipping headers
with open(data_file, 'r') as f:
    lines = f.readlines()
# Filter out comment lines
data_lines = [line for line in lines if line.startswith('#')]
width=float(data_lines[1].split(' ')[2])  #in um
height=float(data_lines[2].split(' ')[2])  #in um

#data set
df = pd.read_csv(data_file, delim_whitespace=True, comment='#', header=None)

xaxis=np.linspace(0, width, num=df.shape[1])
yaxis=np.linspace(0, height, num=df.shape[0])

plt.figure(dpi=dpi)
plt.pcolormesh(xaxis,-yaxis,df, cmap='viridis')
plt.colorbar(label='Height (m)')
plt.xlabel("X (µm)")
plt.ylabel("Y (um)")
plt.title("Crater AFM")
#plt.tight_layout()
plt.show()

# Convert DataFrame to NumPy array
z = df.values*1E6 #um 

# Define physical window radius (in µm) to search around center
search_radius_um = 4.0

# Center coordinates in physical units
x_center = width / 2
y_center = height / 2

# Create masks to select center region
x_mask_center = (xaxis >= x_center - search_radius_um) & (xaxis <= x_center + search_radius_um)
y_mask_center = (yaxis >= y_center - search_radius_um) & (yaxis <= y_center + search_radius_um)

# Subset the data
z_center = z[np.ix_(y_mask_center, x_mask_center)]
xaxis_center = xaxis[x_mask_center]
yaxis_center = yaxis[y_mask_center]

# Find local minimum within center region
local_min_index = np.unravel_index(np.argmin(z_center), z_center.shape)
y_idx_local, x_idx_local = local_min_index

# Map back to global indices
y_idx = np.where(y_mask_center)[0][y_idx_local]
x_idx = np.where(x_mask_center)[0][x_idx_local]

# Get physical coordinates
x_min = xaxis[x_idx]
y_min = yaxis[y_idx]
z_min = z[y_idx, x_idx]

print(f"Crater near center at (x, y, z) = ({x_min:.2f} µm, {y_min:.2f} µm, {z_min:.2f} nm)")
# Find nearest indices to desired physical region
x_mask = (xaxis >= x_min - region_radius) & (xaxis <= x_min + region_radius)
y_mask = (yaxis >= y_min - region_radius) & (yaxis <= y_min + region_radius)

#x_sub = xaxis[x_mask]
#y_sub = yaxis[y_mask]
z_sub = z[np.ix_(y_mask, x_mask)]  # crop z using the mask
x_sub= np.linspace(0, 2*region_radius, num=z_sub.shape[1])
y_sub= np.linspace(0, 2*region_radius, num=z_sub.shape[0])


plt.figure(dpi=dpi)
plt.pcolormesh(x_sub, y_sub, z_sub, cmap='viridis')
plt.colorbar(label='Height (um)')
plt.xlabel("X (µm)")
plt.ylabel("Y (µm)")
plt.title("FIB 45 AFM data")
plt.show()

if save_data==1:
    df = pd.DataFrame(z_sub, index=y_sub, columns=x_sub)
    df.to_csv(f'{data_file.split(".")[0]}+afm_grid.csv')

center_row = z_sub.shape[0] // 2
center_col = z_sub.shape[1] // 2

horizontal_crossection = z_sub[center_row, :]  # along x-axis (left to right)
vertical_crossection = z_sub[:, center_col]    # along y-axis (top to bottom)

# Convert to nanometers
zv = vertical_crossection 
zh = horizontal_crossection 

x_h = x_sub
x_v = y_sub
z_h = horizontal_crossection
z_v = vertical_crossection

def fit_circle_algebraic(x, z):
    A = np.c_[2*x, 2*z, np.ones_like(x)]
    b = x**2 + z**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, zc = c[0], c[1]
    R = np.sqrt(c[2] + xc**2 + zc**2)
    return xc, zc, R

# Fit horizontal (x_h, z_h)
xc_h, zc_h, R_h = fit_circle_algebraic(x_h, z_h)

# Fit vertical (x_v, z_v)
xc_v, zc_v, R_v = fit_circle_algebraic(x_v, z_v)

print(f"Horizontal circle fit: Center = ({xc_h:.2f} µm, {zc_h:.2f} nm), Radius = {R_h:.2f} µm")
print(f"Vertical   circle fit: Center = ({xc_v:.2f} µm, {zc_v:.2f} nm), Radius = {R_v:.2f} µm")

def circle_arc_safe(x, xc, zc, R):
    inside = R**2 - (x - xc)**2
    inside[inside < 0] = 0  # avoid NaNs
    return zc - np.sqrt(inside)

# plt.figure()
# plt.plot(x_h, z_h, 'r.', label="Horizontal data")
# plt.plot(x_h, circle_arc_safe(x_h, xc_h, zc_h, R_h), 'r-', label=f"H fit (R = {R_h:.2f} µm)")
# plt.plot(x_v, z_v, 'b.', label="Vertical data")
# plt.plot(x_v, circle_arc_safe(x_v, xc_v, zc_v, R_v), 'b-', label=f"V fit (R = {R_v:.2f} µm)")
# plt.xlabel("Position (µm)")
# plt.ylabel("Height (µm)")
# plt.title("Robust Circle Fit to Crater Cross-Sections")
# plt.legend()
# plt.grid(True)
# plt.show()


# Example for horizontal Gaussian
p0_h = [np.min(z_h), x_h[np.argmin(z_h)], 1.0, np.max(z_h)]  # initial guess
params_h, pcov_h = curve_fit(gaussian, x_h, z_h, p0=p0_h)
A_h, x0_h, sigma_h, offset_h = params_h
# Standard errors from the covariance matrix
perr_h = np.sqrt(np.diag(pcov_h))
A_h_err, x0_h_err, sigma_h_err, offset_h_err = perr_h
fitted_y1 = gaussian(x_h, *params_h)
roc_h = sigma_h**2 / abs(A_h)  # µm
droc_h = np.sqrt(
    ((2 * sigma_h / abs(A_h)) * sigma_h_err)**2 +
    ((sigma_h**2 / (A_h**2)) * A_h_err)**2)



print(f"Horizontal Gaussian ROC ≈ {roc_h:.2f} µm")
print(f"Horizontal Gaussian ROC err ≈ {droc_h:.2f} µm")
plt.figure(dpi=dpi)
plt.scatter(x_h, z_h, color= 'orange', label='Horizontal data')
plt.plot(x_h, fitted_y1, color= 'orange', label=f"Fit ROC {roc_h:.2f} ")#, aspect='auto',extent=[0.4, 2, cavity_lengths[0], cavity_lengths[-1]], origin='lower'
plt.xlabel("X (µm)")
plt.ylabel("Height (µm)")
plt.legend()
#plt.grid(True)
plt.title("Gaussian fit")
#plt.show()


# Example for vertical gaussian
p0_v = [np.min(z_v), x_v[np.argmin(z_v)], 1, np.max(z_v)]  # initial guess A, x0, sigma, offset
params_v, pcov_v = curve_fit(gaussian, x_v, z_v, p0=p0_v)
A_v, x0_v, sigma_v, offset_v = params_v
# Standard errors from the covariance matrix
perr_v = np.sqrt(np.diag(pcov_v))
A_v_err, x0_v_err, sigma_v_err, offset_v_err = perr_v
fitted_y2 = gaussian(x_v, *params_v)
roc_v = sigma_v**2 / abs(A_v)  # µm
droc_v = np.sqrt(
    ((2 * sigma_v / abs(A_v)) * sigma_v_err)**2 +
    ((sigma_v**2 / (A_v**2)) * A_v_err)**2)

print(f"vertical Gaussian ROC ≈ {roc_v:.2f} µm")
print(f"Vertical Gaussian ROC err ≈ {droc_v:.2f} µm")
#plt.figure(dpi=150)
plt.scatter(x_v, z_v, color= 'green', label='Vertical data')
plt.plot(x_v, fitted_y2, color= 'green', label=f"Fit ROC {roc_v:.2f} µm")#, aspect='auto',extent=[0.4, 2, cavity_lengths[0], cavity_lengths[-1]], origin='lower'
plt.xlabel("Y (µm)")
plt.ylabel("Height (µm)")
plt.legend()
#plt.grid(True)
plt.title("Gaussian fit")
plt.show()

if save_data==1:
    new_df = pd.DataFrame({
        "H_axis": x_h,
        "H_data":z_h,
        "H_fit" :fitted_y1,
        "V_axis": x_v,
        "V_data":z_v,
        "V_fit" :fitted_y2,
    })
    new_df.to_csv(f'{data_file.split(".")[0]}+H-V_plots.csv', index=False)


#Lets try surface fitting now

# Create meshgrid from x_sub, y_sub
X, Y = np.meshgrid(x_sub, y_sub)
Z = z_sub  # already in um

# Flatten the mesh for fitting
xdata = np.vstack((X.ravel(), Y.ravel()))
zdata = Z.ravel()

#initial guess
A0 = np.min(Z) - np.max(Z)
x0 = x_sub[np.argmin(np.abs(x_sub - (x_sub.max()/2)))]
y0 = y_sub[np.argmin(np.abs(y_sub - (y_sub.max()/2)))]
sigma_x0 = sigma_y0 = (x_sub.max() - x_sub.min()) / 4
offset0 = np.max(Z)
p0 = [A0, x0, y0, sigma_x0, sigma_y0, offset0]

params, cov = curve_fit(gaussian_2d, xdata, zdata, p0=p0)
errors = np.sqrt(np.diag(cov))

A, x0, y0, sigma_x, sigma_y, offset = params
A_err, x0_err, y0_err, sigma_x_err, sigma_y_err, offset_err = errors

Z_fit = gaussian_2d((X, Y), *params).reshape(Z.shape)
residuals = (Z - Z_fit) *1000  #in nm

# plt.figure(dpi=150)
# plt.pcolormesh(x_sub, y_sub, Z_fit, cmap='viridis')
# plt.colorbar(label='Fit Height (nm)')
# plt.title("2D Gaussian Fit")
# plt.xlabel("X (µm)")
# plt.ylabel("Y (µm)")
# plt.show()

plt.figure(dpi=dpi)
plt.pcolormesh(x_sub, y_sub, residuals, cmap='bwr')
plt.colorbar(label='Residual (nm)')
plt.title("Fit Residuals")
plt.xlabel("X (µm)")
plt.ylabel("Y (µm)")
plt.show()

if save_data==1:
    df = pd.DataFrame(residuals, index=y_sub, columns=x_sub)
    df.to_csv(f'{data_file.split(".")[0]}+afm_grid_residuals.csv')

ROC_x = sigma_x**2 / abs(A)
ROC_y = sigma_y**2 / abs(A)
dROC_x = np.sqrt(
    ((2 * sigma_x / abs(A)) * sigma_x_err)**2 +
    ((sigma_x**2 / (A**2)) * A_err)**2
)

dROC_y = np.sqrt(
    ((2 * sigma_y / abs(A)) * sigma_y_err)**2 +
    ((sigma_y**2 / (A**2)) * A_err)**2
)
print("----Results from 3D Gaussian fitting----")
print(f"ROC_x: {ROC_x:.2f} ± {dROC_x:.2f} µm, ROC_y: {ROC_y:.2f} µm ± {dROC_y:.2f},")
print(f"Sigma_x: {sigma_x:.3f} ± {sigma_x_err:.3f}µm, sigma_y: {sigma_y:.3f}  {sigma_y_err:.3f} µm,")
print(f"Amplitude: {A:.3f} µm ± {A_err:.3f},")

rmse = np.sqrt(np.mean(residuals**2)) 
print(f"RMS Roughness: {rmse:.2f} nm,")
Ra = np.mean(np.abs(residuals))
print(f"Average Roughness (Ra) = {Ra:.2f} nm,")
Rz = np.max(residuals) - np.min(residuals)
print(f"Peak-to-Valley Height (Rz) = {Rz:.2f} nm")

# ========== PLOTLY 3D OVERLAY ==========

# # Original data points
# scatter = go.Scatter3d(
#     x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
#     mode='markers',
#     marker=dict(size=2, color=Z.ravel(), colorscale='Viridis', opacity=0.7),
#     name='AFM Data'
# )

# # Fitted Gaussian surface
# surface = go.Surface(
#     x=x_sub, y=y_sub, z=Z_fit,
#     colorscale='Viridis', opacity=0.6,
#     showscale=False,
#     name='Gaussian Fit'
# )

# fig = go.Figure(data=[surface, scatter])

# fig.update_layout(
#     scene=dict(
#         xaxis_title='X (µm)',
#         yaxis_title='Y (µm)',
#         zaxis_title='Height (um)'
#     ),
#     title=f"AFM Crater Fit (RMSE = {rmse:.3f} nm)",
#     width=800,
#     height=600
# )

# plot(fig, auto_open=True)

# sleep(1)

# fig = go.Figure(data=[go.Surface(z=residuals, x=x_sub, y=y_sub)])
# fig.update_layout(
#     title=f'Residuals fit with RMSE {rmse:.3f} nm ',
#     scene=dict(
#         xaxis_title='X (um)',
#         yaxis_title='Y (um)',
#         zaxis_title='Height (nm)'
#     )
# )

# plot(fig, auto_open=True)
