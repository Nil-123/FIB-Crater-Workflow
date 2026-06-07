# used to generate stream files with a gaussian milling pattern for FIB: FEI Helios NanoLab 400/400S/400 ML/600
#First version created by Pawel Kulboka later improved by Nilesh Dalla
import numpy as np
from math import floor
from random import shuffle
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
from plotly.offline import plot
import plotly.graph_objs as go
from scipy.optimize import curve_fit
import pandas as pd
from pathlib import Path

###

# crater shape        |..........size_um............|
#                _____                               _____...... substrate surface
#                     |                             |        |
#                     |                             |        | h_step_um - unavoidable initial step
#                     |____                     ____| .......|..
#                          -                  -         |
#                           \                /          |
#                            \              /           |
#                             \            /            | H_um
#                              -          -             |
#                               \_      _/              |
#                                 -____- ...............|...             
#
#
# volume milled per galium ions charge in um^3/nC:
# C 0.18 Au 1.50
# Si 0.27 MgO 0.15
# Al 0.30 SiO2 0.24
# Ti 0.37 Al2O3 0.08
# Cr 0.10 TiO 0.15
# Fe 0.29 Si3N4 0.20
# Ni 0.14 TiN 0.15
# Cu 0.25 Fe2O3 0.25
# Mo 0.12 GaAs 0.61
# Ta 0.32 Pt 0.23
# W 0.12 PMMA 0.40


def generate_streamfile(
    size_um,
    roc_um,
    depth_um,
    current,
    correction_factor_height,
    correction_factor_sigma,
    pixel_order,
    Output_folder,
    show_plot,
    comment
):
    script_version='V1'
    comment='Testing'
    ###user defined variables
    height_correction_method=1 
    #0- no correction, just use set_current
    #1- pick_depth_scaling_factor(set_current): height correction based on the empirical AFM results from previous millings, does not take into account the displayed value of the ion beam current
    #2- set/get_current correction: use the displayed value of the ion beam current during milling of a test pattern instead of the set_current value
    size_um = size_um#length in um of the edge of the square milling area
    RoC_um = roc_um #desired radius of curviture in um
    H_um = depth_um #desired depth of the crater, measured from the plateau after initial milling step
    max_dwell_time=250 #*0.1 us, the higher this value is, the smaller the initial step
    pattern_resolution=913 #keep it odd for the sake of symmetry
    path=Output_folder
    set_current = current #in Amps 
    get_current=26*10**(-12) #in Amps, measured while milling a testhole directly before main milling, used for 
    screen_filling_factor=0.75 #magnification will be chosen to make the milling area take ~this fraction of the visible area
    dose=0.24 # um^3/nC, value suggested by the manufacturer for SiO2
    ordering=['serpentine', 'random']
    correction_method_names=['none', 'pick_depth_scaling_factor(set_current)', 'set/get_current correction']
    iterator_file_path=Output_folder/"fibiterator.txt"
    #### hardware constants
    #x_range=65536
    y_range=56576
    dwell_time_precision=0 #only integer values allowed
    pixel_limit=1159 #keep it odd for the sake of symmetry
    min_dwell_time=1 #*0.1 us, hardware limit
    f=1950 #measured magnification*scaling(nm/px), used to calculate nm/px scaling for different magnifications
    magnifications=[1000, 1200, 1500, 2000, 2500, 3500, 5000, 6500]
    visible_area_shorter_edge_length=[y_range*f/m/1000 for m in magnifications]
    ###
    
    def pick_depth_scaling_factor(curr, correction_method):
        if correction_method==1:
            if curr== 9*10**(-12):
                return correction_factor_height   # 0.5976/1.2   =0.3/0.502 <- this comes form test milling with set/get current correction, and the obtained depth, 1.2 additional correction after second test milling
            elif curr==26*10**(-12):
                return correction_factor_height #=26/33
            else:
                return 1
        else:
            return 1
    
    #read last streamfile iterator and increase it by 1
    with open(iterator_file_path,'r') as iterator_file:
        it=int(iterator_file.readlines()[0])+1
    with open(iterator_file_path,'w') as iterator_file:
        iterator_file.write(str(it))
    
    #pick appropriate magnification for the given size_um and screen_filling_factor
    axis_filling_factor=np.sqrt(screen_filling_factor)
    possible_axis_filling_factors_delta=[]
    for vis_length in visible_area_shorter_edge_length:
        f_x=size_um/vis_length
        if f_x<0.95: #always leave at least 5% of shorter axe unmilled
            possible_axis_filling_factors_delta.append(abs(size_um/vis_length-axis_filling_factor))
    M=magnifications[possible_axis_filling_factors_delta.index(min(possible_axis_filling_factors_delta))]
    
    a=f/(M*1000) #um/px, scaling constant for the selected magnification
    if pattern_resolution>pixel_limit:
        pattern_resolution=pixel_limit
        print('Caution: desired resolution too high, pattern resolution lowered to pixel limit = 1159')
    desired_size_FIB_coor=round(size_um/a) #desired size of the milling area in FIB coordinates, from this range we need to take pixel_limit number of points
    spacing=round(desired_size_FIB_coor/(pattern_resolution-1)) #find a spacing between nodes in a grid then aproximately covers the whole area of size_um**2
    XY=[i*spacing for i in range(pattern_resolution)]  #the grid, X and Y in the same range
    sigma_um=correction_factor_sigma*np.sqrt(H_um*RoC_um)
    center_point_grid=floor(pattern_resolution/2)
    sigma_grid=sigma_um/a/spacing #desired sigma expressed in terms of grid nodes, (sigma_um/a) - size of sigma in FIB coordinates
    
    def gausian(center, sigma, x, y):   #all inputs expressed in terms of the the grid coordinates
        return min_dwell_time+(max_dwell_time-min_dwell_time)*np.exp(-0.5*((x-center)**2+(y-center)**2)/sigma**2)
    
    def text_to_bitmap(text, size):
        # create a blank image with a white background
        img = Image.new('1', size, 1)
    
        # create a drawing object
        draw = ImageDraw.Draw(img)
    
        # set the font and font size
        font_size = 1
        font = ImageFont.truetype('arial.ttf', font_size)
    
        # loop until the text fits almost all of the available area
        while True:
            # calculate the new font size
            new_font_size = font_size+1
            new_font = ImageFont.truetype('arial.ttf', new_font_size)
            # calculate the text bbox and size
            bbox = draw.textbbox((0, 0), text, font=new_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # calculate the scaling factor
            width_scale = size[0] / text_width
            height_scale = size[1]/ text_height
            scale = min(width_scale, height_scale)
            # print(text_width, text_height, width_scale, height_scale, scale)
    
    
            if scale < 1:
                # print(scale, 'scale < 1, box too big')
                break
    
            # set the new font size and keep track of the previous values
    
            font_size = new_font_size
            # print(font_size) 
            font = ImageFont.truetype('arial.ttf', font_size)
    
        # calculate the starting position for the text
        bbox = draw.textbbox((0, 0), text, font=font)
        x = 0#(size[0] - (bbox[2] - bbox[0]) * scale) / 2
        y = -bbox[1]#(size[1] - (bbox[3] - bbox[1]) * scale) / 2
        # print(f'final font size:{font_size}')
        # draw the text on the image
        draw.text((x, y), text, font=font, fill=0)
        # img.show()
        bitmap = np.asarray(img)
        # convert the image to a bitmap and return it
        return 1-bitmap, bbox  # invert the bitmap so that black pixels have value 1 and white pixels have value 0
    
    
    #filling the grid
    pixels=np.zeros((pattern_resolution,pattern_resolution))
    for x in range(pattern_resolution):
        for y in range(pattern_resolution):
            pixels[x,y]=round(gausian(center_point_grid, sigma_grid, x, y))
            
    xa = np.linspace(0, size_um, pattern_resolution)
    ya = np.linspace(0, size_um, pattern_resolution)
    x, y = np.meshgrid(xa, ya)
    
    
    label_text=f'FIB{it}'
    label_array, bbox=text_to_bitmap(label_text, (pattern_resolution, round(pattern_resolution*0.1)))
    label_array[0,0]=1
    label_array[-1,0]=1
    label_array[0,bbox[2]]=1
    label_array[-1,bbox[2]]=1
    label_depth=round(max_dwell_time/2)
    for xi in range(pattern_resolution):
        for yi in range(round(pattern_resolution*0.1)):
            pixels[xi,yi]+=label_array[yi,xi]*label_depth
    
    
    one_pass_milling_time=0
    for x in range(pattern_resolution):
        for y in range(pattern_resolution):
            one_pass_milling_time+=pixels[x,y]*0.0001 #in ms
    
    #calculating number of passes
    if height_correction_method!=2:
        get_current=set_current
    expected_size_um=XY[-1]*a #should be close to size_um, based on the chosen grid, does not take into account the diameter of the ion beam
    print(f'expected size of the milled area- {expected_size_um:.2f} um')
    # assuming relation between height of the crater and height of the initial step: H=((max_dwell_time-min_dwell_time)/min_dwell_time)*initial_step_size
    H_factor=pick_depth_scaling_factor(set_current, height_correction_method)
    h_step_um=H_um*H_factor*min_dwell_time/(max_dwell_time-min_dwell_time)
    step_volume=h_step_um*expected_size_um**2
    step_charge=(step_volume/dose)*10**(-9) #in Coulombs
    part_of_total_time_spent_milling_initial_step=step_charge/get_current
    time_spent_milling_initial_step_per_1pass=(min_dwell_time*0.1*10**(-6))*pattern_resolution**2
    number_of_passes=round(part_of_total_time_spent_milling_initial_step/time_spent_milling_initial_step_per_1pass)
    total_milling_time=number_of_passes*one_pass_milling_time/1000 #in s
    
    #saving output to a stream file and updating database
    streamfile_name=f'FIB{it}_'+"M-"+str(M)+"_i-"+f'{set_current:.1e}'+'_H-'+str(H_um)+"_RoC-"+str(RoC_um)+".str"
    file=open(path/streamfile_name, 'w')
    file.write('s16\n')
    file.write(str(number_of_passes)+'\n')
    file.write(str((pattern_resolution**2)+1)+'\n')
    
    if pixel_order==0: #serpentine
        for x in range(pattern_resolution):
            if x%2==0:
                for y in range(pattern_resolution):
                    current_line=f'{pixels[x,y]:.{dwell_time_precision:.0f}f} {XY[x]} {XY[y]}'
                    file.write(current_line+'\n')
            if x%2==1:
                for y in reversed(range(pattern_resolution)):
                    current_line=f'{pixels[x,y]:.{dwell_time_precision:.0f}f} {XY[x]} {XY[y]}'
                    file.write(current_line+'\n')
        file.write(current_line+' 0')
    if pixel_order==1: #random
        coordinates=[]
        for x in range(pattern_resolution):
            for y in range(pattern_resolution):
                coordinates.append((x,y))
        shuffle(coordinates)
        for xy in coordinates:
            x,y=xy
            current_line=f'{pixels[x,y]:.{dwell_time_precision:.0f}f} {XY[x]} {XY[y]}'
            file.write(current_line+'\n')
        file.write(current_line+' 0')
    file.close()
    
    
    database_path = Path(path) / "Streamfile_database.csv"
    record = {
    "streamfile_name": streamfile_name,
    "date": date,
    "depth_um": H_um,
    "roc_um": RoC_um,
    "sigma_um": sigma_um,
    "max_dwell_time": max_dwell_time,
    "magnification": M,
    "set_current_A": set_current,
    "screen_filling_factor": screen_filling_factor,
    "dose_um3_per_nC": dose,
    "pixel_order": ordering[pixel_order],
    "target_size_um": size_um,
    "actual_size_um": expected_size_um,
    "resolution_px": pattern_resolution,
    "spacing_um": spacing * a,
    "spacing_fib_px": spacing,
    "height_correction_method": correction_method_names[height_correction_method],
    "get_current_A": get_current,
    "depth_scaling_factor": H_factor,
    "num_passes": number_of_passes,
    "one_pass_time_ms": one_pass_milling_time,
    "total_milling_time_s": total_milling_time,
    "script_version": script_version,
    "comments": comment
    }
    
    new_row = pd.DataFrame([record])
    
    if database_path.exists():
        new_row.to_csv(
            database_path,
            mode="a",
            header=False,
            index=False
        )
    else:
        new_row.to_csv(
            database_path,
            mode="w",
            header=True,
            index=False
        )

    #------Fitting part starts here----
    
    # Simulated depth (in microns), scale so max dwell = max depth
    depth_max = H_um  # e.g., 0.3 µm desired crater depth
    depth = (pixels - pixels.min()) / (pixels.max() - pixels.min()) * depth_max
    Z = -depth  # negative = downward crater
    
    # Create meshgrid
    X, Y = np.meshgrid(xa, ya)
    
    # Flatten for fitting
    xdata = np.vstack((X.ravel(), Y.ravel()))
    zdata = Z.ravel()
    
    # 2D Gaussian model
    def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, offset):
        x, y = coords
        return A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2))) + offset
    
    # Initial guess
    A0 = np.min(Z) - np.max(Z)
    x0 = size_um / 2
    y0 = size_um / 2
    sigma0 = size_um / 4
    offset0 = np.max(Z)
    p0 = [A0, x0, y0, sigma0, sigma0, offset0]
    
    # Fit
    params, _ = curve_fit(gaussian_2d, xdata, zdata, p0=p0)
    A, x0, y0, sigma_x, sigma_y, offset = params
    
    # Reconstruct fit
    # Z_fit = gaussian_2d((X, Y), *params).reshape(X.shape)
    # ROC_x = sigma_x**2 / abs(A)
    # ROC_y = sigma_y**2 / abs(A)
    # print(f"Expected depth: {A:.2f} µm")
    # print(f"Radius of Curvature (ROC_x): {ROC_x:.2f} µm")
    # print(f"Radius of Curvature (ROC_y): {ROC_y:.2f} µm")
    # print(f"Sigma X: {sigma_x:.2f} µm")
    # print(f"Sigma_Y: {sigma_y:.2f} µm")
    
    if show_plot==True:
        fig = go.Figure()
        
        # Actual simulated crater
        fig.add_trace(go.Surface(z=Z, x=xa, y=ya, colorscale='Viridis', opacity=0.7, name="Simulated Crater"))
        
        # Fitted surface
        #fig.add_trace(go.Surface(z=Z_fit, x=xa, y=ya, colorscale='Plasma', opacity=0.5, name="Gaussian Fit"))
        
        fig.update_layout(
            title="Generated Crater profile",
            scene=dict(
                xaxis_title='X (µm)',
                yaxis_title='Y (µm)',
                zaxis_title='Depth (µm)'
            ),
            width=800, height=600
        )
        
        plot(fig, auto_open=True)
        
    
    
    #---Fitting part ends---
    print("\n --::Streamfile generation ended::-- \n")

    return {
        "streamfile": streamfile_name,
        "passes": number_of_passes,
        "estimated_time": total_milling_time,
        "roc_target": roc_um,
    }
