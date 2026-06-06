# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:28:42 2026

@author: Nilesh Dalla
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from FIB_generator.Generator import generate_streamfile

desired_depth=0.3 #um Desired depth
Correction_factor=1 #Height Correction factor derived from AFM session
results_for_streamfile=generate_streamfile(
    Output_folder="path\result\\",
    size_um=15, #um size of the milling square area
    roc_um=18,  #um ROC for milling
    depth_um=desired_depth, #um Desired depth
    current=26e-12, # Current used
    correction_factor=Correction_factor, #Height Correction factor derived from AFM session
    pixel_order=0, # Movement of beam, 0- serpentine, 1- random
    Show_Plot=1,  #Turn 0 to turn off plots
    comment='Testing'
)
