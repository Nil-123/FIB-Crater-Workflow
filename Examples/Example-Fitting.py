# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:28:42 2026

@author: Nilesh Dalla
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from AFM_analysis.Fit_craters import analyze_crater

#Make sure the AFM data correspond to the actual streamfile related crater 
result_from_AFM_fitting= analyze_crater(
    filepath="path\AFM_test_data.txt",
    region_radius=3, #This is fitting region radius
    save_data=1,
    show_plot=1
)