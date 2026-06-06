

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:28:42 2026

@author: Nilesh Dalla
"""
# For easy run through IDE like spyder
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

#Importing functionalities
from FIB_generator.Generator import generate_streamfile
from AFM_analysis.Fit_craters import analyze_crater
import numpy as np
from dataclasses import dataclass,asdict


@dataclass
class CraterDesign:

    size_um: float
    roc_um: float
    depth_um: float
    current: float
    correction_factor_height: float 
    correction_factor_sigma: float 
    pixel_order: int # Movement of beam, 0- serpentine, 1- random
    Show_Plot: bool  #Turn 0 to turn off plots
    comment: str
    @property
    def sigma(self):
        return np.sqrt(
            self.depth_um *
            self.roc_um
        )


def main():
    # #Streamfile generation paramters
    design = CraterDesign(
    size_um=15, #um size of the milling square area
    roc_um=18, #um ROC for milling
    depth_um=0.3,  #um Desired depth
    current=26e-12,  # Current used for milling
    correction_factor_height=1,  #Height Correction factor derived from AFM session, varies with current, material, beam profile
    correction_factor_sigma=1,   #Sigma Correction factor derived from AFM session, varies with current, material, beam profile
    pixel_order=1, # Movement of beam, 0- serpentine, 1- random
    Show_Plot=False,  #Turn False to turn off plots
    comment='Testing' )  #Any specific comments associated with the streamfile.
    
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATABASE_DIR = PROJECT_ROOT / "Databases"
    EXAMPLE_DIR = PROJECT_ROOT / "Examples"
    
    results_from_streamfile_generation = generate_streamfile(
        Output_folder=DATABASE_DIR,
        **asdict(design)
    )
    
    # AFM fitting parameters
    # This part assumes the AFM data provided is properly levelled.  
    AFM_fit = {
        "region_radius": 5, #This is fitting region radius
        "save_data": False,  #Save AFM fit data to txt files, False for not saving
        "show_plot": False,  #Turn False to turn off plots
        "save_data_directory": DATABASE_DIR, #Directory to save AFM fit data
    }
    
    #Make sure the AFM data correspond to the actual streamfile related crater 
    result_from_AFM_fitting= analyze_crater(
        filepath=EXAMPLE_DIR / "AFM_test_data.txt"
        ,**AFM_fit
    )
    
    desired_depth=design.depth_um
    Obtained_depth=np.abs(result_from_AFM_fitting["A"])
    Suggested_height_correction_factor=abs(desired_depth/Obtained_depth)
    desired_sigma=design.sigma
    measured_sigma = (result_from_AFM_fitting["sigma_x"]+result_from_AFM_fitting["sigma_y"] )/2
    Suggested_sigma_correction_factor=abs(desired_sigma/measured_sigma) 
    
    print("\n===== FIB WORKFLOW =====")
    print(f"Target depth       : {design.depth_um:.3f} µm")
    print(f"Measured depth     : {Obtained_depth:.3f} µm")
    print(f"Old correction height factor   : {design.correction_factor_height:.3f}")
    print(f"Suggested correction height factor: {Suggested_height_correction_factor:.3f}")
    print(f"Target sigma       : {desired_sigma:.3f} µm")
    print(f"Measured sigma     : {measured_sigma:.3f} µm")
    print(f"Old correction sigma factor   : {design.correction_factor_sigma:.3f}")
    print(f"Suggested correction sigma factor: {Suggested_sigma_correction_factor:.3f}")
        
if __name__ == "__main__":
    main()



