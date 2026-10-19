import os
import json
import ast

### Pausing the optimizer (e.g. to use Fusion without lag) ###
# While this script is running, create PAUSE.flag next to this file to pause
# before the next iteration starts, delete it to resume. No restart needed -
# optimizer state (DE population / L-BFGS-B history) stays intact.
#
# Pause (PowerShell):
#   New-Item "C:\Users\Gaming pc\Documents\GitHub\SurroundSimulationUltraShallo\PAUSE.flag" -ItemType File
#
# Resume (PowerShell):
#   Remove-Item "C:\Users\Gaming pc\Documents\GitHub\SurroundSimulationUltraShallo\PAUSE.flag"


###Change these environment variables to allow multi-core solving
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"  
os.environ["NUMEXPR_NUM_THREADS"] = "16"


import numpy as np
from felupe.constitution.tensortrax.models.hyperelastic import mooney_rivlin
from AnalysisFuncs_Axisym import *
#from RunSubprocess import ModEx
from scipy.optimize import minimize, differential_evolution
from SurroundClasses import *
import traceback
import time
import psutil
import subprocess
#import sys
from pathlib import Path




###
#define user things
###

# Optimization CAD parameters - single source of truth. Each entry is
# (name, lower_bound, upper_bound, initial_guess). `name` must match a user
# parameter in the Fusion design. Add or remove a parameter by adding/removing
# a line here - bounds, x0, and the name->value mapping sent to Fusion are all
# derived from this automatically, so nothing else needs to be touched.
#
# To temporarily drop a parameter from the optimization (hold it fixed),
# comment out its line here - just make sure the Fusion design still has a
# sensible value/expression for it, since it won't be driven by this script.
OptimizationParams = [
    ("InnerThick",       1.2,  3.5,    3.5),
    ("MidThick",           1,  3.5,    1.25),
    ("OuterThick",       1.2,  3.5,    3.37),
    ("OuterLaunchAngle",   91,   170,  101.5),
    ("InnerLaunchAngle",   91,   170,  90.96),
    ("InnerMountThick",   1,   3.5,  2.2), 
    ("OuterMountThick",   1,   3.5,  2.2), 
    #("ApexOffset",         4,    30,   17),
    #("ApexDepth",          4,    15,   17),
    #("MidSplineWeight",    0.1,  7,    6),
    ("InnerSplineWeight",  1,    40,   10.2),
    ("OuterSplineWeight",  1,    40,   20.38),
    # ("SurroundApexOffsetRange", -5, 5, -0.2),  # Not using anymore
    # ("ConeEnclosureGapRange", 30, 50, 47.96),
    # ("ApexSplineWeightRange", 4, 8, 6),
    # ("ConeSplineWeightRange", 15, 25, 20),
]

ParamNames = [p[0] for p in OptimizationParams]

def BuildParams(OptP):
    ##Zips the current optimizer vector OptP together with ParamNames, in the
    ##same order as OptimizationParams above - this is the only place OptP's
    ##positions are turned into named (name, value) pairs.
    return list(zip(ParamNames, OptP))

#Non-optimization geometry parameters and such 
NOPs = NonOptimParams()


NOPs.ConeCornerRadius = 111.406  # radius of the vertical cone-mount wall
NOPs.ConeFlangeWidth = 1.0  # radial extent of the small cone-side flange, inward from ConeCornerRadius
NOPs.ConeOffset = 1.6  # no longer used by CreateBCsAxisymmetric (kept for the old 3D CreateBCs)
NOPs.MountingGap = 0.75
NOPs.MountFlangeThickness = 1.6
NOPs.ConeEnclosureGap = 32.858
NOPs.DebugShowBCs = True  # TEMPORARY: saves Plots/debug_bc_dir<1|-1>.png every iteration - set False (or delete) once verified

#Other things
NOPs.TriggerPath = Path(r"C:\Users\Gaming pc\Documents\GitHub\SurroundSimulation\NeighborhoodWatch\run.trigger")
NOPs.stepout_path = r"C:\Users\Gaming pc\Documents\GitHub\SurroundSimulationUltraShallo\SurroundMutation.step"
NOPs.ExportComponentName = "OptimizationXC"  # export only this Fusion component instead of the whole design
NOPs.Xmax = 20 #mm one way
NOPs.TargetStiffness = 5 #N/mm
# NOTE: ScoreKmsAxisymmetric reads weights by name from this dict - edit these
# freely, no need to keep any particular order. Carried over from the old
# rectangular weights and haven't been retuned for the round design.
NOPs.OptimizationWeights = {
    "KmsFlatness": 5e4,   # flatness of the stroke between the two end ramps
    "RampUp": 1e3,        # shortfall of end-region stiffness below EndStiffnessMultiplier x mid-region avg
    "Shift": 1e2,        # deviation of mid-region avg stiffness from TargetStiffness
}
NOPs.EndStiffnessMultiplier = 6.0  # target end-region stiffness = mid-region avg * this
NOPs.EndRegionFraction = 0.10  # fraction of the stroke (each side) treated as the "end" region
NOPs.MaterialCoefficients = [0.5, 0.15] #C10, C01 butyl rubber estimated
NOPs.MeshFine = 0.75
NOPs.MeshCoarse = 1.5
NOPs.N_Steps = 20
NOPs.K_clamp = 100
NOPs.Node_find_tol = 1e-2
NOPs.maxfevPow = 3000
NOPs.maxiterPow = 800
NOPs.popsizeDE = 100
NOPs.maxfevDE = 5000
FusionExe = os.path.expandvars(r"C:\Users\%USERNAME%\AppData\Local\Autodesk\webdeploy\production\10477bbe50cc169c7bd2cee9059bc7c9d0b71ec0\Fusion360.exe")

##### Things I may want to change each time
#

TrackingFile = Path(__file__).parent / "TrackedBestsMTX818.txt"
NOPs.CachingFile = Path(__file__).parent / "CachedSolutions818.txt"
PauseFile = Path(__file__).parent / "PAUSE.flag"  # create this file to pause the optimizer, delete it to resume
Algorithm = "Both" ##Options are "DE", "L-BFGS-B", or "Both"
Resuming = False  ##If true, then take the best x from the trackedbests file and start from there
NOPs.IncludeSpider = False ##include spider stiffness in scoring and optimizing or not
AnalyzeOnly = True ## Do this to just analyze a solution, not optimize
# Keyed by name, not position - only needs entries for the params you want to
# override for this one-off analysis. Any name not listed (or no longer in
# OptimizationParams) falls back to that parameter's guess value above.
AnalysisOnlyParams = {
    "MidThick": 2.58,
    "InnerThick": 1.57,
    "OuterThck": 2.51,
    "OuterLaunchAngle": 135.45,
    "InnerLaunchAngle": 136.8,
    "InnerMountThick": 3.25,
    "OuterMountThick": 2.41,
    "InnerSplineWeight": 19.3,
    "OuterSplineWeight": 29.8,
}
#
######3

Iter = 0

## bounds and x0 are both derived from OptimizationParams above - edit that
## list, not these.
bounds = [(p[1], p[2]) for p in OptimizationParams]
x0 = np.array([p[3] for p in OptimizationParams], dtype=float)

#global vars to track best solution if optimizer doesn't converge
best_x = x0
best_score = float('inf')

def wait_if_paused():
    ##Blocks here as long as PauseFile exists. Create it (any empty file
    ##named PAUSE.flag next to this script) to pause before the next
    ##iteration's Fusion/mesh/solve work starts - nothing runs in the
    ##background while paused. Delete it to resume.
    if PauseFile.exists():
        print(f"Paused - delete {PauseFile} to resume...")
        while PauseFile.exists():
            time.sleep(2)
        print("Resuming.")

def objective(OptP, NOPs):

    global best_x, best_score
    wait_if_paused()
    try:
        global Iter 

        #params used to create surround geometry
        params = BuildParams(OptP)

        print(params)
        
        #This modifies the cad file and exports as a step
       
        CreateTrigger(params)  ##This creates a trigger file, which tells fusion to open the surround file and make changes according to what's in the trigger file, then export
        WaitOnFusion()
        
        
        #to find the fitness score of the surround
        Kms, Disp = AnalyzeItBothWaysDualSurround(NOPs)

        SurroundScore, WeightedScores = ScoreKmsAxisymmetric(Kms, Disp, NOPs)

        if SurroundScore < best_score:
            best_score = SurroundScore
            best_x = OptP.copy()
            # print("Tracking filepath")
            # print(TrackingFile)

            with open(TrackingFile, 'a') as f:
                f.write(f"score: {best_score}\n")
                f.write(f"params: {best_x.tolist()}\n")

        with open(NOPs.CachingFile, 'a') as f:
                f.write(f"scores: {WeightedScores}\n")
                f.write(f"score: {SurroundScore}\n")
                f.write(f"params: {params}\n")

        print('Current score:')
        print(SurroundScore)
        print('Breakdown')
        print(WeightedScores)
        print("Just finished iteration:", Iter)
        Iter += 1

    except Exception as e:
        print(f"Skipping invalid parameter set {OptP}, reason: {e}")
        traceback.print_exc()
        Iter += 1
        return 1e9
        
    return SurroundScore

def FinishOut(OptP):
    params = BuildParams(OptP)

    CreateTrigger(params)
    WaitOnFusion()

    ShowItOffAxisymmetric(NOPs, params)

    return 0

def PointlessCB(j):
    print("this is pointless")

def CreateTrigger(Parameters):
    ##Creates the trigger file that will tell fusion what to build. Also deletes the old step file for timing purposes
    StepFile = Path(NOPs.stepout_path)
    
    if StepFile.exists():  ##delete if present
        safe_unlink(StepFile)

    Parameters = [(name, float(val)) for name, val in Parameters]

    if not fusion_running():
        subprocess.Popen([FusionExe])
        print("opening fusion360. wait 10 sec before continuing")
        time.sleep(10)

    data = {
        "parameters": dict(Parameters),
        "CloudFile": "MTX/Ultra Shallow - 9500/SurroundOnly",
        "OutputPath": str(StepFile),
        "ComponentName": NOPs.ExportComponentName
    }

    NOPs.TriggerPath.write_text(json.dumps(data, indent=4))

    return None

def fusion_running():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] and 'Fusion360' in proc.info['name']: 
            return True
    return False

def WaitOnFusion(timeout = 100, poll=1):
    start = time.time()
    WatchFile = Path(NOPs.stepout_path)
    
    while not WatchFile.exists():
        if time.time() -start > timeout:
            raise TimeoutError("Fusion took too long")
        if WatchFile.exists():
            try:
                os.replace(WatchFile, WatchFile)
                return
            except PermissionError:
                pass

        time.sleep(poll)

def AnalysisOnlyToArray(overrides):
    ##Turns the name-keyed AnalysisOnlyParams dict into an array ordered to
    ##match OptimizationParams, falling back to each param's guess value if
    ##not present in overrides.
    return np.array([overrides.get(name, guess) for name, low, high, guess in OptimizationParams], dtype=float)

def main():
    global best_x, best_score, x0

    if AnalyzeOnly == True:
        FinishOut(AnalysisOnlyToArray(AnalysisOnlyParams))
        return 0
    
    if Resuming == True:
        with open(TrackingFile, 'r') as f:
            lines = f.readlines()
        try:
            last_params_line = [l for l in lines if l.startswith("params:")][-1]
        
            last_params = np.array(ast.literal_eval(last_params_line.split("params:")[1].strip()))

            print('resuming from last time')
            print(last_params)
            x0 = last_params
        except:
            print("no params found")
        
    else: 
        open("TrackedBests.txt", "w").close() #clear the file
    #Start a clock
    Start_time = time.time()
    
    if Algorithm =="L-BFGS-B":
        Result = minimize(objective, 
                        x0, 
                        method="L-BFGS-B", 
                        args = (NOPs,),
                        bounds=bounds,
                        options={
                            #"maxiter": NOPs.maxiterPow,
                            "maxfev": NOPs.maxfevPow,
                            "ftol": 1e-5,
                            "disp": True}
                        )
    elif Algorithm == "DE":
        Result = differential_evolution(objective, 
                                        bounds = bounds,
                                        args = (NOPs,),
                                        strategy='best1bin',
                                        maxiter= NOPs.maxfevDE//(NOPs.popsizeDE*len(bounds) - 1),
                                        tol = 0.1,
                                        recombination = 0.7,
                                        polish = False, 
                                        popsize = 40)
    elif Algorithm == "Both":
        Search = differential_evolution(objective, 
                                        bounds = bounds,
                                        args = (NOPs,),
                                        strategy='best1bin',
                                        maxiter= NOPs.maxfevDE//(NOPs.popsizeDE*len(bounds) - 1),
                                        tol = 0.1,
                                        recombination = 0.7,
                                        polish = False, 
                                        popsize = 40,
                                        disp= True
                                        )
        
         
        Result = minimize(objective, 
                        best_x, 
                        method="Powell", 
                        args = (NOPs,),
                        bounds=bounds,
                        #callback = PointlessCB,
                        options={
                            #"maxiter": NOPs.maxiterPow,
                            "maxfev": NOPs.maxfevPow,
                            "ftol": 1e-5,
                            "disp": True}
                        )
    else:
        print("Invalid algorithm choice. Please choose DE or Powell")
    
    Elapsed_time_min = (time.time() - Start_time)//60
    Elapsed_hours = Elapsed_time_min//60
    Elapsed_time_min_remainder = Elapsed_time_min%60
    
    print("Final Result:")
    print(Result)
    print('Best result')
    print(best_score)
    print('best params')
    print(best_x)
    print("Optimisation time:", Elapsed_hours, "h ", Elapsed_time_min_remainder, "min")
    # Save all attributes to a text file
    with open("optimization_result.txt", "w") as f:
        for attr in dir(Result):
            # skip built-ins
            if not attr.startswith("_"):
                value = getattr(Result, attr)
                f.write(f"{attr} : {value}\n")
        f.write(f"Best tracked score: {best_score}\n")
        f.write(f"Best tracked params: {best_x}")
    
    ### Show it off
    try:
        #FinishOut(Result.x)
        FinishOut(best_x) #for debugging
    except Exception as e:
        print('This is awkward. The best solution isnt working. try fixing it')
        traceback.print_exc()
       
    return 0 

def ScaleTowardsOnes(Params):
    return Params*NOPs.Scale

def ScaleBackToFull(Params):
    return Params/NOPs.Scale

if __name__ == "__main__":
    main()



#add a callback function print so that I know if it's powelling or if its differentially evolving
