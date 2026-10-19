import felupe as fe



class NonOptimParams:
    def __init__(self):
        self.ConeWidth =None
        self.ConeHeight = None
        self.ConeCornerRadius = None
        self.ConeOffset = None  #'Distance the cone protrudes outward from enclosure
        self.ConeEnclosureGap = None
        self.MountingGap = None
        self.Node_find_tol = None
        self.cadfile_path = None
        self.stepout_path = None
        self.Xmax = None
        self.TargetStiffness = None
        self.OptimizationWeights = None
        self.MaterialCoefficients = None
        self.MeshFine = None
        self.MeshCoarse = None 
        self.N_Steps = None
        self.MountFlangeThickness = None
        self.maxfevPow = None
        self.maxiterPow = None
        self.TriggerPath = None
        self.maxfevDE = None
        self.popsizeDE = None
        self.maxiterDE = None
        self.K_clamp = None
        self.IncludeSpider = None
        self.Scale = None
        self.CachingFile = None
        self.EndStiffnessMultiplier = None  # target end-region stiffness = mid-region avg * this
        self.EndRegionFraction = None  # fraction of the stroke (each side) treated as the "end" region
        self.ExportComponentName = None  # if set, Mutate.py exports only this Fusion component instead of the whole design
        self.SolveTimeoutSeconds = None  # max seconds for the forward+backward solve before killing workers and skipping the iteration (default 120 if None)
        self.ConeFlangeWidth = None  # radial extent (beyond ConeCornerRadius) of the small cone-side flange
        self.DebugShowBCs = None  # TEMPORARY: True saves BCDebug/debug_bc_dir<1|-1>.png every iteration showing the BC node selection
        self.GroundRadialBuffer = None  # extra inward slack (mm) on the enclosure mount's radial threshold, on top of Node_find_tol (default 0.5 if None)
    pass

