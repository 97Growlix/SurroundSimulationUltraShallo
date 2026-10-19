import numpy as np
import meshio
import felupe as fe
import gmsh
import pyvista as pv
import matplotlib.pyplot as plt
from felupe.constitution.tensortrax.models.hyperelastic import mooney_rivlin
from SurroundClasses import *
from multiprocessing import Pool, TimeoutError as MPTimeoutError
from pypardiso import spsolve
from datetime import datetime
import os
import time
import warnings


def safe_unlink(path, retries=10, delay=0.5):
    for _ in range(retries):
        try:
            os.unlink(path)
            return
        except PermissionError:
            time.sleep(delay)
    raise

def AnalyzeItBothWays(NOPs):
    
    args = [(NOPs, 1),(NOPs,-1)] #represents forward and backward sweep

    with Pool(processes=2) as pool:
        OneWayResults = pool.starmap(AnalyzeItOneWay,args)    
    
    KmsOut, DispOut = OneWayResults[0]

    KmsIn, DispIn = OneWayResults[1]
    
    KmsOut = KmsOut[1:] #remove the first element bc it gets repeated
    DispOut = DispOut[1:] 

    #add both sides of stroke together
    KmsTot = np.concatenate([KmsIn[::-1], KmsOut])
    DispTot = np.concatenate([DispIn[::-1], DispOut])

    return KmsTot, DispTot

def LocalRF(job, mask):  #finds the rf for masked nodes at the last step of the job
        Final_step_forces = job.steps[-1].items[0].results.force
        force_arr = Final_step_forces.toarray().flatten()
        num_nodes = force_arr.size // 3
        force_reshaped = force_arr.reshape((num_nodes, 3))

        masked_rfs = force_reshaped[mask, 2]
        rf = sum(masked_rfs)

        return rf

def LocalRFAxisymmetric(job, mask):  # finds the rf for masked nodes at the last step of the job
        Final_step_forces = job.steps[-1].items[0].results.force
        force_arr = Final_step_forces.toarray().flatten()
        num_nodes = force_arr.size // 2  # axisymmetric field has 2 components (axial, radial)
        force_reshaped = force_arr.reshape((num_nodes, 2))

        masked_rfs = force_reshaped[mask, 0]  # component 0 = axial
        rf = sum(masked_rfs)

        return rf

def CreateFeField(NOPs):

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    
    gmsh.open(NOPs.stepout_path)

    #get min z val
    min_z = float("inf")
    for tag in gmsh.model.getEntities(3):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, tag[1])
        if zmin < min_z:
            min_z = zmin

    print("zmin", min_z)

    Refinement_pt_a = [NOPs.ConeWidth/2 - NOPs.ConeCornerRadius, NOPs.ConeHeight/2- NOPs.ConeCornerRadius, NOPs.ConeOffset]
    print("ref pt", Refinement_pt_a)
    #Refinement_pt_b = [NOPs.ConeWidth, NOPs.ConeHeight - NOPs.ConeCornerRadius, NOPs.ConeOffset]

    dist_field = gmsh.model.mesh.field.add("Distance")

    pt_tag = gmsh.model.occ.addPoint(*Refinement_pt_a, 0)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.field.setNumbers(dist_field, "NodesList", [pt_tag])

    th_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(th_field, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(th_field, "SizeMin", NOPs.MeshFine)
    gmsh.model.mesh.field.setNumber(th_field, "SizeMax", NOPs.MeshCoarse)
    gmsh.model.mesh.field.setNumber(th_field, "DistMin", NOPs.ConeCornerRadius*0.9)
    gmsh.model.mesh.field.setNumber(th_field, "DistMax", NOPs.ConeCornerRadius*1.2)

    tip_field = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(tip_field, "VIn", NOPs.MeshFine)
    gmsh.model.mesh.field.setNumber(tip_field, "VOut", NOPs.MeshCoarse)
    gmsh.model.mesh.field.setNumber(tip_field, "XMin", xmin-1)
    gmsh.model.mesh.field.setNumber(tip_field, "XMax", xmax+1)
    gmsh.model.mesh.field.setNumber(tip_field, "YMin", ymin-1)
    gmsh.model.mesh.field.setNumber(tip_field, "YMax", ymax+1)
    gmsh.model.mesh.field.setNumber(tip_field, "ZMin", zmin-1)
    gmsh.model.mesh.field.setNumber(tip_field, "ZMax", min_z+15)
    gmsh.model.mesh.field.setNumber(tip_field, "Thickness", 15)

    #combine the two mesh mmodifier fields
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [th_field, tip_field])
    # gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0) 
        
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    print(gmsh.__version__)
    gmsh.model.mesh.generate(3)

    print("MeshFine:", NOPs.MeshFine)
    print("MeshCoarse:", NOPs.MeshCoarse)
    print("ZMin box:", zmin-1, "ZMax box:", zmin+15)
    print("Full Z range:", zmin, "to", zmax)
    print("Refinement pt:", Refinement_pt_a)
    print("tip_field id:", tip_field)
    print("min_field id:", min_field)

    for tag in gmsh.model.getEntities(3):
        print(gmsh.model.getBoundingBox(3, tag[1]))

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # version 2.2
    gmsh.option.setNumber("Mesh.Binary", 0)            # ASCII, not binary

    gmsh.write("surround.msh")

    #gmsh.fltk.run()
    #gmsh.finalize()

    # Read the Gmsh mesh
    msh_file = "surround.msh"
    try:
        msh = meshio.read(msh_file)
    except Exception as e:
        print("Mesh read failed:", e)
        return 1e9

    # Extract points
    points = msh.points

    # Extract tetrahedral cells
    tetra_cells = None
    for cell_block in msh.cells:
        if cell_block.type == "tetra":
            tetra_cells = cell_block.data
            break

    if tetra_cells is None:
        raise ValueError("No tetrahedral mesh found in the file")

    # Create the FElupe mesh
    tetra_mesh = fe.Mesh(points, tetra_cells, cell_type = "tetra")

    # Assign quadrature for tetrahedra
    quad = fe.quadrature.Tetrahedron(order=2)
    tetra_mesh.quadrature = quad
    elem = fe.element.Tetra()
    # Create a region over all tetra cells
    region = fe.Region(tetra_mesh, elem, quadrature=quad)

    # Create a field
    u = fe.Field(region, dim=3)
    field = fe.FieldContainer([u])

    return field.copy(), tetra_mesh, tetra_cells, points

def CreateFeFieldAxisymmetric(NOPs):
    """Mesh a 2D cross-section STEP (a single planar face, not a solid) and build
    a FElupe axisymmetric field from it.

    Assumes NOPs.stepout_path points to a STEP file containing exactly one planar
    surface: the cross-section profile of the surround. The revolve axis is
    always the global Y axis, so the profile plane contains Y and exactly one
    of X/Z (whichever the sketch was drawn on); the other is ~0 across the
    whole face and is dropped.

    FElupe's FieldAxisymmetric expects mesh column 0 = axial (rotation-axis)
    coordinate and column 1 = radial coordinate (must be >= 0), so points are
    remapped to [Y, radial], mirroring the radial coordinate if the profile was
    sketched on the negative side of the axis. Override with
    NOPs.AxisymAxes = (axial_col, radial_col) as indices into (x, y, z) if a
    given profile doesn't fit this convention.
    """

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)

    gmsh.open(NOPs.stepout_path)

    surfaces = gmsh.model.getEntities(2)
    if len(surfaces) == 0:
        gmsh.finalize()
        raise ValueError(f"No surfaces found in {NOPs.stepout_path}")
    if len(surfaces) > 1:
        print(f"Warning: {len(surfaces)} surfaces found, expected 1. "
              "Using the one with the largest bounding-box diagonal.")
        def bbox_diag(tag):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
            return (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
        surfaces = [max(surfaces, key=lambda s: bbox_diag(s[1]))]

    gmsh.option.setNumber("Mesh.MeshSizeMin", NOPs.MeshFine)
    gmsh.option.setNumber("Mesh.MeshSizeMax", NOPs.MeshCoarse)

    gmsh.model.mesh.generate(2)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)

    # unique per-process filename - AnalyzeItBothWaysAxisymmetric runs two of
    # these in parallel via multiprocessing.Pool, and a shared filename causes
    # one process's gmsh.write() to race with the other's meshio.read()
    msh_file = f"surround_axisym_{os.getpid()}.msh"
    gmsh.write(msh_file)
    gmsh.finalize()

    try:
        msh = meshio.read(msh_file)
    except Exception as e:
        raise RuntimeError(f"Failed to read generated mesh file {msh_file}: {e}") from e
    finally:
        try:
            os.remove(msh_file)
        except OSError:
            pass

    points_3d = msh.points

    tri_cells = None
    for cell_block in msh.cells:
        if cell_block.type == "triangle":
            tri_cells = cell_block.data
            break

    if tri_cells is None:
        raise ValueError("No triangular mesh found in the file")

    # Revolve axis is always the global Y axis, so axial = col 1. The profile
    # plane contains that axis, so exactly one of x/z is ~0 across the whole
    # face (dropped) and the other is the (signed) radial coordinate.
    tol = NOPs.Node_find_tol
    x_span = points_3d[:, 0].max() - points_3d[:, 0].min()
    z_span = points_3d[:, 2].max() - points_3d[:, 2].min()

    if hasattr(NOPs, "AxisymAxes") and NOPs.AxisymAxes is not None:
        axial_col, radial_col = NOPs.AxisymAxes
    else:
        axial_col = 1
        radial_col = 0 if x_span > z_span else 2
        dropped_col = 2 if radial_col == 0 else 0
        if points_3d[:, dropped_col].max() - points_3d[:, dropped_col].min() > tol:
            raise ValueError(
                f"Expected x or z to be constant across the profile face "
                f"(x_span={x_span}, z_span={z_span}). Set NOPs.AxisymAxes "
                "explicitly if the profile isn't in a plane containing the "
                "Y axis."
            )
        print(f"Detected axisymmetric axes: axial=Y(col1), "
              f"radial={'X' if radial_col == 0 else 'Z'}(col{radial_col})")

    radial = points_3d[:, radial_col]
    if radial.min() < -tol:
        # profile was sketched on the negative side of the axis; mirror it
        radial = -radial

    points_2d = np.column_stack([points_3d[:, axial_col], radial])

    tri_mesh = fe.Mesh(points_2d, tri_cells, cell_type="triangle")

    quad = fe.quadrature.Triangle(order=2)
    tri_mesh.quadrature = quad
    elem = fe.element.Triangle()
    # gmsh's 2D triangle winding doesn't always match FElupe's expected
    # orientation, so this first Region build routinely has negative-volume
    # cells - expected, and fixed right below via mesh.flip(). Suppress just
    # FElupe's warning for this one call so it doesn't spam every iteration.
    import numpy as _np
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Negative volumes for cells")
        region = fe.Region(tri_mesh, elem, quadrature=quad)

    negative = _np.any(region.dV < 0, axis=0)
    if _np.any(negative):
        tri_mesh = tri_mesh.flip(negative)
        region = fe.Region(tri_mesh, elem, quadrature=quad)

    u = fe.FieldAxisymmetric(region, dim=2)
    field = fe.FieldContainer([u])

    return field.copy(), tri_mesh, tri_cells, points_2d

def CreateBCsAxisymmetric(NOPs, field, points):
    """Boundary conditions for the axisymmetric cross-section.

    points columns are [axial, radial] (see CreateFeFieldAxisymmetric).

    Grounded boundary (enclosure/basket mount): a flat-riser-flat lip at the
    outer edge. Rather than assuming a fixed riser height (NOPs.MountFlangeThickness
    used to serve this, but the outer mount's thickness is now itself an
    optimized CAD parameter that varies every iteration, making a fixed
    offset stale), the riser is found directly from the mesh: the true outer
    edge of the whole cross-section is radial.max() (always well-defined for
    a revolved part, regardless of current parameter values), and the nodes
    sitting at that radius trace out the riser's actual axial span. The
    mounting flat is the bottom of that span (established convention: the
    "-axial direction" flat is what actually contacts the enclosure, the top
    is just the exposed face) - at radius > NOPs.ConeCornerRadius +
    NOPs.ConeEnclosureGap. Fully fixed.

    Moving boundary (cone/voice-coil-former attachment): now two features,
    unioned together -
      1. A vertical wall at the cone's diameter, radius == NOPs.ConeCornerRadius,
         capped at axial <= Enclosure_Axial_Height - anything above that plane
         (e.g. a relief/chamfer at the very top of the wall) isn't actually in
         contact with the cone.
      2. A small flange radially inward of that wall (radius down to
         NOPs.ConeCornerRadius - NOPs.ConeFlangeWidth), lying on that same
         plane perpendicular to the axial axis as the enclosure mounting
         points (axial == Enclosure_Axial_Height exactly).
    Gets a prescribed axial ramp and is fixed radially (pistonic motion, no
    radial spreading) - same physics as before, just a different node
    selection to match the new geometry.

    NOTE: NOPs.ConeOffset and NOPs.MountFlangeThickness are no longer used
    here - both were fixed-value assumptions about the enclosure mount's
    geometry that broke once that geometry started being driven by optimized
    CAD parameters instead. The riser is now found directly from the mesh
    instead (see above), so this should keep working regardless of how the
    outer mount's thickness/parameterization changes in the future - as long
    as it's still a flat-riser-flat lip at the true outer radius.
    """
    axial = points[:, 0]
    radial = points[:, 1]
    tol = NOPs.Node_find_tol

    radial_max = radial.max()
    Riser_Mask = np.abs(radial - radial_max) <= tol
    if not np.any(Riser_Mask):
        raise ValueError(
            f"No nodes found at the outer edge (radial.max()={radial_max:.3f}) "
            "- can't locate the enclosure mounting riser."
        )
    Enclosure_Axial_Height = axial[Riser_Mask].min()

    # extra inward slack on top of tol - the innermost node(s) of the mounting
    # flat can sit a bit farther inward than the exact threshold, and tol
    # alone isn't always enough margin to catch them
    ground_buffer = NOPs.GroundRadialBuffer if NOPs.GroundRadialBuffer is not None else 0.5

    Ground_Mask = (np.abs(axial - Enclosure_Axial_Height) <= tol) & \
        (radial >= NOPs.ConeCornerRadius + NOPs.ConeEnclosureGap - tol - ground_buffer)

    # cap the wall at the enclosure-mount plane - anything above that (e.g. a
    # relief/chamfer at the very top of the wall) isn't in contact with the cone
    Wall_Mask = (np.abs(radial - NOPs.ConeCornerRadius) <= tol) & \
        (axial <= Enclosure_Axial_Height + tol)

    Flange_Mask = (np.abs(axial - Enclosure_Axial_Height) <= tol) & \
        (radial < NOPs.ConeCornerRadius - tol) & \
        (radial >= NOPs.ConeCornerRadius - NOPs.ConeFlangeWidth - tol)

    Move_Mask = Wall_Mask | Flange_Mask

    # Fail fast and loud if a mask comes up empty. An empty Ground_Mask means
    # the FE solve would run with the enclosure entirely unconstrained
    # (rigid-body motion, no real ground support) and can still silently
    # produce a plausible-looking-but-meaningless "converged" score instead of
    # erroring - catching it here instead lets objective()'s existing
    # exception handling skip the iteration cleanly, same as a mesh/solve
    # failure, rather than only surfacing later as a confusing pyvista crash.
    if not np.any(Ground_Mask):
        raise ValueError(
            f"Ground_Mask matched zero nodes (axial range [{axial.min():.3f}, "
            f"{axial.max():.3f}], radial range [{radial.min():.3f}, "
            f"{radial.max():.3f}], looked for axial~={Enclosure_Axial_Height:.3f}, "
            f"radial>={NOPs.ConeCornerRadius + NOPs.ConeEnclosureGap - tol:.3f}) "
            "- this geometry doesn't have the expected enclosure mounting flat."
        )
    if not np.any(Move_Mask):
        raise ValueError(
            f"Move_Mask (wall+flange) matched zero nodes (axial range "
            f"[{axial.min():.3f}, {axial.max():.3f}], radial range "
            f"[{radial.min():.3f}, {radial.max():.3f}], looked for "
            f"radial~={NOPs.ConeCornerRadius:.3f}) - this geometry doesn't "
            "have the expected cone wall/flange."
        )

    ###
    #boundary cond
    ###

    # Fully fixed support at the enclosure mount
    bc_ground = fe.Boundary(
        field[0],
        mask=Ground_Mask,
        skip=(False, False)  # constrain both axial and radial
    )

    # Prescribed axial displacement at the cone attachment (ramped later)
    move_z = fe.Boundary(
        field[0],
        mask=Move_Mask,
        skip=(False, True)  # constrain axial (component 0), skip radial
    )

    # Cone attachment can't spread radially - pistonic motion only
    bc_cone = fe.Boundary(
        field[0],
        mask=Move_Mask,
        skip=(True, False)  # constrain radial (component 1), skip axial
    )

    bcs = dict()
    bcs["ground"] = bc_ground
    bcs["move_z"] = move_z
    bcs["bc_cone"] = bc_cone

    masks = dict()
    masks["Move"] = Move_Mask
    masks["Ground"] = Ground_Mask
    masks["Wall"] = Wall_Mask  # subset of Move - the vertical cone wall only
    masks["Flange"] = Flange_Mask  # subset of Move - the small flange only

    return bcs, masks

def ShowBCsAxisymmetric(NOPs):
    """Mesh the cross-section, build the axisymmetric BCs, and visualize the
    mesh with the moving (orange) and grounded (blue) node groups highlighted.
    """
    field, tri_mesh, tri_cells, points = CreateFeFieldAxisymmetric(NOPs)
    bcs, masks = CreateBCsAxisymmetric(NOPs, field, points)

    print("Number of points in mesh:", points.shape[0])
    print("Number of moving nodes:", np.sum(masks["Move"]))
    print("Number of grounded nodes:", np.sum(masks["Ground"]))

    # pad to 3D (z=0) for pyvista
    points_3d = np.column_stack([points[:, 0], points[:, 1], np.zeros(points.shape[0])])

    cells_pv = np.hstack([np.full((tri_cells.shape[0], 1), 3), tri_cells]).flatten()
    grid = pv.UnstructuredGrid(cells_pv, np.full(tri_cells.shape[0], 5), points_3d)  # 5 = VTK_TRIANGLE

    pv_plotter = pv.Plotter()
    pv_plotter.add_mesh(grid, show_edges=True)

    add_bc_nodes(pv_plotter, bcs["ground"].points, 'blue', grid, points_3d)
    add_bc_nodes(pv_plotter, bcs["move_z"].points, 'orange', grid, points_3d)

    pv_plotter.view_xy()
    pv_plotter.show()

    return field, bcs, masks

def DebugSnapshotBCs(NOPs, tri_mesh, bcs, tag=""):
    """TEMPORARY debug aid: off-screen render of the mesh + BC node selection
    to a PNG, overwritten every call (BCDebug/debug_bc_<tag>.png, absolute
    path depends on the process's working directory when it runs - BCDebug/
    next to wherever OptimusMTX.py is invoked from) - lets you eyeball what
    CreateBCsAxisymmetric picked on every optimizer iteration without
    blocking (no interactive window, safe to call from the parallel workers).

    Toggle with NOPs.DebugShowBCs = True/False. Once you've verified the BCs
    look right, set it back to False (or just delete this function and its
    call site in AnalyzeItOneWayAxisymmetric) to remove the per-iteration
    rendering overhead.
    """
    if not getattr(NOPs, "DebugShowBCs", False):
        return

    try:
        grid = tri_mesh.as_unstructured_grid()  # handles the 2D->3D padding
        points_3d = grid.points

        pv_plotter = pv.Plotter(off_screen=True)
        pv_plotter.add_mesh(grid, show_edges=True, color='lightgray')

        add_bc_nodes(pv_plotter, bcs["ground"].points, 'blue', grid, points_3d)
        add_bc_nodes(pv_plotter, bcs["move_z"].points, 'orange', grid, points_3d)

        pv_plotter.view_xy()
        os.makedirs("BCDebug", exist_ok=True)
        pv_plotter.screenshot(f"BCDebug/debug_bc_{tag}.png")
        pv_plotter.close()
    except Exception as e:
        print(f"DebugSnapshotBCs failed (non-fatal, BC debug view skipped): {e}")

def AnalyzeItOneWayAxisymmetric(NOPs, Direction, fe_mesh_field=None):
    """Axisymmetric equivalent of AnalyzeItOneWay. Sweeps the cone-attachment
    boundary axially by NOPs.Xmax*Direction and returns the stiffness curve.

    Unlike the 3D quarter-symmetry model, the axisymmetric formulation already
    integrates the full 360 degrees (the 2*pi*R factor is baked into FElupe's
    axisymmetric integral form), so reaction forces need no symmetry multiplier.
    """

    C10 = NOPs.MaterialCoefficients[0]
    C01 = NOPs.MaterialCoefficients[1]

    if fe_mesh_field is None:
        field, tri_mesh, tri_cells, points = CreateFeFieldAxisymmetric(NOPs)
    else:
        field = fe_mesh_field
        tri_mesh = field[0].region.mesh
        points = tri_mesh.points

    bcs, masks = CreateBCsAxisymmetric(NOPs, field, points)

    DebugSnapshotBCs(NOPs, tri_mesh, bcs, tag=f"dir{Direction}")

    material = fe.Hyperelastic(mooney_rivlin, C10=C10, C01=C01)

    solid = fe.SolidBody(material, field)

    steps = fe.math.linsteps([0, NOPs.Xmax * Direction], num=round(NOPs.N_Steps / 2))

    step = fe.Step(
        items=[solid],
        ramp={bcs["move_z"]: steps},
        boundaries=bcs
    )

    job = fe.CharacteristicCurve(steps=[step], boundary=bcs["move_z"])

    job.evaluate(solver=spsolve)

    # component 0 = axial for FieldAxisymmetric
    disp = np.array([d[0] for d in job.x])
    force = np.array([f[0] for f in job.y])

    kms = np.gradient(force, disp)

    return kms, disp

def AnalyzeItBothWaysAxisymmetric(NOPs):
    """Runs the forward and backward sweeps in parallel. Bounded by
    NOPs.SolveTimeoutSeconds (default 120s) so a hung/stuck worker (bad mesh,
    solver stall, etc.) can't block the optimizer forever - on timeout the
    pool is terminated (killing both workers) and a TimeoutError is raised,
    which callers like OptimusMTX's objective() already catch to skip the
    iteration and move on.
    """

    args = [(NOPs, 1), (NOPs, -1)]  # represents forward and backward sweep
    timeout = NOPs.SolveTimeoutSeconds if getattr(NOPs, "SolveTimeoutSeconds", None) else 120

    with Pool(processes=2) as pool:
        async_result = pool.starmap_async(AnalyzeItOneWayAxisymmetric, args)
        try:
            OneWayResults = async_result.get(timeout=timeout)
        except MPTimeoutError:
            # letting this propagate out of the `with` block makes Pool.__exit__
            # call pool.terminate(), which kills the hung worker processes
            raise TimeoutError(
                f"Axisymmetric solve timed out after {timeout}s "
                "(forward/backward sweep) - workers killed"
            )

    KmsOut, DispOut = OneWayResults[0]

    KmsIn, DispIn = OneWayResults[1]

    KmsOut = KmsOut[1:]  # remove the first element bc it gets repeated
    DispOut = DispOut[1:]

    KmsTot = np.concatenate([KmsIn[::-1], KmsOut])
    DispTot = np.concatenate([DispIn[::-1], DispOut])

    return KmsTot, DispTot

def CombineDualSurroundKms(Kms, Disp, tol=1e-6):
    """This driver uses two surrounds mirrored about the zx-plane. When the
    shared voice-coil-former displaces by x, one surround sees local
    displacement x (its own outstroke/instroke response, i.e. Kms(x) as
    already computed) while the mirrored surround sees local displacement -x
    (since its own "outward" direction is flipped). Their stiffnesses add in
    parallel, so the physical dual-surround stiffness at x is Kms(x) + Kms(-x).

    Requires Disp to be a symmetric grid (Disp[::-1] == -Disp), which is what
    AnalyzeItBothWaysAxisymmetric produces, so Kms(-x_i) is just Kms reversed.
    """
    if not np.allclose(Disp[::-1], -Disp, atol=tol):
        raise ValueError(
            "Disp grid isn't symmetric about 0; can't look up Kms(-x) by "
            "reversal. Re-run AnalyzeItBothWaysAxisymmetric with matching "
            "step counts/extents in both directions."
        )

    KmsDual = Kms + Kms[::-1]

    return KmsDual, Disp

def AnalyzeItBothWaysDualSurround(NOPs):
    """Full pipeline for this driver's mirrored dual-surround stiffness:
    solve the single-surround response both ways, then superpose it with its
    own mirror image to get the actual physical stiffness curve.
    """
    KmsSingle, Disp = AnalyzeItBothWaysAxisymmetric(NOPs)
    KmsDual, Disp = CombineDualSurroundKms(KmsSingle, Disp)

    return KmsDual, Disp

def ScoreKmsAxisymmetric(Kms, Disp, NOPs):
    """Scoring for the axisymmetric dual-surround stiffness curve.

    The outer NOPs.EndRegionFraction (default 10%) of the stroke on each side
    is the "ramp" - it's excluded from the flatness calc and scored on its own
    as RampUp: it's required to be much stiffer than the flat middle region
    (mechanical end-stop behavior) via NOPs.EndStiffnessMultiplier (default 5x
    the middle-region average). Only a shortfall below that target is
    penalized - stiffer than target at the ramps is fine.

    Volume and asymmetry are not scored here (the dual-surround curve is
    symmetric about x=0 by construction, so an asymmetry term is redundant).

    KmsFlatness and RampUp are both normalized by Mid_Avg^2 (the squared
    coefficient of variation / relative shortfall) rather than left as raw
    (N/mm)^2 quantities - unnormalized, both scale with the square of the
    surround's own stiffness magnitude, which rewards the optimizer for
    simply making the surround softer overall rather than actually flatter or
    better end-stiffened. Normalizing makes them dimensionless and scale-
    invariant, leaving Shift as the only term pulling toward a specific
    absolute stiffness.

    NOPs.OptimizationWeights must be a dict keyed by score name, giving a
    weight for each of: "KmsFlatness" (between the ramps only), "RampUp"
    (end-stiffness shortfall), "Shift" (deviation from TargetStiffness). e.g.:
        NOPs.OptimizationWeights = {"KmsFlatness": 5e3, "RampUp": 1e5, "Shift": 5e-1}
    NOTE: since KmsFlatness/RampUp are now dimensionless fractions (typically
    << 1) instead of raw (N/mm)^2 values, existing weights tuned against the
    old unnormalized scores will need to be scaled up significantly.
    """
    n = len(Kms)

    end_frac = NOPs.EndRegionFraction if NOPs.EndRegionFraction is not None else 0.10
    end_n = max(1, int(round(n * end_frac)))

    if NOPs.IncludeSpider:
        KmsSpider = np.array([CalcKSpider(u) for u in Disp])
        Kms = Kms + KmsSpider

    Kms_BetweenRamps = Kms[end_n: n - end_n]
    Kms_Ramps = np.concatenate([Kms[:end_n], Kms[n - end_n:]])

    # normalize by Mid_Avg^2 (squared coefficient of variation) - raw variance
    # scales with the square of the signal's own magnitude, so an unnormalized
    # score rewards simply making the surround less stiff overall, regardless
    # of how proportionally flat/steep it actually is. A tiny epsilon guards
    # against a stray near-zero-stiffness candidate blowing this up to inf.
    Mid_Avg = np.mean(Kms_BetweenRamps)
    Mid_Avg_sq = Mid_Avg ** 2 + 1e-12

    Mid_Dev = Kms_BetweenRamps - Mid_Avg
    KmsFlatness = (np.sum(Mid_Dev ** 2) / len(Kms_BetweenRamps)) / Mid_Avg_sq

    RampAvg = np.mean(Kms_Ramps)
    EndStiffnessMultiplier = NOPs.EndStiffnessMultiplier if NOPs.EndStiffnessMultiplier is not None else 5.0
    Target_RampAvg = Mid_Avg * EndStiffnessMultiplier
    Ramp_Shortfall = max(0.0, Target_RampAvg - RampAvg)
    RampUp = (Ramp_Shortfall ** 2) / Mid_Avg_sq

    Shift = (Mid_Avg - NOPs.TargetStiffness) ** 2

    ScoreNames = ["KmsFlatness", "RampUp", "Shift"]
    Scores = np.array([KmsFlatness, RampUp, Shift], dtype=float)

    PureWeights = np.array([NOPs.OptimizationWeights[name] for name in ScoreNames], dtype=float)

    WeightedScores = PureWeights * Scores
    SurroundScore = np.sum(WeightedScores)

    return SurroundScore, WeightedScores

def ShowItOffAxisymmetric(NOPs, params):
    """Axisymmetric equivalent of ShowItOff: solves both directions, animates
    the deformed cross-section, scores the dual-surround stiffness curve, and
    plots force/stiffness.

    NOTE: the rectangular ShowItOff's "clamp" boundary (contact against the
    enclosure on the forward stroke) and its curve-vs-straight reaction-force
    breakdown are both rectangular-geometry-specific and are not implemented
    here - there's no clamp mask yet for the axisymmetric case, and there's no
    curve/straight distinction on a body of revolution.
    """
    import os
    import subprocess
    import imageio

    field, tri_mesh, tri_cells, points = CreateFeFieldAxisymmetric(NOPs)

    bcs, masks = CreateBCsAxisymmetric(NOPs, field, points)

    print("Number of points in mesh:", points.shape[0])

    cells_pv = np.hstack([np.full((tri_cells.shape[0], 1), 3), tri_cells]).flatten()
    points_3d = np.column_stack([points[:, 0], points[:, 1], np.zeros(points.shape[0])])

    grid = pv.UnstructuredGrid(cells_pv, np.full(tri_cells.shape[0], 5), points_3d)  # 5 = VTK_TRIANGLE

    pv_plotter = pv.Plotter()
    pv_plotter.add_mesh(grid, show_edges=True)

    add_bc_nodes(pv_plotter, bcs["ground"].points, 'blue', grid, points_3d)
    add_bc_nodes(pv_plotter, bcs["move_z"].points, 'orange', grid, points_3d)

    pv_plotter.view_xy()

    #pv_plotter.show()

    ###
    #material data
    ###

    material = fe.Hyperelastic(mooney_rivlin, C10=NOPs.MaterialCoefficients[0], C01=NOPs.MaterialCoefficients[1])

    solid_out = fe.SolidBody(material, field)

    bcs_out = bcs

    steps_out = fe.math.linsteps([0, NOPs.Xmax], num=round(NOPs.N_Steps / 2))

    step_out = fe.Step(
        items=[solid_out],
        ramp={bcs_out["move_z"]: steps_out},
        boundaries=bcs_out
    )

    cell_types = np.full(tri_cells.shape[0], 5)  # 5 = VTK_TRIANGLE

    original_pts = points_3d.copy()

    deformed_meshes = []

    def cbout(stepnum, substepnum, substep, **kwargs):
        disp_values = substep.x[0].values  # (npoints, 2): [axial, radial]
        disp_3d = np.column_stack([disp_values[:, 0], disp_values[:, 1], np.zeros(disp_values.shape[0])])
        pts_def = original_pts + disp_3d
        mesh_pv = pv.UnstructuredGrid(cells_pv, cell_types, pts_def)
        deformed_meshes.append(mesh_pv)

    def cbin(stepnum, substepnum, substep, **kwargs):
        disp_values = substep.x[0].values
        disp_3d = np.column_stack([disp_values[:, 0], disp_values[:, 1], np.zeros(disp_values.shape[0])])
        pts_def = original_pts + disp_3d
        mesh_pv = pv.UnstructuredGrid(cells_pv, cell_types, pts_def)
        deformed_meshes.insert(0, mesh_pv)

    ### Create the CharacteristicCurve job
    job_out = fe.CharacteristicCurve(steps=[step_out], boundary=bcs_out["move_z"], callback=cbout)
    job_out.evaluate(solver=spsolve)

    #Now do the other way, but recompute the starting points first
    field, tri_mesh, tri_cells, points = CreateFeFieldAxisymmetric(NOPs)

    bcs, masks = CreateBCsAxisymmetric(NOPs, field, points)
    solid_in = fe.SolidBody(material, field)
    bcs_in = bcs
    steps_in = fe.math.linsteps([0, -NOPs.Xmax], num=round(NOPs.N_Steps / 2))
    step_in = fe.Step(
        items=[solid_in],
        ramp={bcs_in["move_z"]: steps_in},
        boundaries=bcs_in
    )
    job_in = fe.CharacteristicCurve(steps=[step_in], boundary=bcs_in["move_z"], callback=cbin)
    job_in.evaluate(solver=spsolve)

    ### For animation
    TimeStamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    gif_path2 = f"SurroundGifs/solution_axisym_{TimeStamp}.gif"
    gif_path = "surround_mesh_animation_axisym.gif"

    pl = pv.Plotter(window_size=(900, 700))
    pl.add_mesh(deformed_meshes[0], show_edges=True)
    pl.add_text("Cross-section displacement animation", font_size=10)
    pl.view_xy()
    pl.show(auto_close=False, interactive=False)

    frames = []
    for t in range(len(deformed_meshes)):
        pl.mesh.points[:] = deformed_meshes[t].points
        pl.render()
        frames.append(pl.screenshot(return_img=True))

    pl.close()
    imageio.mimsave(gif_path, frames + frames[::-1], fps=round(NOPs.N_Steps / 2), loop=0)
    imageio.mimsave(gif_path2, frames + frames[::-1], fps=round(NOPs.N_Steps / 2), loop=0)
    print("GIF saved:", gif_path)

    try:
        os.startfile(gif_path)  # windows
    except AttributeError:
        subprocess.call(["open", gif_path])  # mac or Linux

    # Reaction force at max stroke, single surround (sanity readout - no
    # curve/straight split since the geometry is uniform around the axis)
    MoveRf = LocalRFAxisymmetric(job_out, masks["Move"])
    print('Reaction force in Newtons at the cone attachment (single surround, full 360deg):', MoveRf)

    disp_out = np.array([d[0] for d in job_out.x])
    force_out = np.array([f[0] for f in job_out.y])
    disp_in = np.array([d[0] for d in job_in.x])
    force_in = np.array([f[0] for f in job_in.y])

    disp_in = disp_in[1:]
    force_in = force_in[1:]

    force = np.concatenate([force_in[::-1], force_out])
    disp = np.concatenate([disp_in[::-1], disp_out])

    # single-surround stiffness, then the actual physical (mirrored dual-
    # surround) stiffness curve used for scoring
    kms = np.gradient(force, disp)
    kmsDual, dispDual = CombineDualSurroundKms(kms, disp)

    SurroundScore, WeightedScores = ScoreKmsAxisymmetric(kmsDual, dispDual, NOPs)

    print(f'Score = {SurroundScore}')
    print(f'weighted scores = {WeightedScores}')

    with open(NOPs.CachingFile, 'a') as f:
                f.write(f"scores: {WeightedScores}\n")
                f.write(f"score: {SurroundScore}\n")
                f.write(f"params: {params}\n")

    ##plot the graphs of force and kms
    label_fs = 10
    title_fs = 12
    tick_fs = 9

    fig, axs = plt.subplots(3, 1, figsize=(14, 9))

    axs[0].plot(disp, force, '-', linewidth=2)
    axs[0].set_xlabel("Displacement [mm]", fontsize=label_fs, labelpad=10)
    axs[0].set_ylabel("Reaction Force [N]", fontsize=label_fs, labelpad=10)
    axs[0].set_title("Reaction Force - one surround (this FE model)", fontsize=title_fs, pad=14)
    axs[0].tick_params(labelsize=tick_fs)
    axs[0].grid(True)

    # This model only solves ONE surround. This driver mirrors two of them
    # about the zx-plane, so plot 3 below combines this curve with its own
    # mirror image (Kms(x) + Kms(-x)) to get the real assembly's stiffness -
    # shown together here so the doubling/reshaping is easy to see directly.
    axs[1].plot(disp, kms, '-', linewidth=2, color='tab:blue')
    axs[1].set_xlabel("Displacement [mm]", fontsize=label_fs, labelpad=10)
    axs[1].set_ylabel("Stiffness [N/mm]", fontsize=label_fs, labelpad=10)
    axs[1].set_title("Kms - one surround only (not what gets scored)", fontsize=title_fs, pad=14)
    axs[1].tick_params(labelsize=tick_fs)
    axs[1].grid(True)
    axs[1].set_ylim([0, max(kms) * 1.3])

    axs[2].plot(dispDual, kmsDual, '-', linewidth=2, color='tab:blue')
    axs[2].set_xlabel("Displacement [mm]", fontsize=label_fs, labelpad=10)
    axs[2].set_ylabel("Stiffness [N/mm]", fontsize=label_fs, labelpad=10)
    axs[2].set_title("Kms - both surrounds combined (actual physical curve, this is what gets scored)", fontsize=title_fs, pad=14)
    axs[2].tick_params(labelsize=tick_fs)
    axs[2].grid(True)
    axs[2].set_ylim([0, max(kmsDual) * 1.3])

    fig.tight_layout(h_pad=4)

    TimeStamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Plots/plot_axisym_{TimeStamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    try:
        mgr = plt.get_current_fig_manager()
        mgr.window.wm_geometry("+100+20")  # move the window near the top of the screen
    except Exception:
        pass  # backend doesn't support window positioning - not worth failing over

    plt.show()

    return 0

def CreateBCs(NOPs, field, points):
    #masking
    #Inner Curved Surface
    Arc_Center_X = NOPs.ConeWidth/2 - NOPs.ConeCornerRadius
    Arc_Center_Y = NOPs.ConeHeight/2 - NOPs.ConeCornerRadius
    Cone_Z_Height = NOPs.MountFlangeThickness #- 5 #add the -5 for analyzing the IA surround 
    Enclosure_Z_Height = Cone_Z_Height - NOPs.ConeOffset 
    Clamp_Z_Height = Enclosure_Z_Height + NOPs.MountFlangeThickness
    tol = NOPs.Node_find_tol

    x_mod = points[:,0] - Arc_Center_X
    y_mod = points[:,1] - Arc_Center_Y
    z = points[:,2]

    r = np.sqrt(x_mod**2 + y_mod**2)

    R_outer_for_inner = NOPs.ConeCornerRadius+tol
    R_inner_for_outer = NOPs.ConeCornerRadius+NOPs.ConeEnclosureGap
    

    Cone_Corner_Mask = (np.abs(z - Cone_Z_Height) <= tol) &\
        (r<=R_outer_for_inner) &\
        (x_mod>=-tol) & (y_mod>=-tol)
    
    #InnerTop
    Cone_Top_Edge_Mask = (np.abs(z - Cone_Z_Height) <= tol) &\
        (x_mod <= tol) & (y_mod <= NOPs.ConeCornerRadius + tol)

    #InnerRight
    Cone_Right_Edge_Mask = (np.abs(z - Cone_Z_Height) <= tol) &\
        (y_mod <= tol) &\
        (x_mod <= NOPs.ConeCornerRadius + tol)

    #Outer anchored surface

    Enclosure_Mounting_Mask = ((np.abs(z-Enclosure_Z_Height) <= tol)) &\
        (((r>=R_inner_for_outer) & (x_mod >= 0) & (y_mod >= 0)) |\
        ((y_mod>=R_inner_for_outer) | (x_mod >= R_inner_for_outer)))

    Enclosure_Clamp_Mask = ((np.abs(z-Clamp_Z_Height) <= tol)) &\
        (((r>=(R_inner_for_outer-NOPs.ConeEnclosureGap/2)) & (x_mod >= 0) & (y_mod >= 0)) |\
        ((y_mod>=(R_inner_for_outer-NOPs.ConeEnclosureGap/2)) | (x_mod >= (R_inner_for_outer-NOPs.ConeEnclosureGap/2))))


    ###
    #boundary cond
    ###

    # Symmetry plane at x=0, fix only x displacement
    bc_symx = fe.Boundary(
        field[0],
        fx=0.0,             # select points where x ≈ 0
        skip=(False, True, True)  # fix x, skip y and z
    )

    # Symmetry plane at y=0, fix only y displacement
    bc_symy = fe.Boundary(
        field[0],
        fy=0.0,             # select points where y ≈ 0
        skip=(True, False, True)  # fix y, skip x and z
    )

    # Fixed support at z=0, fix all components
    bc_bottom = fe.Boundary(
        field[0],
        mask = Enclosure_Mounting_Mask,             # select points where enclosure mount is
        skip = (False, False, False)
    )
    #Create BC for applying displacements later
    move_z = fe.Boundary(
        field[0],
        mask = (Cone_Corner_Mask | Cone_Top_Edge_Mask | Cone_Right_Edge_Mask),            # selects all nodes from the earlier mask
        skip=(True, True, False) # constrain only Z (doesn't actually matter I think)
    )
    bc_cone = fe.Boundary(
        field[0],
        mask = (Cone_Corner_Mask | Cone_Top_Edge_Mask | Cone_Right_Edge_Mask),            # selects all nodes from the earlier mask
        skip=(False, False, True) # constrain only Z (doesn't actually matter I think)
    )
   

    #group all the bcs together
    bcs = dict()
    bcs["symx"] = bc_symx
    bcs["symy"] = bc_symy
    bcs["ground"] = bc_bottom
    bcs["move_z"] = move_z
    bcs["bc_cone"] = bc_cone

    masks = dict()
    masks["InnerMask"] = Cone_Corner_Mask
    masks["InnerTop"] = Cone_Top_Edge_Mask
    masks["InnerRight"] = Cone_Right_Edge_Mask
    masks["Outer"] = Enclosure_Mounting_Mask
    masks["Clamp"] = Enclosure_Clamp_Mask


    return bcs, masks

#function for adding glyphs to show BCs
def add_bc_nodes(plotter, nodes, color, grid, points):
    nodes_cloud = pv.PolyData(points[nodes])
    bbox = grid.bounds
    radius = 0.003 * max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
    glyphs = nodes_cloud.glyph(scale=False, geom=pv.Sphere(radius=radius), orient = False)
    plotter.add_mesh(glyphs, color=color)

def ShowItOff(NOPs, params):
    import os
    import subprocess
    import imageio

    field, tetra_mesh, tetra_cells, points = CreateFeField(NOPs)

    bcs, masks = CreateBCs(NOPs, field, points)

    print("Number of points in mesh:", tetra_mesh.points.shape[0])

    # PyVista requires each cell to start with the number of points in the cell (4 for tetra)
    cells_pv = np.hstack([np.full((tetra_cells.shape[0], 1), 4), tetra_cells]).flatten()

    #format the data for pyvista
    grid = pv.UnstructuredGrid(cells_pv, np.full(tetra_cells.shape[0], 10), points)
   
    pv_plotter = pv.Plotter()
    pv_plotter.add_mesh(grid, show_edges = True)

    add_bc_nodes(pv_plotter, bcs["symx"].points, 'red', grid, points)
    add_bc_nodes(pv_plotter, bcs["symy"].points, 'green', grid, points)
    add_bc_nodes(pv_plotter, bcs["ground"].points, 'blue', grid, points)
    add_bc_nodes(pv_plotter, bcs["move_z"].points, 'orange', grid, points)

    ##unconnemt this to show boundary conds
    pv_plotter.camera_position = [(-125.0,-125.0,-25),(50.0,75.0,7.0),(0.0,0.0,1.0)]
    
    #pv_plotter.show()

    ###
    #material data
    ###

    material = fe.Hyperelastic(mooney_rivlin, C10 = NOPs.MaterialCoefficients[0], C01 = NOPs.MaterialCoefficients[1])

    # Define your solid body
    solid_out = fe.SolidBody(material, field)
    
    #Make separate bcs for each one
    bc_clamp = fe.Boundary(
        field[0],
        mask = masks["Clamp"],             # select points where enclosure mount is
        skip = (True, True, False)
    )
    bcs["clamp"] = bc_clamp
   


    bcs_out = bcs

    # Define the displacement “ramp” (0 → some value) in linsteps

    steps_out = fe.math.linsteps([0,NOPs.Xmax], num=round(NOPs.N_Steps/2))
 
    # Define a step
    step_out = fe.Step(
        items=[solid_out],
        ramp={bcs_out["move_z"]: steps_out},
        boundaries=bcs_out  # includes your fixed BCs + the “move_z” boundary
    )

   

    # Precompute PyVista cells
    cells_pv = np.hstack([np.full((tetra_cells.shape[0], 1), 4), tetra_cells]).flatten()
    cell_types = np.full(tetra_cells.shape[0], 10)  # 10 = VTK_TETRA

    original_pts = tetra_mesh.points.copy()

    deformed_meshes = []

    #Callback funtion to grab the deformed meshes at each step
    def cbout(stepnum, substepnum, substep, **kwargs):
        # Extract displacement field as NumPy array
        disp_values = substep.x[0].values  # convert Field → ndarray
        pts_def = original_pts + disp_values
        
        # Create a PyVista UnstructuredGrid for this timestep
        mesh_pv = pv.UnstructuredGrid(cells_pv, cell_types, pts_def)
        deformed_meshes.append(mesh_pv)

    def cbin(stepnum, substepnum, substep, **kwargs):
        # Extract displacement field as NumPy array
        disp_values = substep.x[0].values  # convert Field → ndarray
        pts_def = original_pts + disp_values
        
        # Create a PyVista UnstructuredGrid for this timestep
        mesh_pv = pv.UnstructuredGrid(cells_pv, cell_types, pts_def)
        deformed_meshes.insert(0,mesh_pv)


    ### Create the CharacteristicCurve job
    
    job_out = fe.CharacteristicCurve(steps=[step_out], boundary=bcs_out["move_z"], callback = cbout)

    # Evaluate the characteristic curve
    job_out.evaluate()
    
    #Now do the other way, but recompute the starting points first
    field, tetra_mesh, tetra_cells, points = CreateFeField(NOPs)

    bcs, masks= CreateBCs(NOPs, field, points)
    solid_in = fe.SolidBody(material, field)
    bcs_in = bcs
    steps_in = fe.math.linsteps([0,-NOPs.Xmax], num=round(NOPs.N_Steps/2))
     # Define a step
    step_in = fe.Step(
        items=[solid_in],
        ramp={bcs_in["move_z"]: steps_in},
        boundaries=bcs_in  # includes your fixed BCs + the “move_z” boundary
    )
    job_in = fe.CharacteristicCurve(steps=[step_in], boundary=bcs_in["move_z"], callback = cbin)
    job_in.evaluate()
    

    ### For animation
    TimeStamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    gif_path2 = f"SurroundGifs/solution_{TimeStamp}.gif"
    gif_path = "surround_mesh_animation.gif"

    # Create plotter window (visible)
    pl = pv.Plotter(window_size=(900, 700))
    pl.camera_position = [(NOPs.ConeWidth/2,-200.0, 7.0),(NOPs.ConeWidth/2,0, 7.0),(0.0,0.0,1.0)]
    pl.add_mesh(deformed_meshes[0], show_edges=True)
    pl.add_text("Mesh displacement animation", font_size=10)
    cpos = pl.camera_position
    pl.show(auto_close=False, interactive=False)


    #loop through the meshes collected to create frames
    frames = []
    for t in range(len(deformed_meshes)):
        # Update the mesh points of the same PyVista object
        pl.mesh.points[:] = deformed_meshes[t].points  # or replace with new mesh
        pl.render()
        frames.append(pl.screenshot(return_img=True))

    pl.close()
    imageio.mimsave(gif_path, frames + frames[::-1], fps=round(NOPs.N_Steps/2), loop = 0)
    imageio.mimsave(gif_path2, frames + frames[::-1], fps=round(NOPs.N_Steps/2), loop = 0)
    print("GIF saved:", gif_path)

    #open the gif automatically
    try:
        os.startfile(gif_path) #windows
    except AttributeError:
        subprocess.call(["open", gif_path]) #mac or Linux
        

    # Extract data I want
    #split rfs between straight and curved sections
    CurveRf = LocalRF(job_out, masks["InnerMask"])
    InnerTopRF = LocalRF(job_out, masks["InnerTop"])
    InnerRightRF = LocalRF(job_out, masks["InnerRight"])

    Angle_to_top_corner = np.arctan2(NOPs.ConeHeight/2+NOPs.ConeCornerRadius, NOPs.ConeWidth/2)  #abs angle to top starting part of arc
    Angle_to_bottom_corner = np.arctan2(NOPs.ConeHeight/2, NOPs.ConeWidth/2+NOPs.ConeCornerRadius)#abs angle to bottom point of arc
    
    Angular_span_curve = np.degrees(Angle_to_top_corner-Angle_to_bottom_corner)
    Rf_per_deg_curve = CurveRf/Angular_span_curve

    Angular_span_straight = np.degrees(Angle_to_bottom_corner + (np.pi/2-Angle_to_top_corner))
    Rf_per_deg_straight = (InnerTopRF + InnerRightRF)/Angular_span_straight

    print('Reaction force in Newtons per degree (curve vs flat section)')
    print(Rf_per_deg_curve)
    print(Rf_per_deg_straight)

    disp_out = np.array([d[2] for d in job_out.x])      # Z displacement
    force_out = 4*np.array([f[2] for f in job_out.y])     # Z reaction force (total)
    disp_in = np.array([d[2] for d in job_in.x])      # Z displacement
    force_in = 4*np.array([f[2] for f in job_in.y])     # Z reaction force (total)
    
    disp_in = disp_in[1:]
    force_in = force_in[1:]

    force = np.concatenate([force_in[::-1], force_out])
    disp = np.concatenate([disp_in[::-1], disp_out])
    
    # Compute stiffness
    kms = np.gradient(force, disp)

    #compute total Kms with spider
    kmsSpider = [CalcKSpider(u) for u in disp]
    kmsT = kms + kmsSpider
    #print(f'kmsT {kmsT}')

    SurroundScore, WeightedScores = ScoreKms(kms, disp, NOPs)

    print(f'Score = {SurroundScore}')
    print(f'weighted scores = {WeightedScores}')

    with open(NOPs.CachingFile, 'a') as f:
                f.write(f"scores: {WeightedScores}\n")
                f.write(f"score: {SurroundScore}\n")
                f.write(f"params: {params}\n")

    ##plot the graphs of force and kms
    fig, axs = plt.subplots(3, 1, figsize=(17, 8))

    axs[0].plot(disp, force, '-')
    axs[0].set_xlabel("Displacement [mm]")
    axs[0].set_ylabel("Reaction Force [N]")
    axs[0].set_title("Reaction Force")
    axs[0].grid(True)

    axs[1].plot(disp, kms, '-')
    axs[1].set_xlabel("Displacement [mm]")
    axs[1].set_ylabel("Stiffness [N/mm]")
    axs[1].set_title("Kms")
    axs[1].grid(True)
    axs[1].set_ylim([0,max(kms)*1.3])

    axs[2].plot(disp, kmsT, '-', label='Total Kms')
    axs[2].set_xlabel("Displacement [mm]")
    axs[2].set_ylabel("Total Stiffness (including spider)")
    axs[2].set_title("Total Kms")
    axs[2].grid(True)
    axs[2].set_ylim([0, max(kmsT)*1.3])
    axs[2].plot(disp, kmsSpider, '-', color='red', label='Spider Only')
    axs[2].legend(loc='lower left')

    #plt.tight_layout()

    TimeStamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Plots/plot_{TimeStamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()

    return 0 

def AnalyzeItOneWay(NOPs, Direction, fe_mesh_field = None):
    #use this section for debugging parallelization
    # import threading, multiprocessing, os, psutil

    # p = psutil.Process(os.getpid())
    # print(f"Process {os.getpid()} starting sweep {Direction}")
    # print(f"Thread count (Python-level): {threading.active_count()}")
    # print("Threads (OS-level):", p.num_threads())


    ###
    # Moony-Rivlin model params
    ###
    
    C10 = NOPs.MaterialCoefficients[0]
    C01 = NOPs.MaterialCoefficients[1]

    if fe_mesh_field == None:
        field, tetra_mesh, tetra_cells, points = CreateFeField(NOPs)
    
    else:
        field = fe_mesh_field
        tetra_mesh = field[0].region.mesh
        points = tetra_mesh.points

    bcs, masks = CreateBCs(NOPs, field, points)
   
    ###
    #material data
    ###

    material = fe.Hyperelastic(mooney_rivlin, C10=C10, C01=C01)
    

    # Define your solid body

    solid = fe.SolidBody(material, field)
    
    # Define the displacement “ramp” (0 → some value) in linsteps

    steps = fe.math.linsteps([0,NOPs.Xmax*Direction], num=round(NOPs.N_Steps/2))
    
    if Direction == 1:
        bc_clamp = fe.Boundary(
            field[0],
            mask = masks["Clamp"],             # select points where enclosure mount is
            skip = (True, True, False)
        )
        bcs["clamp"] = bc_clamp
        print('Clamping')

    # Define a step
    step = fe.Step(
        items=[solid],
        ramp={bcs["move_z"]: steps},
        boundaries=bcs  # includes your fixed BCs + the “move_z” boundary
    )
    
    ### Create the CharacteristicCurve job
    job = fe.CharacteristicCurve(steps=[step], boundary=bcs["move_z"])
    # Evaluate the characteristic curve

    job.evaluate(solver = spsolve)#filename=xdmf_filename)

    # Extract data useful for optimization
    disp = np.array([d[2] for d in job.x])      # Z displacement
    force = 4*np.array([f[2] for f in job.y])     # Z reaction force (total)

    # Compute stiffness
    kms = np.gradient(force, disp)

    return kms, disp

def CalcKSpider(X):
    # K = 5 + abs(X)*10/40  This is obsolete now
    K = 8.76 + 0.0188*X - 2.88E-4*X**2 - 2.94E-5*X**3 + 4.06E-6*X**4

    return K

def ScoreKms(Kms, Disp, NOPs):
    Volume = 0  # TODO: replace with a Pappus-theorem volume calc from the 2D axisymmetric mesh
    print("Volume=", Volume)
    #Calculate KmsFlatness
    KmsAvg = np.mean(Kms)
    KmsDev = Kms - KmsAvg
    KmsF = np.sum(KmsDev**2)/len(Kms)

    
    #Calculate Kms90Flatness
    #Now more like kms50 flatness
    n = len(Kms)
    start = int(n*.23)
    end = n-start
    Kms90 = Kms[start:end]
    Kms90Avg = np.mean(Kms90)
    print(f'kms90avg {Kms90Avg}')
    Kms90Dev = Kms90 - Kms90Avg
    Kms90F = np.sum(Kms90Dev**2)/len(Kms90)

    #print(f'Disp90, {Disp[start:end]}')

    #Calculate symmetry about x=0
    ZeroIndex = len(Kms90)//2

    if len(Kms90) % 2 ==0:
        LeftSide = Kms90[:ZeroIndex]
    else:
        LeftSide = Kms90[:(ZeroIndex+1)]
    RightSide = Kms90[ZeroIndex:]
    RightSideMirror = RightSide[::-1]
    Asym = np.sum((LeftSide-RightSideMirror)**2)

    

    #Calculate shift (difference in avg stiffness from target stiffness)
    Shift = (KmsAvg - NOPs.TargetStiffness)**2
    
    
    if NOPs.IncludeSpider == True:
        #Calculate total stiffness (spider + surround)
        KmsSpider = [CalcKSpider(u) for u in Disp]
        KmsT = Kms + KmsSpider
        
        #Calculate KmsT90
        KmsT90 = KmsT[start:end]
        KmsT90Avg = np.mean(KmsT90)
        KmsT90Dev = KmsT90 - KmsT90Avg
        KmsT90F = np.sum(KmsT90Dev**2)/len(KmsT90)

        #Calculate KmsT Flatness
        KmsTAvg = np.mean(KmsT)
        KmsTDev = KmsT - KmsTAvg
        KmsTF = np.sum(KmsTDev**2)/len(KmsT90)

        Scores = np.array([KmsTF, KmsT90F, Volume, Shift, Asym], dtype = float)
    else:
        Scores = np.array([KmsF, Kms90F, Volume, Shift, Asym], dtype = float)

    PureWeights = [w[1] for w in NOPs.OptimizationWeights]
    PureWeights = np.array(PureWeights, dtype = float)
    
    WeightedScores = PureWeights*Scores
    SurroundScore = np.sum(WeightedScores)

    return SurroundScore, WeightedScores

def main():

  
    return 0 

if __name__ == "__main__":
    main()
