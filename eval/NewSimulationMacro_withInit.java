package macro;

import java.util.*;

import star.common.*;
import star.base.neo.*;
import star.meshing.*;
import star.cadmodeler.*;
import star.material.*;
import star.base.report.*;
import star.coupledflow.*;
import star.flow.*;
import star.energy.*;
import star.metrics.*;
import star.topologyoptimization.*;
import star.automation.*;
import star.twodmesher.*;
import star.resurfacer.*;
import star.turbulence.*;

// import star.adjoint.*; // Not available in this environment/version; avoid explicit import
// imports for json reading
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.lang.reflect.Method;
public class NewSimulationMacro extends StarMacro {

    private static final String STEP_FILE_PATH = "./box_with_pipes.step";
    private static final String CSV_FILE_PATH = "./";
    private static final double SPLIT_ANGLE = 89.0;

  private static final double DEFAULT_DENSITY = 1.0;
  private static final double DEFAULT_VISCOSITY = 0.001;
  private static final double DEFAULT_REL_CHANGE_THRESHOLD = 5e-3; // 0.5%
  private static final int DEFAULT_MAX_ITER = 10;
  private static final double DEFAULT_INITIAL_PRIMAL_ITERATIONS = 500.0;
  private static final double DEFAULT_OPTIMIZATION_PRIMAL_ITERATIONS = 300.0;
  private static final double DEFAULT_FIRST_ITER_PRIMAL_ITERATIONS = 200.0;
  private static final double DEFAULT_TOPOLOGY_STEP_SIZE = 10.0;
  private static final double DEFAULT_FIRST_TOPOLOGY_STEP_SIZE = 0.05;
  private static final double DEFAULT_WARMUP_PRIMAL_ITERATIONS = 600.0;
  private static final int DEFAULT_WARMUP_CYCLES = 1;
  private static final double DEFAULT_BRINKMAN_PENALTY = 1.0e7; // 1/m^2
  private static final double DEFAULT_VOLUME_LOSS_WEIGHT = 1.0e-3; // dimensionless
  private static final double DEFAULT_GEOMETRY_SCALE = 1.0; // unitless
  private static final double DEFAULT_MATIND_VALUE = 1.0; // dimensionless
  // Runtime values (overridable via JSON)
  private double densityValue = DEFAULT_DENSITY;
  private double viscosityValue = DEFAULT_VISCOSITY;
  private double relChangeThreshold = DEFAULT_REL_CHANGE_THRESHOLD; // overridable via JSON key: early_stop_rel_change
  private double brinkmanPenalty = DEFAULT_BRINKMAN_PENALTY; // overridable via JSON key: brinkman_penalty
  private double volumeLossWeight = DEFAULT_VOLUME_LOSS_WEIGHT; // overridable via JSON key: volume_loss_weight
  private String turbulenceModel = "laminar"; // overridable via JSON key: turbulence_model (laminar|k-epsilon|k-omega)
  private String wallTreatment = "all-y+"; // overridable via JSON key: wall_treatment (all-y+|two-layer|low-y+)
  private double geometryScale = DEFAULT_GEOMETRY_SCALE; // overridable via JSON key: geometry_scale
  // Non-uniform geometry scaling support (overridable via JSON key: geometry_scale_xyz: [sx, sy, sz])
  private boolean useGeometryScaleXYZ = false;
  private double geometryScaleX = 1.0, geometryScaleY = 1.0, geometryScaleZ = 1.0;
  // Primal iteration counts (overridable via JSON keys: initial_primal_iterations, optimization_primal_iterations)
  private double initialPrimalIterations = DEFAULT_INITIAL_PRIMAL_ITERATIONS;
  private double optimizationPrimalIterations = DEFAULT_OPTIMIZATION_PRIMAL_ITERATIONS;
  private double firstIterationPrimalIterations = DEFAULT_FIRST_ITER_PRIMAL_ITERATIONS;
  private double topologyStepSize = DEFAULT_TOPOLOGY_STEP_SIZE;
  private double firstTopologyStepSize = DEFAULT_FIRST_TOPOLOGY_STEP_SIZE;
  // Warm-up flow solve before first export (overridable via JSON keys: warmup_primal_iterations, warmup_cycles)
  private double warmupPrimalIterations = DEFAULT_WARMUP_PRIMAL_ITERATIONS;
  private int warmupCycles = DEFAULT_WARMUP_CYCLES;
  // Material indicator initialization controls
  private String matIndInitMode = "table"; // constant|expression|table
  private double matIndInitValue = DEFAULT_MATIND_VALUE;
  private String matIndInitExpression = null;
  private String matIndInitTablePath = "./pred_material_indicator.csv";
  // Cached reports to avoid name lookups during loop
  private VolumeIntegralReport matIndIntegralReport;
  private VolumeIntegralReport regionVolumeIntegralReport;
  private PressureDropReport pressureDropReport;
  private SimDriverWorkflow topoWorkflow;
    private static final int COUNTER_MACRO = 0;
    //read json file and get value for counter
    private static final String JSON_FILE_PATH = "./topology_config.json";


    @Override
    public void execute() {
      // read json file and get value for counter
        int counter = COUNTER_MACRO;
    try {
      Path path = Paths.get(JSON_FILE_PATH);
      String content = Files.readString(path);

      Integer parsedCounter = parseJsonInt(content, "counter");
      if (parsedCounter != null) {
        counter = parsedCounter.intValue();
      }

      Double parsedVisc = parseJsonDouble(content, "viscosity");
      if (parsedVisc != null && parsedVisc > 0) {
        viscosityValue = parsedVisc;
      } else {
        viscosityValue = DEFAULT_VISCOSITY;
      }

      Double parsedRho = parseJsonDouble(content, "density");
      if (parsedRho != null && parsedRho > 0) {
        densityValue = parsedRho;
      } else {
        densityValue = DEFAULT_DENSITY;
      }

      // early-stop relative change threshold (suggested range: 1e-6 .. 0.2)
      Double parsedRel = parseJsonDouble(content, "early_stop_rel_change");
      if (parsedRel != null && parsedRel > 0 && parsedRel < 1) {
        relChangeThreshold = parsedRel;
      } else {
        relChangeThreshold = DEFAULT_REL_CHANGE_THRESHOLD;
      }

      // brinkman penalty (1/m^2)
      Double parsedBrink = parseJsonDouble(content, "brinkman_penalty");
      if (parsedBrink != null && parsedBrink > 0) {
        brinkmanPenalty = parsedBrink;
      } else {
        brinkmanPenalty = DEFAULT_BRINKMAN_PENALTY;
      }

      // geometry scale (unitless > 0)
      Double parsedScale = parseJsonDouble(content, "geometry_scale");
      if (parsedScale != null && parsedScale > 0) {
        geometryScale = parsedScale;
      } else {
        geometryScale = DEFAULT_GEOMETRY_SCALE;
      }

      // optional non-uniform geometry scale: geometry_scale_xyz: [sx, sy, sz]
      double[] parsedScaleXYZ = parseJsonDoubleArray(content, "geometry_scale_xyz", 3);
      if (parsedScaleXYZ != null) {
        // validate > 0
        if (parsedScaleXYZ[0] > 0 && parsedScaleXYZ[1] > 0 && parsedScaleXYZ[2] > 0) {
          geometryScaleX = parsedScaleXYZ[0];
          geometryScaleY = parsedScaleXYZ[1];
          geometryScaleZ = parsedScaleXYZ[2];
          useGeometryScaleXYZ = true;
        } else {
          useGeometryScaleXYZ = false; // fall back to uniform
        }
      } else {
        useGeometryScaleXYZ = false; // use uniform scale if provided
      }

      // turbulence model (string): laminar | k-epsilon | k-omega | sst
      String parsedTm = parseJsonString(content, "turbulence_model");
      if (parsedTm != null) {
        String tml = parsedTm.trim().toLowerCase(Locale.ROOT);
        if (tml.contains("eps")) {
          turbulenceModel = "k-epsilon";
        } else if (tml.contains("omega") || tml.contains("sst") || tml.equals("k-\u03c9") || tml.equals("k-w")) {
          turbulenceModel = "k-omega"; // treat SST as k-omega family
        } else {
          turbulenceModel = "laminar";
        }
      } else {
        turbulenceModel = "laminar";
      }

      // wall treatment (string): all-y+ | two-layer | low-y+ (default all-y+)
      String parsedWt = parseJsonString(content, "wall_treatment");
      if (parsedWt != null) {
        String wtl = parsedWt.trim().toLowerCase(Locale.ROOT);
        if (wtl.contains("two") || wtl.contains("2")) {
          wallTreatment = "two-layer";
        } else if (wtl.contains("low")) {
          wallTreatment = "low-y+";
        } else {
          wallTreatment = "all-y+";
        }
      } else {
        wallTreatment = "all-y+";
      }

      // volume_loss_weight (dimensionless >= 0)
      Double parsedW = parseJsonDouble(content, "volume_loss_weight");
      if (parsedW != null && parsedW >= 0.0) {
        volumeLossWeight = parsedW;
      } else {
        volumeLossWeight = DEFAULT_VOLUME_LOSS_WEIGHT;
      }

      // Primal iteration counts
      Double parsedInitPrimal = parseJsonDouble(content, "initial_primal_iterations");
      if (parsedInitPrimal != null && parsedInitPrimal > 0) {
        initialPrimalIterations = parsedInitPrimal;
      } else {
        initialPrimalIterations = DEFAULT_INITIAL_PRIMAL_ITERATIONS;
      }

      Double parsedOptPrimal = parseJsonDouble(content, "optimization_primal_iterations");
      if (parsedOptPrimal != null && parsedOptPrimal > 0) {
        optimizationPrimalIterations = parsedOptPrimal;
      } else {
        optimizationPrimalIterations = DEFAULT_OPTIMIZATION_PRIMAL_ITERATIONS;
      }

      Double parsedTopoStep = parseJsonDouble(content, "topology_step_size");
      if (parsedTopoStep != null && parsedTopoStep > 0) {
        topologyStepSize = parsedTopoStep;
      } else {
        topologyStepSize = DEFAULT_TOPOLOGY_STEP_SIZE;
      }

      Double parsedFirstTopoStep = parseJsonDouble(content, "first_topology_step_size");
      if (parsedFirstTopoStep != null && parsedFirstTopoStep > 0) {
        firstTopologyStepSize = parsedFirstTopoStep;
      } else {
        firstTopologyStepSize = DEFAULT_FIRST_TOPOLOGY_STEP_SIZE;
      }

      Double parsedFirstPrimal = parseJsonDouble(content, "first_iteration_primal_iterations");
      if (parsedFirstPrimal != null && parsedFirstPrimal > 0) {
        firstIterationPrimalIterations = parsedFirstPrimal;
      } else {
        firstIterationPrimalIterations = DEFAULT_FIRST_ITER_PRIMAL_ITERATIONS;
      }

      Double parsedWarmupPrimal = parseJsonDouble(content, "warmup_primal_iterations");
      if (parsedWarmupPrimal != null && parsedWarmupPrimal > 0) {
        warmupPrimalIterations = parsedWarmupPrimal;
      } else {
        warmupPrimalIterations = DEFAULT_WARMUP_PRIMAL_ITERATIONS;
      }

      Integer parsedWarmupCycles = parseJsonInt(content, "warmup_cycles");
      if (parsedWarmupCycles != null && parsedWarmupCycles.intValue() >= 0) {
        warmupCycles = parsedWarmupCycles.intValue();
      } else {
        warmupCycles = DEFAULT_WARMUP_CYCLES;
      }

      String parsedMiMode = parseJsonString(content, "material_indicator_mode");
      if (parsedMiMode != null && !parsedMiMode.isEmpty()) {
        matIndInitMode = parsedMiMode.trim().toLowerCase(Locale.ROOT);
      } else {
        matIndInitMode = "table";
      }

      Double parsedMiVal = parseJsonDouble(content, "material_indicator_value");
      if (parsedMiVal != null) {
        matIndInitValue = parsedMiVal;
      } else {
        matIndInitValue = DEFAULT_MATIND_VALUE;
      }

      String parsedMiExpr = parseJsonString(content, "material_indicator_expression");
      if (parsedMiExpr != null && !parsedMiExpr.isEmpty()) {
        matIndInitExpression = parsedMiExpr;
      } else {
        matIndInitExpression = null;
      }

      String parsedMiTable = parseJsonString(content, "material_indicator_table");
      if (parsedMiTable != null && !parsedMiTable.isEmpty()) {
        matIndInitTablePath = parsedMiTable;
      } else {
        matIndInitTablePath = "./pred_material_indicator.csv";
      }
    } catch (Exception e) {
      // On any failure, revert to defaults
      viscosityValue = DEFAULT_VISCOSITY;
      densityValue = DEFAULT_DENSITY;
      relChangeThreshold = DEFAULT_REL_CHANGE_THRESHOLD;
      brinkmanPenalty = DEFAULT_BRINKMAN_PENALTY;
      volumeLossWeight = DEFAULT_VOLUME_LOSS_WEIGHT;
      turbulenceModel = "laminar";
      wallTreatment = "all-y+";
      geometryScale = DEFAULT_GEOMETRY_SCALE;
      useGeometryScaleXYZ = false;
      geometryScaleX = geometryScaleY = geometryScaleZ = 1.0;
      initialPrimalIterations = DEFAULT_INITIAL_PRIMAL_ITERATIONS;
      optimizationPrimalIterations = DEFAULT_OPTIMIZATION_PRIMAL_ITERATIONS;
      warmupPrimalIterations = DEFAULT_WARMUP_PRIMAL_ITERATIONS;
      warmupCycles = DEFAULT_WARMUP_CYCLES;
      matIndInitMode = "table";
      matIndInitValue = DEFAULT_MATIND_VALUE;
      matIndInitExpression = null;
      matIndInitTablePath = "./pred_material_indicator.csv";
      e.printStackTrace();
    }
    Simulation simulation = getActiveSimulation();
    // Ensure snapshot parameters exist before any automation or report references
    ensureSnapshotParameters(simulation);
    setupSimulation(simulation);
    int maxIter = (counter > 0) ? counter : DEFAULT_MAX_ITER;
    setupTopologyOptimizationAutomation(simulation, maxIter);
    setupMaterialIndicatorReports(simulation);
        //runTopologyOptimization(simulation);
        XyzInternalTable table = setupTable(simulation);
        // exportTable(table, savePath);

    double prevAvgMatInd = Double.NaN;

    // Optional warm-up flow solve before first export to reduce initial pressure-drop transient
    if (warmupCycles > 0 && warmupPrimalIterations > 0) {
      runWarmupPhysics(simulation, warmupCycles, warmupPrimalIterations, optimizationPrimalIterations);
    }

    // determine iteration bounds: start from simulation counter parameter if present,
    // end at 'counter' from JSON (parsed earlier) or DEFAULT_MAX_ITER
    int startIter = 0;
    try {
      Object obj = simulation.get(GlobalParameterManager.class).getObject("counter");
      if (obj instanceof ScalarGlobalParameter) {
        ScalarGlobalParameter sgp = (ScalarGlobalParameter) obj;
        try {
          startIter = (int) Math.round(sgp.getQuantity().getRawValue());
        } catch (Exception e) {
          startIter = 0;
        }
      }
    } catch (Exception e) {
      // fallback
      startIter = 0;
    }

    simulation.println(String.format(
      Locale.US,
      "Run config: density=%.6g [kg/m^3], viscosity=%.6g [Pa-s], brinkman_penalty=%.6g [kg/m^3-s], volume_loss_weight=%.6g, turbulence=%s, wall_treatment=%s, %s, early_stop_rel=%.3g, iter_start=%d, iter_end=%d",
      densityValue, viscosityValue, brinkmanPenalty, volumeLossWeight, turbulenceModel, wallTreatment,
      (useGeometryScaleXYZ
        ? String.format(Locale.US, "geometry_scale_xyz=[%.6g, %.6g, %.6g]", geometryScaleX, geometryScaleY, geometryScaleZ)
        : String.format(Locale.US, "geometry_scale=%.6g", geometryScale)
      ),
      relChangeThreshold, startIter, maxIter));

    for (int i = startIter; i <= maxIter; i++) {
      // Apply topology step size each outer iteration using loop index (not stale counter param)
      try {
        TopologyOptimizationSolver topologySolver = simulation.getSolverManager().getSolver(TopologyOptimizationSolver.class);
        double targetStep = (i <= startIter) ? firstTopologyStepSize : topologyStepSize;
        setTopologyStepSize(simulation, topologySolver, targetStep);
      } catch (Exception e) {
        simulation.println("Warning: failed to set topology step size: " + e.getMessage());
      }

      runTopologyOptimization(simulation);

    // Evaluate material indicator integral and region volume to compute average
    double integralVal = getScalarReportValue(matIndIntegralReport);
    double volumeVal = getScalarReportValue(regionVolumeIntegralReport);
      double avgMatInd = (!Double.isNaN(integralVal) && volumeVal > 0.0) ? (integralVal / volumeVal) : Double.NaN;

      // Early stopping check on relative percentage change
      Double relChangeDisplay = null;
      if (!Double.isNaN(prevAvgMatInd) && !Double.isNaN(avgMatInd)) {
        double denom = Math.max(Math.abs(prevAvgMatInd), 1e-12);
        double relChange = Math.abs(avgMatInd - prevAvgMatInd) / denom;
        relChangeDisplay = relChange;
        if (relChange < relChangeThreshold) {
          simulation.println(String.format("Early stop at iter %d: avg(MaterialIndicator) change %.6f < threshold %.6f", i, relChange, relChangeThreshold));
          // Ensure pressure drop report is current for the final export on early stop
          getReportValue(pressureDropReport);
          exportTable(table, CSV_FILE_PATH + "exportData_iter_" + Integer.toString(i) + ".csv");
          break;
        }
      }

      simulation.println(String.format(
        "Iter %d: avg(MaterialIndicator)=%.6f relChange=%s threshold=%.6f",
        i,
        avgMatInd,
        (relChangeDisplay == null ? "NA" : String.format("%.6f", relChangeDisplay)),
        relChangeThreshold
      ));

      prevAvgMatInd = avgMatInd;

      // Ensure pressure drop report is up-to-date before exporting the CSV
      getReportValue(pressureDropReport);

      exportTable(table, CSV_FILE_PATH + "exportData_iter_" + Integer.toString(i) + ".csv");
    }
    exportTable(table, CSV_FILE_PATH + "exportData_final" + ".csv");
    }

    // Create snapshot globals up front so any later reference (workflow/reports) won't fail
    private void ensureSnapshotParameters(Simulation simulation) {
      // Pressure drop snapshot (pressure units)
      try {
        ScalarGlobalParameter pdSnapshotParam = null;
        try {
          Object obj = simulation.get(GlobalParameterManager.class).getObject("pressure_drop_snapshot");
          if (obj instanceof ScalarGlobalParameter) {
            pdSnapshotParam = (ScalarGlobalParameter) obj;
          }
        } catch (Exception ignore) { /* create new */ }
        if (pdSnapshotParam == null) {
          pdSnapshotParam = simulation.get(GlobalParameterManager.class)
            .createGlobalParameter(ScalarGlobalParameter.class, "pressure_drop_snapshot");
        }
        Units pUnits = simulation.getUnitsManager().getPreferredUnits(Dimensions.Builder().pressure(1).build());
        pdSnapshotParam.getQuantity().setValueAndUnits(0.0, pUnits);
      } catch (Throwable t) {
        simulation.println("Warning: ensureSnapshotParameters failed to init pressure_drop_snapshot: " + t.getMessage());
      }

      // Volume integral snapshot (volume units)
      try {
        ScalarGlobalParameter volSnapshotParam = null;
        try {
          Object obj = simulation.get(GlobalParameterManager.class).getObject("volume_integral_snapshot");
          if (obj instanceof ScalarGlobalParameter) {
            volSnapshotParam = (ScalarGlobalParameter) obj;
          }
        } catch (Exception ignore) { /* create new */ }
        if (volSnapshotParam == null) {
          volSnapshotParam = simulation.get(GlobalParameterManager.class)
            .createGlobalParameter(ScalarGlobalParameter.class, "volume_integral_snapshot");
        }
        Units vUnits = simulation.getUnitsManager().getPreferredUnits(Dimensions.Builder().length(3).build());
        volSnapshotParam.getQuantity().setValueAndUnits(0.0, vUnits);
      } catch (Throwable t) {
        simulation.println("Warning: ensureSnapshotParameters failed to init volume_integral_snapshot: " + t.getMessage());
      }
    }

    private void setupSimulation(Simulation simulation) {
        CompositePart compositePart = importCadFile(simulation);
        splitSurfaces(compositePart);
        GeometryPart meshPart = subtractParts(simulation, compositePart);
        if (meshPart == null) {
            meshPart = compositePart; // Fallback to composite part if subtract failed
        }
        
        try {
            prepareFor2dMeshing(simulation, meshPart);
        } catch (Throwable t) {
            simulation.println("Warning: 2D prepare failed: " + t.getMessage() + ". Continuing anyway.");
        }
        
        createRegionAndBoundaries(simulation, meshPart);
        setBoundaryTypes(simulation);
        setupPhysics(simulation);
        setupMeshing(simulation, meshPart);
        // Recorder-style: scale the mesh after mesh generation (uniform or per-axis)
        applyMeshScaleIfRequested(simulation);
        setupPressureDropReport(simulation);
        setupTopologyOptimization(simulation);
        initializeMaterialIndicator(simulation);
    }

    // Recorder-style mesh scaling using RepresentationManager.scaleMesh
    private void applyMeshScaleIfRequested(Simulation simulation) {
      double sx, sy, sz;
      if (useGeometryScaleXYZ) {
        sx = geometryScaleX; sy = geometryScaleY; sz = geometryScaleZ;
      } else {
        sx = sy = sz = geometryScale;
      }
      boolean needsScale = (Math.abs(sx - 1.0) > 1e-12) || (Math.abs(sy - 1.0) > 1e-12) || (Math.abs(sz - 1.0) > 1e-12);
      if (!needsScale) return;
      try {
        Region region = simulation.getRegionManager().getRegion("Region");
        if (region == null) {
          simulation.println("Warning: Mesh scaling requested but region 'Region' not found.");
          return;
        }
        LabCoordinateSystem labCS = simulation.getCoordinateSystemManager().getLabCoordinateSystem();
        simulation.getRepresentationManager().scaleMesh(
          new ArrayList<>(Arrays.<Region>asList(region)),
          new DoubleVector(new double[] { sx, sy, sz }),
          labCS
        );
        if (useGeometryScaleXYZ) {
          simulation.println(String.format(Locale.US, "Mesh scaled by XYZ factors [%.6g, %.6g, %.6g]", sx, sy, sz));
        } else {
          simulation.println(String.format(Locale.US, "Mesh scaled uniformly by factor %.6g", sx));
        }
      } catch (Throwable t) {
        simulation.println("Warning: Failed to scale mesh: " + t.getMessage());
      }
    }


    private void initializeMaterialIndicator(Simulation simulation) {
      ScalarProfile profile = resolveMaterialIndicatorProfile(simulation);
      if (profile == null) {
        simulation.println("Warning: InitialMaterialIndicatorProfile not found; skipping initialization.");
        return;
      }

      String mode = (matIndInitMode == null ? "constant" : matIndInitMode.toLowerCase(Locale.ROOT));
      try {
        if ("expression".equals(mode) && matIndInitExpression != null && !matIndInitExpression.isEmpty()) {
          profile.setDefinition(matIndInitExpression);
          simulation.println("MaterialIndicator initialized from expression.");
          return;
        }

        if ("table".equals(mode) && matIndInitTablePath != null && !matIndInitTablePath.isEmpty()) {
          Table table = loadXyzTable(simulation, matIndInitTablePath);
          if (table != null) {
            profile.setTabularXyzMethod(table, "MaterialIndicator");
            simulation.println("MaterialIndicator initialized from table: " + matIndInitTablePath);
            return;
          }
          simulation.println("Warning: table mode requested but table could not be loaded; falling back to constant.");
        }

        profile.setValue(matIndInitValue);
        simulation.println(String.format(Locale.US, "MaterialIndicator initialized to constant %.6f", matIndInitValue));
      } catch (Throwable t) {
        simulation.println("Warning: Failed to initialize MaterialIndicator: " + t.getMessage());
      }
    }

    private ScalarProfile resolveMaterialIndicatorProfile(Simulation simulation) {
      try {
        Region r = simulation.getRegionManager().getRegion("Region");
        if (r != null) {
          Object p = r.getValues().get(InitialMaterialIndicatorProfile.class);
          if (p instanceof ScalarProfile) return (ScalarProfile) p;
        }
      } catch (Throwable ignore) { }

      try {
        PhysicsContinuum pc = (PhysicsContinuum) simulation.getContinuumManager().getContinuum("Physics 1");
        if (pc != null) {
          Object p = pc.getInitialConditions().get(InitialMaterialIndicatorProfile.class);
          if (p instanceof ScalarProfile) return (ScalarProfile) p;
        }
      } catch (Throwable ignore) { }

      try {
        PhysicsContinuum pc = (PhysicsContinuum) simulation.getContinuumManager().getContinuum("Physics 1");
        if (pc != null) {
          TopologyOptimizationModel tom = pc.getModelManager().getModel(TopologyOptimizationModel.class);
          if (tom != null) {
            Method m = tom.getClass().getMethod("getInitialMaterialIndicatorProfile");
            Object p = m.invoke(tom);
            if (p instanceof ScalarProfile) return (ScalarProfile) p;
          }
        }
      } catch (Throwable ignore) { }

      return null;
    }

    private Table loadXyzTable(Simulation simulation, String path) {
      try {
        TableManager tm = simulation.getTableManager();
        try {
          Method m = tm.getClass().getMethod("createFromFile", Class.class, String.class);
          Object t = m.invoke(tm, XyzInternalTable.class, path);
          if (t instanceof Table) return (Table) t;
        } catch (Throwable ignore) { }

        try {
          Method m = tm.getClass().getMethod("createFromFile", String.class);
          Object t = m.invoke(tm, path);
          if (t instanceof Table) return (Table) t;
        } catch (Throwable ignore) { }

        XyzInternalTable t = (XyzInternalTable) tm.create("star.common.XyzInternalTable");
        String[] mnames = new String[] {"importFile", "importAsciiFile", "importXyzFile", "read", "loadFile"};
        for (String mn : mnames) {
          try {
            Method imp = t.getClass().getMethod(mn, String.class);
            imp.invoke(t, path);
            return t;
          } catch (Throwable ignore) { /* try next */ }
        }
        simulation.println("Warning: XyzInternalTable import not available; table init skipped.");
      } catch (Throwable t) {
        simulation.println("Warning: loadXyzTable failed for " + path + ": " + t.getMessage());
      }
      return null;
    }


    private CompositePart importCadFile(Simulation simulation) {
        PartImportManager importManager = simulation.get(PartImportManager.class);
        importManager.importCadPart(resolvePath(STEP_FILE_PATH), "SharpEdges", 30.0, 2, true, 1.0E-5, true, false, false, false, true, NeoProperty.fromString("{\'NX\': 0, \'STEP\': 0, \'SE\': 0, \'CGR\': 0, \'SW\': 0, \'IFC\': 0, \'ACIS\': 0, \'JT\': 0, \'IGES\': 0, \'CATIAV5\': 0, \'CATIAV4\': 0, \'3DXML\': 0, \'CREO\': 0, \'INV\': 0}"), true, false);
        CompositePart cp = (CompositePart) simulation.get(SimulationPartManager.class).getPart("pipe"); //pipe
        // Note: We scale the mesh after meshing using RepresentationManager.scaleMesh; no part scaling here.
        return cp;
    }

    private void applyGeometryScale(Simulation simulation, CompositePart part, double scale) {
      // Delegate uniform scaling to the per-axis method
      applyGeometryScale(simulation, part, scale, scale, scale);
    }

    private void applyGeometryScale(Simulation simulation, CompositePart part, double sx, double sy, double sz) {
      try {
        MeshOperationManager mom = simulation.get(MeshOperationManager.class);
        // Try ScalePartsOperation
        try {
          Class<?> scaleOpCls = Class.forName("star.meshing.ScalePartsOperation");
          Object op = mom.getClass().getMethod("createScalePartsOperation", NeoObjectVector.class)
              .invoke(mom, new NeoObjectVector(new Object[]{ part }));
          boolean set = false;
          try {
            // Common pattern: op.getScale().setXYZ(scale,scale,scale)
            Method mGetScale = op.getClass().getMethod("getScale");
            Object scaleObj = mGetScale.invoke(op);
            try {
              Method mSetXYZ = scaleObj.getClass().getMethod("setXYZ", double.class, double.class, double.class);
              mSetXYZ.invoke(scaleObj, sx, sy, sz);
              set = true;
            } catch (Throwable ignore2) {
              try {
                Method mSetComponents = scaleObj.getClass().getMethod("setComponents", double.class, double.class, double.class);
                mSetComponents.invoke(scaleObj, sx, sy, sz);
                set = true;
              } catch (Throwable ignore3) { /* continue */ }
            }
          } catch (Throwable ignore) { /* continue */ }
          if (!set) {
            // Fallback: op.setScale if available
            try {
              Method mSetScale = op.getClass().getMethod("setScale", double.class, double.class, double.class);
              mSetScale.invoke(op, sx, sy, sz);
              set = true;
            } catch (Throwable ignore4) { /* continue */ }
          }
          // Execute operation
          op.getClass().getMethod("execute").invoke(op);
          simulation.println(String.format(Locale.US, "Geometry scaled by factors [%.6g, %.6g, %.6g] (ScalePartsOperation)", sx, sy, sz));
          return;
        } catch (Throwable ignoreOp1) { /* try next approach */ }

        // Try TransformPartsOperation with scale
        try {
          Object op = mom.getClass().getMethod("createTransformPartsOperation", NeoObjectVector.class)
              .invoke(mom, new NeoObjectVector(new Object[]{ part }));
          boolean set = false;
          try {
            Method mGetTransform = op.getClass().getMethod("getTransform");
            Object transform = mGetTransform.invoke(op);
            try {
              Method mGetScale = transform.getClass().getMethod("getScale");
              Object scaleObj = mGetScale.invoke(transform);
              try {
                Method mSetXYZ = scaleObj.getClass().getMethod("setXYZ", double.class, double.class, double.class);
                mSetXYZ.invoke(scaleObj, sx, sy, sz);
                set = true;
              } catch (Throwable ignore3) {
                try {
                  Method mSetComponents = scaleObj.getClass().getMethod("setComponents", double.class, double.class, double.class);
                  mSetComponents.invoke(scaleObj, sx, sy, sz);
                  set = true;
                } catch (Throwable ignore4) { /* continue */ }
              }
            } catch (Throwable ignore2) {
              // Fallback: setScale directly on transform
              try {
                Method mSetScale = transform.getClass().getMethod("setScale", double.class, double.class, double.class);
                mSetScale.invoke(transform, sx, sy, sz);
                set = true;
              } catch (Throwable ignore5) { /* continue */ }
            }
          } catch (Throwable ignore1) { /* continue */ }
          op.getClass().getMethod("execute").invoke(op);
          simulation.println(String.format(Locale.US, "Geometry scaled by factors [%.6g, %.6g, %.6g] (TransformPartsOperation)", sx, sy, sz));
          return;
        } catch (Throwable ignoreOp2) { /* no transform op */ }

        simulation.println("Warning: Could not find a geometry scaling operation in this version; geometry_scale ignored.");
      } catch (Throwable t) {
        simulation.println("Warning: Failed to scale geometry: " + t.getMessage());
      }
    }

    private void splitSurfaces(CompositePart compositePart) {
        try {
            for (GeometryPart part : compositePart.getChildParts().getParts()) {
                if (part instanceof CadPart) {
                    CadPart cadPart = (CadPart) part;
                    try {
                        PartSurface surface = cadPart.getPartSurfaceManager().getPartSurface("Faces");
                        if (surface != null) {
                            cadPart.getPartSurfaceManager().splitPartSurfacesByAngle(new NeoObjectVector(new Object[] {surface}), SPLIT_ANGLE);
                        }
                    } catch (Throwable t) {
                        // Surface "Faces" might not exist; try splitting all surfaces instead
                        try {
                            Collection<PartSurface> allSurfaces = cadPart.getPartSurfaceManager().getPartSurfaces();
                            if (!allSurfaces.isEmpty()) {
                                cadPart.getPartSurfaceManager().splitPartSurfacesByAngle(
                                    new NeoObjectVector(allSurfaces.toArray()), SPLIT_ANGLE);
                            }
                        } catch (Throwable t2) {
                            // Skip if splitting fails
                        }
                    }
                }
            }
        } catch (Throwable t) {
            // Best effort - continue if splitting fails
        }
    }

    private GeometryPart subtractParts(Simulation simulation, CompositePart compositePart) {
        try {
            CadPart mainPart = (CadPart) compositePart.getChildParts().getPart("main");
            SubtractPartsOperation subtractOperation = (SubtractPartsOperation) simulation.get(MeshOperationManager.class).createSubtractPartsOperation(new NeoObjectVector(new Object[] {compositePart}));
            subtractOperation.getTargetPartManager().setObjects(mainPart);
            subtractOperation.execute();
            GeometryPart result = simulation.get(SimulationPartManager.class).getPart("Subtract");
            if (result != null) return result;
            return null;
        } catch (Throwable t) {
            simulation.println("Warning: Subtract operation failed: " + t.getMessage() + ". Will use composite part instead.");
            // If subtract fails, return null and handle in calling method
            return null;
        }
    }

    private void prepareFor2dMeshing(Simulation simulation, GeometryPart part) {
        PrepareFor2dOperation prepareFor2dOperation = (PrepareFor2dOperation) simulation.get(MeshOperationManager.class).createPrepareFor2dOperation(new NeoObjectVector(new Object[] {part}));
        prepareFor2dOperation.execute();
    }

    private void createRegionAndBoundaries(Simulation simulation, GeometryPart part) {
        Region region = simulation.getRegionManager().createEmptyRegion();
        region.setPresentationName("Region");
        region.getBoundaryManager().removeObjects(region.getBoundaryManager().getBoundary("Default"));
        simulation.getRegionManager().newRegionsFromParts(new NeoObjectVector(new Object[] {part}), "OneRegion", region, "OneBoundaryPerPartSurface", null, RegionManager.CreateInterfaceMode.BOUNDARY, "OneEdgeBoundaryPerPart", null);
    }

    private void setBoundaryTypes(Simulation simulation) {
        Region region = simulation.getRegionManager().getRegion("Region");
        InletBoundary inletBoundaryType = (InletBoundary) simulation.get(ConditionTypeManager.class).get(InletBoundary.class);
        PressureBoundary pressureBoundaryType = (PressureBoundary) simulation.get(ConditionTypeManager.class).get(PressureBoundary.class);

        for (Boundary boundary : region.getBoundaryManager().getBoundaries()) {
            if (boundary.getPresentationName().toLowerCase().contains("inlet")) {
                boundary.setBoundaryType(inletBoundaryType);
            } else if (boundary.getPresentationName().toLowerCase().contains("outlet")) {
                boundary.setBoundaryType(pressureBoundaryType);
            }
        }
    }
  private void setupPhysics(Simulation simulation) {
	Units units_0 = 
	((Units) simulation.getUnitsManager().getObject("Pa"));
        PhysicsContinuum physicsContinuum = simulation.getContinuumManager().createContinuum(PhysicsContinuum.class);

        physicsContinuum.enable(TwoDimensionalModel.class);
        physicsContinuum.enable(SteadyModel.class);
        physicsContinuum.enable(SingleComponentLiquidModel.class);
        physicsContinuum.enable(CoupledFlowModel.class);
        physicsContinuum.enable(ConstantDensityModel.class);
  // Enable turbulence/laminar model based on config
        enableSelectedTurbulenceModel(simulation, physicsContinuum);
        physicsContinuum.enable(AdjointModel.class);
        physicsContinuum.enable(TopologyOptimizationModel.class);
        physicsContinuum.enable(TopologyPhysicsModel.class);
        physicsContinuum.enable(AdjointFlowModel.class);
    	physicsContinuum.getReferenceValues().get(MinimumAllowableAbsolutePressure.class).setValueAndUnits(-1000.0, units_0);
    	physicsContinuum.getReferenceValues().get(ReferencePressure.class).setValueAndUnits(0.0, units_0);

    	SingleComponentLiquidModel singleComponentLiquidModel_0 = 
      	physicsContinuum.getModelManager().getModel(SingleComponentLiquidModel.class);

    	Liquid liquid_0 = 
      	((Liquid) singleComponentLiquidModel_0.getMaterial());

    	ConstantMaterialPropertyMethod constantMaterialPropertyMethod_0 = 
      	((ConstantMaterialPropertyMethod) liquid_0.getMaterialProperties().getMaterialProperty(ConstantDensityProperty.class).getMethod());

    	Units units_2 = 
      	((Units) simulation.getUnitsManager().getObject("kg/m^3"));

    	constantMaterialPropertyMethod_0.getQuantity().setValueAndUnits(densityValue, units_2);

   	ConstantMaterialPropertyMethod constantMaterialPropertyMethod_1 = 
      	((ConstantMaterialPropertyMethod) liquid_0.getMaterialProperties().getMaterialProperty(DynamicViscosityProperty.class).getMethod());

	Units units_3 = 
	((Units) simulation.getUnitsManager().getObject("Pa-s"));

  constantMaterialPropertyMethod_1.getQuantity().setValueAndUnits(viscosityValue, units_3);

    // Apply wall treatment if turbulence is selected
    if (!"laminar".equalsIgnoreCase(turbulenceModel)) {
      enableSelectedWallTreatment(simulation, physicsContinuum);
    }
    }

    // Helper to enable turbulence model based on configuration
    private void enableSelectedTurbulenceModel(Simulation simulation, PhysicsContinuum physicsContinuum) {
      String choice = (turbulenceModel == null) ? "laminar" : turbulenceModel.toLowerCase(Locale.ROOT);
      if ("laminar".equals(choice)) {
        physicsContinuum.enable(LaminarModel.class);
        simulation.println("Turbulence: LaminarModel enabled");
        return;
      }

      // Prefer disabling laminar if present
      try {
        LaminarModel lm = physicsContinuum.getModelManager().getModel(LaminarModel.class);
        if (lm != null) physicsContinuum.disableModel(lm);
      } catch (Throwable ignoreLm) { /* best-effort */ }

      // Enable base turbulence stacks used by recorded macro
      tryEnableClass(simulation, physicsContinuum, "star.turbulence.TurbulentModel");
      tryEnableClass(simulation, physicsContinuum, "star.turbulence.RansTurbulenceModel");
      // Optional frozen turbulence for adjoint compatibility (best-effort)
      tryEnableClass(simulation, physicsContinuum, "star.turbulence.AdjointFrozenTurbulenceModel");

      boolean enabled = false;
      if (choice.contains("eps")) {
        // k-epsilon family based on recorded macro classes
        // If user wants two-layer wall treatment, prefer enabling the two-layer k-epsilon model first
        String[] kEps = (wallTreatment != null && wallTreatment.toLowerCase(Locale.ROOT).startsWith("two"))
          ? new String[] { "star.keturb.RkeTwoLayerTurbModel", "star.keturb.KEpsilonTurbulence" }
          : new String[] { "star.keturb.KEpsilonTurbulence", "star.keturb.RkeTwoLayerTurbModel" };
        enabled = tryEnableAny(simulation, physicsContinuum, kEps, "k-epsilon");
      } else {
        // treat as k-omega/SST family (per recorded macro)
        String[] kOmega = new String[] {
          "star.kwturb.KOmegaTurbulence",
          "star.kwturb.SstKwTurbModel"
        };
        enabled = tryEnableAny(simulation, physicsContinuum, kOmega, "k-omega");
        // Optional transition model
        tryEnableClass(simulation, physicsContinuum, "star.turbulence.GammaTransitionModel");
      }

      if (!enabled) {
        simulation.println("Warning: Could not enable requested turbulence model ('" + choice + "'). Falling back to LaminarModel.");
        physicsContinuum.enable(LaminarModel.class);
      }
    }

    private void tryEnableClass(Simulation simulation, PhysicsContinuum pc, String fqcn) {
      try {
        Class<?> cls = Class.forName(fqcn);
        //noinspection unchecked
        pc.enable((Class<? extends Model>) cls);
      } catch (Throwable ignore) { /* best-effort */ }
    }

    private boolean tryEnableAny(Simulation simulation, PhysicsContinuum pc, String[] classNames, String label) {
      for (String name : classNames) {
        try {
          Class<?> cls = Class.forName(name);
          //noinspection unchecked
          pc.enable((Class<? extends Model>) cls);
          simulation.println("Turbulence: enabled " + name + (label != null ? " (" + label + ")" : ""));
          return true;
        } catch (Throwable ignored) {
          // try next
        }
      }
      return false;
    }

    // Helper to enable wall treatment based on configuration
    private void enableSelectedWallTreatment(Simulation simulation, PhysicsContinuum physicsContinuum) {
      String wt = (wallTreatment == null) ? "all-y+" : wallTreatment.toLowerCase(Locale.ROOT);

      // Candidate class names across versions
      if (wt.startsWith("two")) {
        // Ensure two-layer k-epsilon turbulence model is available when using k-epsilon
        if (turbulenceModel != null && turbulenceModel.toLowerCase(Locale.ROOT).contains("eps")) {
          tryEnableClass(simulation, physicsContinuum, "star.keturb.RkeTwoLayerTurbModel");
        }
        String[] candidates = new String[] {
          // Recorded macro k-epsilon two-layer all-y+ treatment
          "star.keturb.KeTwoLayerAllYplusWallTreatment",
          "star.flow.TwoLayerAllYPlusWallTreatment",
          "star.flow.TwoLayerAllYplusWallTreatment",
          "star.flow.TwoLayerAllYPlusWT",
          "star.flow.TwoLayerAllyPlusWallTreatment"
        };
        if (!tryEnableAny(simulation, physicsContinuum, candidates, "two-layer wall treatment")) {
          simulation.println("Warning: Two-layer wall treatment not found; trying All y+.");
          enableAllYPlus(simulation, physicsContinuum);
        }
      } else if (wt.startsWith("low")) {
        String[] candidates = new String[] {
          "star.flow.LowYPlusWallTreatment",
          "star.flow.LowReynoldsNumberWallTreatment",
          "star.flow.LowYplusWallTreatment"
        };
        if (!tryEnableAny(simulation, physicsContinuum, candidates, "low-y+ wall treatment")) {
          simulation.println("Warning: Low y+ wall treatment not found; trying All y+.");
          enableAllYPlus(simulation, physicsContinuum);
        }
      } else {
        enableAllYPlus(simulation, physicsContinuum);
      }
    }

    private void enableAllYPlus(Simulation simulation, PhysicsContinuum physicsContinuum) {
      String[] candidates = new String[] {
        // k-epsilon variant from recorded macro
        "star.keturb.KeTwoLayerAllYplusWallTreatment",
        // k-omega variant from recorded macro
        "star.kwturb.KwAllYplusWallTreatment",
        // legacy/flow package fallbacks
        "star.flow.AllYPlusWallTreatment",
        "star.flow.AllYplusWallTreatment",
        "star.flow.AllYPlusWT"
      };
      if (!tryEnableAny(simulation, physicsContinuum, candidates, "all-y+ wall treatment")) {
        simulation.println("Warning: Could not enable any All y+ wall treatment variant.");
      }
    }

    private void setupPressureDropReport(Simulation simulation) {
      pressureDropReport = simulation.getReportManager().create("star.energy.PressureDropReport");
        
        Region region = simulation.getRegionManager().getRegion("Region");
        
        // Set high pressure parts (inlets)
        pressureDropReport.getParts().setQuery(null);
        List<Boundary> inletBoundaries = new ArrayList<>();
        for (Boundary boundary : region.getBoundaryManager().getBoundaries()) {
            if (boundary.getPresentationName().toLowerCase().contains("inlet")) {
                inletBoundaries.add(boundary);
            }
        }
        pressureDropReport.getParts().setObjects(inletBoundaries.toArray(new Boundary[0]));

        // Set low pressure parts (outlets)
        pressureDropReport.getLowPressureParts().setQuery(null);
        List<Boundary> outletBoundaries = new ArrayList<>();
        for (Boundary boundary : region.getBoundaryManager().getBoundaries()) {
            if (boundary.getPresentationName().toLowerCase().contains("outlet")) {
                outletBoundaries.add(boundary);
            }
        }
        pressureDropReport.getLowPressureParts().setObjects(outletBoundaries.toArray(new Boundary[0]));

	FvRepresentation fvRepresentation = 
	      ((FvRepresentation) simulation.getRepresentationManager().getObject("Volume Mesh"));

    	pressureDropReport.setRepresentation(fvRepresentation);

	    OversetVolumeIntegralReport oversetVolumeIntegralReport_0 = 
	      simulation.getReportManager().create("star.base.report.OversetVolumeIntegralReport");

	    ExpressionReport expressionReport_0 = 
	      simulation.getReportManager().create("star.base.report.ExpressionReport");

	    expressionReport_0.printReport();

	    PrimitiveFieldFunction primitiveFieldFunction_0 = 
	      ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("MaterialIndicator"));

	    oversetVolumeIntegralReport_0.setFieldFunction(primitiveFieldFunction_0);

	    oversetVolumeIntegralReport_0.getParts().setQuery(null);

	    Region region_0 = 
	      simulation.getRegionManager().getRegion("Region");

	    oversetVolumeIntegralReport_0.getParts().setObjects(region_0);

	    Units units_0 = 
	      simulation.getUnitsManager().getPreferredUnits(Dimensions.Builder().pressure(1).build());

	    Units units_1 = 
	      simulation.getUnitsManager().getPreferredUnits(Dimensions.Builder().length(3).build());

        // Create or update a global parameter for the volume loss weight
        ScalarGlobalParameter weightParam;
        Object existing = null;
        try { existing = simulation.get(GlobalParameterManager.class).getObject("volume_loss_weight"); } catch (Exception ignore) {}
        if (existing instanceof ScalarGlobalParameter) {
          weightParam = (ScalarGlobalParameter) existing;
        } else {
          weightParam = simulation.get(GlobalParameterManager.class)
            .createGlobalParameter(ScalarGlobalParameter.class, "volume_loss_weight");
        }

        Units unitless = (Units) simulation.getUnitsManager().getObject("");
        weightParam.getQuantity().setValueAndUnits(volumeLossWeight, unitless);

	    expressionReport_0.setDefinition("${PressureDrop1Report} + ${volume_loss_weight}*${OversetVolumeIntegral1Report}");

        // Ensure snapshot parameter for event-driven monitor exists
        ScalarGlobalParameter pdSnapshotParam;
        Object existingPdSnap = null;
        try { existingPdSnap = simulation.get(GlobalParameterManager.class).getObject("pressure_drop_snapshot"); } catch (Exception ignore) {}
        if (existingPdSnap instanceof ScalarGlobalParameter) {
          pdSnapshotParam = (ScalarGlobalParameter) existingPdSnap;
        } else {
          pdSnapshotParam = simulation.get(GlobalParameterManager.class)
            .createGlobalParameter(ScalarGlobalParameter.class, "pressure_drop_snapshot");
          Units pUnits = simulation.getUnitsManager().getPreferredUnits(Dimensions.Builder().pressure(1).build());
          pdSnapshotParam.getQuantity().setValueAndUnits(0.0, pUnits);
        }

        // Create a snapshot expression report that references the snapshot parameter
        ExpressionReport pdSnapshotReport = simulation.getReportManager().createReport(ExpressionReport.class);
        pdSnapshotReport.setPresentationName("Pressure Drop Snapshot");
        pdSnapshotReport.setDefinition("${pressure_drop_snapshot}");

        // Create monitor and plot for the snapshot (ticks only when parameter is updated in workflow)
        try {
          simulation.getMonitorManager().createMonitorAndPlot(new ArrayList<>(Arrays.<Report>asList(pdSnapshotReport)), true, "%1$s Plot");
        } catch (Throwable t) {
          simulation.println("Warning: Failed to create monitor/plot for Pressure Drop Snapshot: " + t.getMessage());
        }

    }

  private void setupMaterialIndicatorReports(Simulation simulation) {
    Region region = simulation.getRegionManager().getRegion("Region");
    FvRepresentation fvRepresentation = (FvRepresentation) simulation.getRepresentationManager().getObject("Volume Mesh");

    PrimitiveFieldFunction matIndFn = (PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("MaterialIndicator");

    // Create integral report (avoid name-based retrieval to prevent manager lookup issues)
    matIndIntegralReport = simulation.getReportManager().createReport(VolumeIntegralReport.class);
    matIndIntegralReport.setFieldFunction(matIndFn);
    matIndIntegralReport.getParts().setObjects(region);
    matIndIntegralReport.setRepresentation(fvRepresentation);

    // Create or reuse constant-one user field function for geometric volume via integral
    FieldFunction oneFF = simulation.getFieldFunctionManager().getFunction("OneFF");
    if (oneFF == null || !(oneFF instanceof UserFieldFunction)) {
      UserFieldFunction uff = simulation.getFieldFunctionManager().createFieldFunction();
      uff.setPresentationName("OneFF");
      uff.setFunctionName("OneFF");
      uff.setDefinition("1");
      uff.setDimensions(Dimensions.Builder().build());
      oneFF = uff;
    }

    // Create region volume integral report (integral of 1 over region)
    regionVolumeIntegralReport = simulation.getReportManager().createReport(VolumeIntegralReport.class);
    regionVolumeIntegralReport.setFieldFunction(oneFF);
    regionVolumeIntegralReport.getParts().setObjects(region);
    regionVolumeIntegralReport.setRepresentation(fvRepresentation);

    // Ensure snapshot parameter for event-driven volume monitor exists
    ScalarGlobalParameter volSnapshotParam;
    Object existingVolSnap = null;
    try { existingVolSnap = simulation.get(GlobalParameterManager.class).getObject("volume_integral_snapshot"); } catch (Exception ignore) {}
    if (existingVolSnap instanceof ScalarGlobalParameter) {
      volSnapshotParam = (ScalarGlobalParameter) existingVolSnap;
    } else {
      volSnapshotParam = simulation.get(GlobalParameterManager.class)
        .createGlobalParameter(ScalarGlobalParameter.class, "volume_integral_snapshot");
      Units vUnits = simulation.getUnitsManager().getPreferredUnits(Dimensions.Builder().length(3).build());
      volSnapshotParam.getQuantity().setValueAndUnits(0.0, vUnits);
    }

    // Create snapshot expression report referencing the snapshot parameter
    ExpressionReport volSnapshotReport = simulation.getReportManager().createReport(ExpressionReport.class);
    volSnapshotReport.setPresentationName("Volume Integral Snapshot");
    volSnapshotReport.setDefinition("${volume_integral_snapshot}");

    // Create monitor and plot for the snapshot (ticks only when parameter is updated in workflow)
    try {
      simulation.getMonitorManager().createMonitorAndPlot(new ArrayList<>(Arrays.<Report>asList(volSnapshotReport)), true, "%1$s Plot");
    } catch (Throwable t) {
      simulation.println("Warning: Failed to create monitor/plot for Volume Integral Snapshot: " + t.getMessage());
    }
  }

    private void setupTopologyOptimization(Simulation simulation) {
        PhysicsContinuum physicsContinuum = (PhysicsContinuum) simulation.getContinuumManager().getContinuum("Physics 1");
        Region region = simulation.getRegionManager().getRegion("Region");

        // Create and set up the cost function
        ExpressionReport pressureDropReport = (ExpressionReport) simulation.getReportManager().getReport("Expression 1");
        ReportCostFunction costFunction = simulation.get(AdjointCostFunctionManager.class).createAdjointCostFunction(ReportCostFunction.class);
        costFunction.setReport(pressureDropReport);

        // Set up the topology optimization solver
        TopologyOptimizationSolver topologySolver = simulation.getSolverManager().getSolver(TopologyOptimizationSolver.class);
        topologySolver.setAdjointCostFunction(costFunction);

        // Set up the topology physics model
        TopologyPhysicsModel topologyPhysicsModel = physicsContinuum.getModelManager().getModel(TopologyPhysicsModel.class);
	@SuppressWarnings("unchecked")
        TopologyPhase solidPhase = (TopologyPhase) topologyPhysicsModel.getPhaseManager().getPhase("Solid Phase");

        solidPhase.setRegions(new NeoObjectVector(new Object[] {region}));

    // Apply Brinkman penalty per recorder API: getBrinkmanPenalization() with units [kg/m^3-s]; fallback to legacy penalty APIs
    try {
      Units kg_per_m3_s = (Units) simulation.getUnitsManager().getObject("kg/m^3-s");
      Class<?> tpCls = topologyPhysicsModel.getClass();
      // Preferred per commandsRecord.java
      try {
        Method mGetPenalization = tpCls.getMethod("getBrinkmanPenalization");
        Object penalization = mGetPenalization.invoke(topologyPhysicsModel);
        Method mSet = penalization.getClass().getMethod("setValueAndUnits", double.class, Units.class);
        mSet.invoke(penalization, brinkmanPenalty, kg_per_m3_s);
        simulation.println(String.format("Brinkman penalization set to %.3e [kg/m^3-s]", brinkmanPenalty));
      } catch (Throwable preferFail) {
        // Fallbacks: legacy methods using [1/m^2]
        try {
          Units invM2 = simulation.getUnitsManager().getPreferredUnits(Dimensions.Builder().length(-2).build());
          Object penaltyHolder = null;
          try {
            Method mGetSettings = tpCls.getMethod("getBrinkmanSettings");
            Object settings = mGetSettings.invoke(topologyPhysicsModel);
            Method mGetPenalty = settings.getClass().getMethod("getBrinkmanPenalty");
            penaltyHolder = mGetPenalty.invoke(settings);
          } catch (Throwable ignore) {
            try {
              Method mGetPenalty = tpCls.getMethod("getBrinkmanPenalty");
              penaltyHolder = mGetPenalty.invoke(topologyPhysicsModel);
            } catch (Throwable ignore2) {
              penaltyHolder = null;
            }
          }
          if (penaltyHolder != null) {
            try {
              Method mSet = penaltyHolder.getClass().getMethod("setValueAndUnits", double.class, Units.class);
              mSet.invoke(penaltyHolder, brinkmanPenalty, invM2);
              simulation.println(String.format("Brinkman penalty set to %.3e [1/m^2] (fallback API)", brinkmanPenalty));
            } catch (Throwable t3) {
              simulation.println("Warning: Found Brinkman penalty object but could not set value: " + t3.getMessage());
            }
          } else {
            simulation.println("Warning: Could not locate Brinkman penalty/penalization API on TopologyPhysicsModel.");
          }
        } catch (Throwable t2) {
          simulation.println("Warning: Failed to set Brinkman penalty via fallback: " + t2.getMessage());
        }
      }
    } catch (Throwable t) {
      simulation.println("Warning: Failed to set Brinkman penalization: " + t.getMessage());
    }


    }
    private void setupMeshing(Simulation simulation, GeometryPart subtractedPart) {
        //AutoMeshOperation2d autoMeshOperation2d = 
        //    simulation.get(MeshOperationManager.class).createAutoMeshOperation2d(
        //        new StringVector(new String[] {"star.twodmesher.QuadAutoMesher2d", "star.prismmesher.PrismAutoMesher"}),
        //        new NeoObjectVector(new Object[] {subtractedPart}));

        AutoMeshOperation2d autoMeshOperation2d = 
            simulation.get(MeshOperationManager.class).createAutoMeshOperation2d(
                new StringVector(new String[] {"star.twodmesher.QuadAutoMesher2d"}),
                new NeoObjectVector(new Object[] {subtractedPart}));
        Units meters = (Units) simulation.getUnitsManager().getObject("m");

        // Set base size
        //autoMeshOperation2d.getDefaultValues().get(BaseSize.class).setValueAndUnits(0.001, meters);
        autoMeshOperation2d.getDefaultValues().get(BaseSize.class).setValueAndUnits(0.001, meters);

        // Set number of prism layers
        //NumPrismLayers numPrismLayers = autoMeshOperation2d.getDefaultValues().get(NumPrismLayers.class);
        //IntegerValue integerValue = numPrismLayers.getNumLayersValue();
        //integerValue.getQuantity().setValue(5.0);

	SurfaceGrowthRate surfaceGrowthRate_0 = autoMeshOperation2d.getDefaultValues().get(SurfaceGrowthRate.class);

	surfaceGrowthRate_0.setGrowthRateOption(SurfaceGrowthRate.GrowthRateOption.USER_SPECIFIED);

	Units units_4 = 
	((Units) simulation.getUnitsManager().getObject(""));

	surfaceGrowthRate_0.getGrowthRateScalar().setValueAndUnits(1.001, units_4);


        // Execute the mesh operation
        autoMeshOperation2d.execute();
    }
    private void setupTopologyOptimizationAutomation(Simulation simulation, int maxIter) {
        // Create global parameter for iterations
        ScalarGlobalParameter iterationsParameter = simulation.get(GlobalParameterManager.class)
            .createGlobalParameter(ScalarGlobalParameter.class, "IterationsPrimal");
        iterationsParameter.getQuantity().setValueAndUnits(initialPrimalIterations, simulation.getUnitsManager().getObject(""));

        ScalarGlobalParameter counterParameter = simulation.get(GlobalParameterManager.class)
            .createGlobalParameter(ScalarGlobalParameter.class, "counter");
    counterParameter.getQuantity().setValueAndUnits(0.0, simulation.getUnitsManager().getObject(""));

        // Setup topology optimization model
        PhysicsContinuum physicsContinuum = (PhysicsContinuum) simulation.getContinuumManager().getContinuum("Physics 1");
        TopologyOptimizationModel topModel = physicsContinuum.getModelManager().getModel(TopologyOptimizationModel.class);
        topModel.setAllowHoles(true);
        topModel.getSourceSettings().getSourceStrength().setValueAndUnits(200.0, simulation.getUnitsManager().getObject(""));

        // Setup stopping criteria
        simulation.getSolverStoppingCriterionManager().getSolverStoppingCriterion("Maximum Steps").setIsUsed(false);
        FixedStepsStoppingCriterion topStoppingCriterion = simulation.getSolverStoppingCriterionManager().create("star.common.FixedStepsStoppingCriterion");

        topStoppingCriterion.getFixedStepsObject().getQuantity().setValue(200.0);

        SteadySolver steadySolver = simulation.getSolverManager().getSolver(SteadySolver.class);
        FixedStepsStoppingCriterion steadyStoppingCriterion = steadySolver.getSolverStoppingCriterionManager()
            .create("star.common.FixedStepsStoppingCriterion");
        steadyStoppingCriterion.getFixedStepsObject().getQuantity().setDefinition("${IterationsPrimal}");

        AdjointSolver adjointSolver = simulation.getSolverManager().getSolver(AdjointSolver.class);
        AdjointSteadySolver adjointSteadySolver = adjointSolver.getAdjointSolverManager().getSolver(AdjointSteadySolver.class);
        FixedStepsStoppingCriterion adjointStoppingCriterion = adjointSteadySolver.getSolverStoppingCriterionManager()
            .create("star.common.FixedStepsStoppingCriterion");
        adjointStoppingCriterion.getFixedStepsObject().getQuantity().setValue(200.0);

        // Create automation workflow
        topoWorkflow = simulation.get(SimDriverWorkflowManager.class).createSimDriverWorkflow("Topology Optimization");
        SimDriverWorkflow workflow = topoWorkflow;
        
        workflow.getBlocks().createBlock("star.automation.ClearSolutionAutomationBlock", "Clear Solution");
        ClearSolutionAutomationBlock clearSolutionBlock = (ClearSolutionAutomationBlock) workflow.getBlocks().getObject("Clear Solution");
        clearSolutionBlock.setResetMesh(true);
        clearSolutionBlock.setClearAdjointFlow(true);

        workflow.getBlocks().createBlock("star.automation.InitializeSolutionAutomationBlock", "Initialize Solution");
        
        workflow.getBlocks().createBlock("star.common.SetParameterAutomationBlock", "Set Initial Iterations");
        SetParameterAutomationBlock setInitialIterationsBlock = (SetParameterAutomationBlock) workflow.getBlocks().getObject("Set Initial Iterations");
        setInitialIterationsBlock.setParameter(iterationsParameter);
        setInitialIterationsBlock.getScalarQuantity().setValueAndUnits(initialPrimalIterations, simulation.getUnitsManager().getObject(""));

        workflow.getBlocks().createBlock("star.common.SetParameterAutomationBlock", "Set Initial Counter");
        SetParameterAutomationBlock setInitialIterationsBlock_ = (SetParameterAutomationBlock) workflow.getBlocks().getObject("Set Initial Counter");
        setInitialIterationsBlock_.setParameter(counterParameter);
        setInitialIterationsBlock_.getScalarQuantity().setValueAndUnits(0.0, simulation.getUnitsManager().getObject(""));

        workflow.getBlocks().createBlock("star.common.SolvePhysics", "Solve Initial Physics");
        SolvePhysics solveInitialPhysicsBlock = (SolvePhysics) workflow.getBlocks().getObject("Solve Initial Physics");
        solveInitialPhysicsBlock.getSimulationObjects().setObjects(physicsContinuum);

        workflow.getBlocks().createBlock("star.common.SetParameterAutomationBlock", "Set Optimization Iterations");
        SetParameterAutomationBlock setOptimizationIterationsBlock = (SetParameterAutomationBlock) workflow.getBlocks().getObject("Set Optimization Iterations");
        setOptimizationIterationsBlock.setParameter(iterationsParameter);
        setOptimizationIterationsBlock.getScalarQuantity().setValueAndUnits(optimizationPrimalIterations, simulation.getUnitsManager().getObject(""));

        workflow.getBlocks().createBlock("star.automation.LoopAutomationBlock", "Optimization Loop");
        LoopAutomationBlock loopBlock = (LoopAutomationBlock) workflow.getBlocks().getObject("Optimization Loop");
        AutomationScalarExpressionPredicate loopPredicate = (AutomationScalarExpressionPredicate) loopBlock.getAutomationPredicateManager().getObject("Expression Predicate");


	loopBlock.setSelectedPredicate(loopPredicate);

	Units units_0 = 
	simulation.getUnitsManager().getInternalUnits(new IntVector(new int[] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

	loopPredicate.getQuantity().setDefinition("${counter} < " + maxIter);

        //loopPredicate.setExpressionPredicate("${counter} < 50");

        // Set IterationsPrimal per iteration: small on first pass, normal afterwards
        loopBlock.getBlocks().createBlock("star.common.SetParameterAutomationBlock", "Set Loop Iterations");
        SetParameterAutomationBlock setLoopIterationsBlock = (SetParameterAutomationBlock) loopBlock.getBlocks().getObject("Set Loop Iterations");
        setLoopIterationsBlock.setParameter(iterationsParameter);
        setLoopIterationsBlock.getScalarQuantity().setDefinition(String.format(Locale.US, "${counter} < 0.5 ? %.1f : %.1f", firstIterationPrimalIterations, optimizationPrimalIterations));

        loopBlock.getBlocks().createBlock("star.common.SolvePhysics", "Solve Physics");
        SolvePhysics solvePhysicsBlock = (SolvePhysics) loopBlock.getBlocks().getObject("Solve Physics");
        solvePhysicsBlock.getSimulationObjects().setObjects(physicsContinuum);

        // Event: Sample Pressure Drop after Solve Physics
        loopBlock.getBlocks().createBlock("star.common.SetParameterAutomationBlock", "Sample Pressure Drop");
        SetParameterAutomationBlock samplePD = (SetParameterAutomationBlock) loopBlock.getBlocks().getObject("Sample Pressure Drop");
        ScalarGlobalParameter pdSnapshotParam = (ScalarGlobalParameter) simulation.get(GlobalParameterManager.class).getObject("pressure_drop_snapshot");
        if (pdSnapshotParam == null) {
          pdSnapshotParam = simulation.get(GlobalParameterManager.class)
            .createGlobalParameter(ScalarGlobalParameter.class, "pressure_drop_snapshot");
          Units pUnits = simulation.getUnitsManager().getPreferredUnits(Dimensions.Builder().pressure(1).build());
          pdSnapshotParam.getQuantity().setValueAndUnits(0.0, pUnits);
        }
        samplePD.setParameter(pdSnapshotParam);
        samplePD.getScalarQuantity().setDefinition("${PressureDrop1Report}");
	
	//loopBlock.getBlocks().createBlock("star.automation.StopWorkflowAutomationBlock", "Stop Simulation Operations");


        loopBlock.getBlocks().createBlock("star.common.SolveAdjoint", "Solve Adjoint");
        SolveAdjoint solveAdjointBlock = (SolveAdjoint) loopBlock.getBlocks().getObject("Solve Adjoint");
        solveAdjointBlock.getAdjointCostFunctions().setObjects(simulation.get(AdjointCostFunctionManager.class).getAdjointCostFunction("Report"));
        solveAdjointBlock.setComputeParameterSensitivity(true);

        // Event: Sample Volume Integral after Solve Adjoint
        loopBlock.getBlocks().createBlock("star.common.SetParameterAutomationBlock", "Sample Volume Integral");
        SetParameterAutomationBlock sampleVol = (SetParameterAutomationBlock) loopBlock.getBlocks().getObject("Sample Volume Integral");
        ScalarGlobalParameter volSnapshotParam = (ScalarGlobalParameter) simulation.get(GlobalParameterManager.class).getObject("volume_integral_snapshot");
        if (volSnapshotParam == null) {
          volSnapshotParam = simulation.get(GlobalParameterManager.class)
            .createGlobalParameter(ScalarGlobalParameter.class, "volume_integral_snapshot");
          Units vUnits = simulation.getUnitsManager().getPreferredUnits(Dimensions.Builder().length(3).build());
          volSnapshotParam.getQuantity().setValueAndUnits(0.0, vUnits);
        }
        sampleVol.setParameter(volSnapshotParam);
        sampleVol.getScalarQuantity().setDefinition("${OversetVolumeIntegral1Report}");

        // Re-solve primal flow after adjoint/topology update so reports reflect updated material indicator
        loopBlock.getBlocks().createBlock("star.common.SolvePhysics", "Solve final Physics");
        SolvePhysics solveFinalPhysicsBlockLoop = (SolvePhysics) loopBlock.getBlocks().getObject("Solve final Physics");
        solveFinalPhysicsBlockLoop.getSimulationObjects().setObjects(physicsContinuum);

        loopBlock.getBlocks().createBlock("star.common.SetParameterAutomationBlock", "Increase Counter");
        SetParameterAutomationBlock increaseIterationsBlock = (SetParameterAutomationBlock) loopBlock.getBlocks().getObject("Increase Counter");
        increaseIterationsBlock.setParameter(counterParameter);
        increaseIterationsBlock.getScalarQuantity().setDefinition("${counter} +1");
        simulation.get(SimDriverWorkflowManager.class).setSelectedWorkflow(workflow);
    }

    private SimDriverWorkflow getTopologyWorkflow(Simulation simulation) {
      if (topoWorkflow != null) return topoWorkflow;
      SimDriverWorkflowManager mgr = simulation.get(SimDriverWorkflowManager.class);
      SimDriverWorkflow wf = (SimDriverWorkflow) mgr.getObject("Topology Optimization");
      if (wf == null) {
        wf = (SimDriverWorkflow) mgr.getObject("Topology Optimization 1");
      }
      topoWorkflow = wf;
      return wf;
    }

    private void runTopologyOptimization(Simulation simulation) {

  SimDriverWorkflow workflow = getTopologyWorkflow(simulation);
  if (workflow == null) {
    simulation.println("Warning: Topology Optimization workflow not found; skipping run.");
    return;
  }

  LoopAutomationBlock loopBlock = (LoopAutomationBlock) workflow.getBlocks().getObject("Optimization Loop");
	//SolvePhysics solvePhysicsBlock = (SolvePhysics) loopBlock.getBlocks().getObject("Solve Physics");
        SolveAdjoint solveAdjointBlock = (SolveAdjoint) loopBlock.getBlocks().getObject("Solve Adjoint");
        workflow.playTo(solveAdjointBlock);

        // After adjoint/topology update, re-solve primal physics within the loop so reports reflect the updated material indicator
        SolvePhysics solveFinalPhysicsBlock = (SolvePhysics) loopBlock.getBlocks().getObject("Solve final Physics");
        if (solveFinalPhysicsBlock != null) {
          workflow.playTo(solveFinalPhysicsBlock);
        } else {
          simulation.println("Warning: 'Solve final Physics' block not found in loop; post-adjoint physics solve skipped.");
        }
    }

    // Perform warm-up Solve Physics cycles before the main export loop to reduce initial transients
    private void runWarmupPhysics(Simulation simulation, int cycles, double warmupIterations, double restoreOptimizationIterations) {
      SimDriverWorkflow workflow = getTopologyWorkflow(simulation);
      if (workflow == null) {
        simulation.println("Warning: Topology Optimization workflow not found; skipping warm-up.");
        return;
      }

      // Locate the loop Solve Physics block
      LoopAutomationBlock loopBlock = (LoopAutomationBlock) workflow.getBlocks().getObject("Optimization Loop");
      if (loopBlock == null) {
        simulation.println("Warning: Optimization Loop not found; skipping warm-up.");
        return;
      }
      SolvePhysics solvePhysicsBlock = (SolvePhysics) loopBlock.getBlocks().getObject("Solve Physics");
      if (solvePhysicsBlock == null) {
        simulation.println("Warning: Solve Physics block not found; skipping warm-up.");
        return;
      }

      // Temporarily set IterationsPrimal to warm-up value
      ScalarGlobalParameter iterationsParam = null;
      try {
        Object obj = simulation.get(GlobalParameterManager.class).getObject("IterationsPrimal");
        if (obj instanceof ScalarGlobalParameter) {
          iterationsParam = (ScalarGlobalParameter) obj;
        }
      } catch (Exception ignore) { /* no-op */ }

      Double originalVal = null;
      Units unitless = (Units) simulation.getUnitsManager().getObject("");
      if (iterationsParam != null) {
        try {
          originalVal = iterationsParam.getQuantity().getRawValue();
          iterationsParam.getQuantity().setValueAndUnits(warmupIterations, unitless);
        } catch (Exception e) {
          simulation.println("Warning: Failed to set warm-up iterations: " + e.getMessage());
        }
      }

      for (int c = 0; c < cycles; c++) {
        workflow.playTo(solvePhysicsBlock);
        simulation.println(String.format(Locale.US, "Warm-up cycle %d/%d completed (IterationsPrimal=%.0f)", c + 1, cycles, warmupIterations));
      }

      // Restore optimization iterations if provided
      if (iterationsParam != null) {
        try {
          double restoreVal = (restoreOptimizationIterations > 0) ? restoreOptimizationIterations : (originalVal != null ? originalVal : optimizationPrimalIterations);
          iterationsParam.getQuantity().setValueAndUnits(restoreVal, unitless);
        } catch (Exception e) {
          simulation.println("Warning: Failed to restore optimization iterations: " + e.getMessage());
        }
      }
    }

    private XyzInternalTable setupTable(Simulation simulation) {
	XyzInternalTable xyzInternalTable_0 = 
	simulation.getTableManager().create("star.common.XyzInternalTable");

    PrimitiveFieldFunction primitiveFieldFunction_0 = 
      ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("Adjoint1::MaterialIndicatorSensitivity"));

    PrimitiveFieldFunction primitiveFieldFunction_1 = 
      ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("Adjoint1::CoordSensitivity"));

    VectorMagnitudeFieldFunction vectorMagnitudeFieldFunction_0 = 
      ((VectorMagnitudeFieldFunction) primitiveFieldFunction_1.getMagnitudeFunction());

    VectorComponentFieldFunction vectorComponentFieldFunction_0 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_1.getComponentFunction(0));

    VectorComponentFieldFunction vectorComponentFieldFunction_1 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_1.getComponentFunction(1));

    VectorComponentFieldFunction vectorComponentFieldFunction_2 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_1.getComponentFunction(2));

    PrimitiveFieldFunction primitiveFieldFunction_2 = 
      ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("PressureDrop1Report"));

    PrimitiveFieldFunction primitiveFieldFunction_3 = 
      ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("Velocity"));

    VectorMagnitudeFieldFunction vectorMagnitudeFieldFunction_1 = 
      ((VectorMagnitudeFieldFunction) primitiveFieldFunction_3.getMagnitudeFunction());

    VectorComponentFieldFunction vectorComponentFieldFunction_3 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_3.getComponentFunction(0));

    VectorComponentFieldFunction vectorComponentFieldFunction_4 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_3.getComponentFunction(1));

    VectorComponentFieldFunction vectorComponentFieldFunction_5 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_3.getComponentFunction(2));

    PrimitiveFieldFunction primitiveFieldFunction_4 = 
      ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("MaterialIndicator"));

    PrimitiveFieldFunction primitiveFieldFunction_5 = 
      ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("Pressure"));

    PrimitiveFieldFunction primitiveFieldFunction_6 = 
      ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("WallShearStress"));

    VectorMagnitudeFieldFunction vectorMagnitudeFieldFunction_2 = 
      ((VectorMagnitudeFieldFunction) primitiveFieldFunction_6.getMagnitudeFunction());

    VectorComponentFieldFunction vectorComponentFieldFunction_6 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_6.getComponentFunction(0));

    VectorComponentFieldFunction vectorComponentFieldFunction_7 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_6.getComponentFunction(1));

    VectorComponentFieldFunction vectorComponentFieldFunction_8 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_6.getComponentFunction(2));

    PrimitiveFieldFunction primitiveFieldFunction_7 = 
      ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("TopologyLevelSet"));

    PrimitiveFieldFunction primitiveFieldFunction_8 = 
      ((PrimitiveFieldFunction) simulation.getFieldFunctionManager().getFunction("Position"));

    VectorMagnitudeFieldFunction vectorMagnitudeFieldFunction_3 = 
      ((VectorMagnitudeFieldFunction) primitiveFieldFunction_8.getMagnitudeFunction());

    VectorComponentFieldFunction vectorComponentFieldFunction_9 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_8.getComponentFunction(0));

    VectorComponentFieldFunction vectorComponentFieldFunction_10 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_8.getComponentFunction(1));

    VectorComponentFieldFunction vectorComponentFieldFunction_11 = 
      ((VectorComponentFieldFunction) primitiveFieldFunction_8.getComponentFunction(2));

    xyzInternalTable_0.setFieldFunctions(new NeoObjectVector(new Object[] {primitiveFieldFunction_0, vectorMagnitudeFieldFunction_0, vectorComponentFieldFunction_0, vectorComponentFieldFunction_1, vectorComponentFieldFunction_2, primitiveFieldFunction_2, vectorMagnitudeFieldFunction_1, vectorComponentFieldFunction_3, vectorComponentFieldFunction_4, vectorComponentFieldFunction_5, primitiveFieldFunction_4, primitiveFieldFunction_5, vectorMagnitudeFieldFunction_2, vectorComponentFieldFunction_6, vectorComponentFieldFunction_7, vectorComponentFieldFunction_8, primitiveFieldFunction_7, vectorMagnitudeFieldFunction_3, vectorComponentFieldFunction_9, vectorComponentFieldFunction_10, vectorComponentFieldFunction_11}));


	xyzInternalTable_0.getParts().setQuery(null);

	Region region_0 = 
	simulation.getRegionManager().getRegion("Region");

	xyzInternalTable_0.getParts().setObjects(region_0);
	
	return xyzInternalTable_0;
    }


    private void exportTable(XyzInternalTable xyzInternalTable_0, String savePath) {
	xyzInternalTable_0.extract();

	xyzInternalTable_0.export(savePath, ",");

    }

  private double getScalarReportValue(ScalarReport report) {
    if (report == null) return Double.NaN;
    try {
      report.printReport();
      return report.getValue();
    } catch (Exception e) {
      e.printStackTrace();
      return Double.NaN;
    }
  }

  private double getReportValue(Report report) {
    if (report == null) return Double.NaN;
    try {
      report.printReport();
      try {
        return report.getReportMonitorValue();
      } catch (Exception ignore) {
        // Fall back to ScalarReport if available
        if (report instanceof ScalarReport) {
          return ((ScalarReport) report).getValue();
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
    return Double.NaN;
  }

  // Best-effort setter for topology optimizer step size across versions/APIs
  private void setTopologyStepSize(Simulation simulation, TopologyOptimizationSolver solver, double value) {
    if (solver == null) {
      simulation.println("Warning: TopologyOptimizationSolver not found; cannot set step size.");
      return;
    }

    Units unitless = (Units) simulation.getUnitsManager().getObject("");

    // Prefer the documented API first
    try {
      ScalarPhysicalQuantityInput spi = solver.getStepSizeInput();
      if (spi != null) {
        try {
          spi.getQuantity().setValueAndUnits(value, unitless);
          simulation.println(String.format(Locale.US, "Topology step size set via getStepSizeInput().getQuantity() to %.3g", value));
          return;
        } catch (Throwable ignore) { /* fall through */ }
        try {
          spi.getQuantity().setValue(value);
          simulation.println(String.format(Locale.US, "Topology step size set via getStepSizeInput().getQuantity().setValue to %.3g", value));
          return;
        } catch (Throwable ignore) { /* fall through */ }
      }
    } catch (Throwable ignore) { /* fall back to reflection */ }

    String[] candidateMethods = new String[] {
      "getStepSizeInput", "getStepSize", "getTopologyStepSize", "getRelativeStepSize", "getDesignUpdateStepSize", "getDesignChangeLimiter", "getMaxStepSize"
    };
    for (String mn : candidateMethods) {
      try {
        Method m = solver.getClass().getMethod(mn);
        Object holder = m.invoke(solver);
        if (holder == null) continue;

        // Direct type handling for common quantity holders
        try {
          if (holder instanceof ScalarPhysicalQuantityInput) {
            ScalarPhysicalQuantityInput h = (ScalarPhysicalQuantityInput) holder;
            try {
              h.getQuantity().setValueAndUnits(value, unitless);
              simulation.println(String.format(Locale.US, "Topology step size set via %s.getQuantity().setValueAndUnits to %.3g", mn, value));
              return;
            } catch (Throwable ignore) { }
            try {
              h.getQuantity().setValue(value);
              simulation.println(String.format(Locale.US, "Topology step size set via %s.getQuantity().setValue to %.3g", mn, value));
              return;
            } catch (Throwable ignore) { }
          }
          if (holder instanceof ScalarPhysicalQuantity) {
            ScalarPhysicalQuantity h = (ScalarPhysicalQuantity) holder;
            try {
              h.setValue(value);
              simulation.println(String.format(Locale.US, "Topology step size set via %s.setValue to %.3g", mn, value));
              return;
            } catch (Throwable ignore) { }
            try {
              h.setValueAndUnits(value, unitless);
              simulation.println(String.format(Locale.US, "Topology step size set via %s.setValueAndUnits to %.3g", mn, value));
              return;
            } catch (Throwable ignore) { }
          }
        } catch (Throwable ignore) { /* continue to reflective setters */ }

        // Generic reflective setters
        try {
          Method setVal = holder.getClass().getMethod("setValue", double.class);
          setVal.invoke(holder, value);
          simulation.println(String.format(Locale.US, "Topology step size set via %s.setValue to %.3g", mn, value));
          return;
        } catch (Throwable ignore) { }
        try {
          Method setValUnits = holder.getClass().getMethod("setValueAndUnits", double.class, Units.class);
          setValUnits.invoke(holder, value, unitless);
          simulation.println(String.format(Locale.US, "Topology step size set via %s.setValueAndUnits to %.3g", mn, value));
          return;
        } catch (Throwable ignore) { }
      } catch (Throwable ignoreOuter) { /* try next */ }
    }
    simulation.println("Warning: Could not set topology step size; no compatible API found.");
  }

  // Minimal JSON number parsing without external libraries
  private static Double parseJsonDouble(String json, String key) {
    if (json == null || key == null) return null;
    String regex = "\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*([-+]?(?:\\d*\\.\\d+|\\d+)(?:[eE][-+]?\\d+)?)";
    Matcher m = Pattern.compile(regex).matcher(json);
    if (m.find()) {
      try {
        return Double.parseDouble(m.group(1));
      } catch (Exception ignore) { /* fall through */ }
    }
    return null;
  }

  private static Integer parseJsonInt(String json, String key) {
    Double d = parseJsonDouble(json, key);
    if (d == null) return null;
    try {
      return (int) Math.round(d);
    } catch (Exception ignore) {
      return null;
    }
  }

  // Minimal JSON string parsing without external libraries
  private static String parseJsonString(String json, String key) {
    if (json == null || key == null) return null;
    String regex = "\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"";
    Matcher m = Pattern.compile(regex).matcher(json);
    if (m.find()) {
      try {
        return m.group(1);
      } catch (Exception ignore) { /* fall through */ }
    }
    return null;
  }

  // Minimal JSON double-array parsing for keys like: "geometry_scale_xyz": [sx, sy, sz]
  private static double[] parseJsonDoubleArray(String json, String key, int expectedLen) {
    if (json == null || key == null || expectedLen <= 0) return null;
    String regex = "\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*\\[(.*?)\\]"; // non-greedy capture inside brackets
    Matcher m = Pattern.compile(regex, Pattern.DOTALL).matcher(json);
    if (!m.find()) return null;
    String inner = m.group(1);
    if (inner == null) return null;
    String[] parts = inner.split(",");
    if (parts.length < expectedLen) return null;
    double[] out = new double[expectedLen];
    try {
      for (int i = 0; i < expectedLen; i++) {
        String t = parts[i].trim();
        // strip quotes if present
        if (t.startsWith("\"") && t.endsWith("\"") && t.length() >= 2) {
          t = t.substring(1, t.length() - 1);
        }
        out[i] = Double.parseDouble(t);
      }
      return out;
    } catch (Exception e) {
      return null;
    }
  }

}

