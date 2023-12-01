# include <stdio.h>
# include <stdlib.h>
# include <time.h>

int main ( int argc, char *argv[] ){
  if(argc!=2){
    printf("Usage: ./levelSet input.dat\n");
    return -1;  
  }
  static int frame=0;

  // Create an advection solver
  solver_t advc; 
  // Create uniform rectangular (Cartesian) mesh
  advc.msh = createMesh(argv[1]); 
  // Create time stepper 
  tstep_t tstep = createTimeStepper(advc.msh.Nnodes); 
  // Create Initial Field
  initialCondition(&advc);

  // Read input file for time variables 
  tstep.tstart = readInputFile(argv[1], "TSART");
  tstep.tend   = readInputFile(argv[1], "TEND");
  tstep.dt     = readInputFile(argv[1], "DT");
  tstep.time = 0.0; 

  // adjust time step size 
  int Nsteps = ceil( (tstep.tend - tstep.tstart)/tstep.dt);
  tstep.dt = (tstep.tend - tstep.tstart)/Nsteps;

  // Read input file for OUTPUT FREQUENCY i.e. in every 1000 steps
  int Noutput = readInputFile(argv[1], "OUTPUT_FREQUENCY");


  // write the initial solution i.e. q at t = tstart
  {
    char fname[BUFSIZ];
    sprintf(fname, "test_%04d.csv", frame++);
    solverPlot(fname, &advc.msh, advc.q);
  }


  // ********************Time integration***************************************/
  // for every steps
  for(int step = 0; step<Nsteps; step++){
    // for every stage
    for(int stage=0; stage<tstep.Nstage; stage++){
      // Call integration function
      RhsQ(&advc, &tstep, stage); 
    }

    tstep.time = tstep.time+tstep.dt;

    if(step%Noutput == 0){
      char fname[BUFSIZ];
      sprintf(fname, "test_%04d.csv", frame++);
      solverPlot(fname, &advc.msh, advc.q);
    }
  }
}

double readInputFile(char *input.dat, char* tag){
  FILE *fp = fopen(fileName, "r");
  if (fp == NULL) {
    printf("let's gooo\n");
    return 0;
  }
}
