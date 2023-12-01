/* This is a sample Advection solver in C 
The advection equation-> \partial q / \partial t - u \cdot \nabla q(x,y) = 0
The grid of NX by NX evenly spaced points are used for discretization.  
The first and last points in each direction are boundary points. 
Approximating the advection operator by 1st order finite difference. 
*/
# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include "advection.h"

#define BUFSIZE 512
/* ************************************************************************** */
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

/* ************************************************************************** */
void RhsQ(solver_t *solver, tstep_t *tstep, int stage) {
    mesh_t *msh = &solver->msh;
    double *q = solver->q;
    double *u = solver->u;
    double dt = tstep->dt;

    for (int j = 0; j < msh->NY; j++) {
        for (int i = 0; i < msh->NX; i++) {
            int n = j * msh->NX + i;
            double dq_dt;

            if(u[2*n]>=0 && u[2*n +1]>=0){
              dq_dt = -(((u[2*n]*q[n] - u[2*(msh->N2N[4*n+2])] * q[msh->N2N[4*n+2]])/(msh->x[n] - msh->x[msh->N2N[4*n+2]])) + (((u[2*n+1]*q[n] - u[2*(msh->N2N[4*n+2])+1] * q[msh->N2N[4*n+2]])/(msh->y[n] - msh->y[msh->N2N[4*n+3]]))));
            } else if (u[2*n]<0 && u[2*n + 1]>=0) {
              dq_dt = -(((u[2*(msh->N2N[4*n])]*q[msh->N2N[4*n]] - u[2*n] * q[n])/(msh->x[msh->N2N[4*n]] - msh->x[n])) + (((u[2*n+1]*q[n] - u[2*(msh->N2N[4*n+2])+1] * q[msh->N2N[4*n+2]])/(msh->y[n] - msh->y[msh->N2N[4*n+3]]))));
            } else if (u[2*n]<0 && u[2*n +1]<0) {
              dq_dt = -(((u[2*(msh->N2N[4*n])]*q[msh->N2N[4*n]] - u[2*n] * q[n])/(msh->x[msh->N2N[4*n]] - msh->x[n])) + (((u[2*(msh->N2N[4*n+1])+1]*q[msh->N2N[4*n+1]] - u[2*n+1] * q[n])/(msh->y[msh->N2N[4*n+1]] - msh->y[n]))));
            } else {
              dq_dt = -(((u[2*n]*q[n] - u[2*(msh->N2N[4*n+2])] * q[msh->N2N[4*n+2]])/(msh->x[n] - msh->x[msh->N2N[4*n+2]])) + (((u[2*(msh->N2N[4*n+1])+1]*q[msh->N2N[4*n+1]] - u[2*n+1] * q[n])/(msh->y[msh->N2N[4*n+1]] - msh->y[n]))));
            } 
            // Time integration in 2 steps
            // Step 1: Update residual
            tstep->resq[n] = tstep->rk4a[stage] * tstep->resq[n] + dt * dq_dt;
            // Step 2: Update solution and store
            q[n] = q[n] + tstep->rk4b[stage] * tstep->resq[n];
        }
    }
}


/* ************************************************************************** */
void initialCondition(solver_t *solver){
  mesh_t *msh = &(solver->msh); 

  solver->q = (double *)malloc(msh->Nnodes*sizeof(double)); 
  solver->u = (double *)malloc(2*msh->Nnodes*sizeof(double));
  double xc = 0.5;
  double yc = 0.75;
  double r = 0.15;

  for(int j=0; j<msh->NY; j++){
    for(int i=0; i<msh->NX; i++){
      int n = j * msh->NX + i;
      double x = msh->x[n];
      double y = msh->y[n];
      // Initial condition for q
      solver->q[n] = sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc)) - r;

      // Initial conditions for u and v
      double angle = 4.0 * M_PI * (x + 0.5);
      solver->u[2 * n] = sin(angle) * sin(4.0 * M_PI * (y + 0.5));
      solver->u[2 * n + 1] = cos(angle) * cos(4.0 * M_PI * (y + 0.5));
     /*
     Create initial condition and velocity field
     */
    }
  }
}

/* ************************************************************************** */
// void createMesh(struct mesh *msh){
mesh_t createMesh(char* inputFile){

  mesh_t msh; 

  msh.NX   = readInputFile(inputFile, "[NX]");
  msh.NY   = readInputFile(inputFile, "[NY]");
  msh.xmin = readInputFile(inputFile, "[XMIN]");
  msh.xmax = readInputFile(inputFile, "[XMAX]");
  msh.ymin = readInputFile(inputFile, "[YMIN]");
  msh.ymax = readInputFile(inputFile, "[YMAX]");

  /* 
  Continue with other required fields
  */

  msh.Nnodes = msh.NX*msh.NY;
  msh.x = (double *) malloc(msh.Nnodes*sizeof(double));
  msh.y = (double *) malloc(msh.Nnodes*sizeof(double));
  /*
  Compute Coordinates of the nodes
  */
  for(int j=0; j<msh.NY; j++){
    for(int i=0; i<msh.NX; i++){
      int n = j * msh.NX + i; // Linear index
      msh.x[n] = ((double)i / (msh.NX - 1)) * (msh.xmax - msh.xmin) + msh.xmin; /*Compute x-coordinate based on i */
      msh.y[n] = ((double)j / (msh.NY - 1)) * (msh.ymax - msh.ymin) + msh.ymin;/* Compute y-coordinate based on j */
     /*
      Complete this part
    */
    }
  }
  // Create connectivity and periodic connectivity
  /* 
  for every node 4 connections east north west and south
  Nothe that periodic connections require specific treatment
  */
  msh.N2N = (int *)malloc(4*msh.Nnodes*sizeof(int));

  for(int j=0; j<msh.NY; j++){
    for(int i=0; i<msh.NX; i++){
            int n = j * msh.NX + i;  // Linear index

            // Compute the indices of the neighboring nodes
            int east = (i + 1) % msh.NX + j * msh.NX;
            int north = i + ((j + 1) % msh.NY) * msh.NX;
            int west = (i - 1 + msh.NX) % msh.NX + j * msh.NX;
            int south = i + ((j - 1 + msh.NY) % msh.NY) * msh.NX;

            // Assign the connectivity
            msh.N2N[4 * n] = east;
            msh.N2N[4 * n + 1] = north;
            msh.N2N[4 * n + 2] = west;
            msh.N2N[4 * n + 3] = south;
    /*
     Complete this part
    */
      
    }
  }
  
  return msh; 
}


/* ************************************************************************** */
void solverPlot(char *fileName, mesh_t *msh, double *Q){
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening file\n");
        return;
    }

    fprintf(fp, "X,Y,Z,Q \n");
    for(int n=0; n< msh->Nnodes; n++){
      fprintf(fp, "%.8f, %.8f,%.8f,%.8f\n", msh->x[n], msh->y[n], 0.0, Q[n]);
    } 
    fclose(fp);
}

/* ************************************************************************** */
double readInputFile(char *fileName, char* tag) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        printf("Error opening the input file\n");
        return -1;
    }

    char line[100];
    double value;

    while (fgets(line, 100, fp)) {
        if (strstr(line, tag)) {
            if (fgets(line, sizeof(line), fp)) {
                // Extract the value from the next line
                sscanf(line, "%lf", &value);
            }

            break;
        }
    }

    if (value == -1.0) {
        printf("Error: Tag '%s' not found in the input file.\n", tag);
    } else {
        printf("%lf\n", value);
    }

    fclose(fp);
    return value;
}

/* ************************************************************************** */
// Time stepper clas RK(4-5)
// resq = rk4a(stage)* resq + dt*rhsq
//  q = q + rk4b(stage)*resq
tstep_t createTimeStepper(int Nnodes){
  tstep_t tstep; 
  tstep.Nstage = 5; 
  tstep.resq = (double *)calloc(Nnodes,sizeof(double)); 
  tstep.rhsq = (double *)calloc(Nnodes,sizeof(double));
  tstep.rk4a = (double *)malloc(tstep.Nstage*sizeof(double));
  tstep.rk4b = (double *)malloc(tstep.Nstage*sizeof(double));
  tstep.rk4c = (double *)malloc(tstep.Nstage*sizeof(double));

  tstep.rk4a[0] = 0.0; 
  tstep.rk4a[1] = -567301805773.0/1357537059087.0; 
  tstep.rk4a[2] = -2404267990393.0/2016746695238.0;
  tstep.rk4a[3] = -3550918686646.0/2091501179385.0;
  tstep.rk4a[4] = -1275806237668.0/842570457699.0;
        
  tstep.rk4b[0] = 1432997174477.0/9575080441755.0;
  tstep.rk4b[1] = 5161836677717.0/13612068292357.0; 
  tstep.rk4b[2] = 1720146321549.0/2090206949498.0;
  tstep.rk4b[3] = 3134564353537.0/4481467310338.0;
  tstep.rk4b[4] = 2277821191437.0/14882151754819.0;
             
  tstep.rk4c[0] = 0.0;
  tstep.rk4c[1] = 1432997174477.0/9575080441755.0;
  tstep.rk4c[2] = 2526269341429.0/6820363962896.0;
  tstep.rk4c[3] = 2006345519317.0/3224310063776.0;
  tstep.rk4c[4] = 2802321613138.0/2924317926251.0;
  return tstep; 
}