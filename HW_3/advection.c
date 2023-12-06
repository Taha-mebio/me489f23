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
# include <omp.h>
void InfNorm(mesh_t *msh, double *Q);
#define BUFSIZE 512
/* ************************************************************************** */
int main ( int argc, char *argv[] ){
  if(argc!=3){
    printf("Usage: ./levelSet input.dat thread_number\n");
    return -1;  
  }
  static int frame=0;
  float t0 = omp_get_wtime();

  int thread_count=strtol(argv[2],NULL,10);
  printf("threadcount: %s\n",argv[2]);
  omp_set_num_threads(thread_count);
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
      InfNorm(&advc.msh, advc.q);
    }
  }
  float t1 = omp_get_wtime();
  printf("parallel for shared took: %lf\n", t1-t0);
}


////////////////////////////////////////////

void InfNorm(mesh_t *msh, double *Q){
  double max1=0,max2=0,max3=0,max4=0,local_max5=0,max5=0;
  double t1=omp_get_wtime();
  for(int n=0; n+1< msh->Nnodes; n++){
      if (fabs(Q[n+1])>=max1){
        max1 = Q[n+1];
        }
      }
  printf("Max1:%.8f\n",max1);
  double t2=omp_get_wtime();
  printf("The time with no parallelization:%f\n",t2-t1);
  double t3=omp_get_wtime();   
  #pragma omp parallel for 
  for(int ii=0; ii< msh->Nnodes-1; ii++){
    if (fabs(Q[ii+1])>=max2){
        max2 = Q[ii+1];
        }
      }
  printf("Max2:%.8f\n",max2);
  double t4=omp_get_wtime();,
  printf("The time with only parallel for:%f\n",t4-t3);
  double t5=omp_get_wtime();
  #pragma omp parallel for 
    for(int n=0; n< msh->Nnodes-1; n++){
  #pragma omp critical
      {
        if (fabs(Q[n+1])>=max3){
        max3 = Q[n+1];
        }
      }
      }
    
  printf("Max3:%.8f\n",max3);
  double t6=omp_get_wtime(); 
  printf("The time with critical:%f\n",t6-t5);
     double t9=omp_get_wtime();  
    #pragma omp parallel for reduction(max:local_max5)
    for(int n = 0;n < msh->Nnodes-1; n++) {
    if (fabs(Q[n+1]) >= local_max5) {
      local_max5 = Q[n+1];
    }
  }
  printf("Max4:%.8f\n",local_max5);
  double t10=omp_get_wtime();
  printf("The time with reduction%f\n",t10-t9);
    #pragma omp critical
  {
    if (local_max5 > max5) {
      max5= local_max5;
    }
  }
     //printf("The time with no parallelization:%f\n",t10-t9);
  }




/* ************************************************************************** */
void RhsQ(solver_t *solver, tstep_t *tstep, int stage){
mesh_t *msh = &solver->msh;
#pragma omp parallel for default(none) shared(solver,msh,tstep,stage)
for(int j=0; j<msh->NY; j++){
    for(int i=0; i<msh->NX; i++){
      const int idn = j*msh->NX + i; 
      const double unx = solver->u[2*idn + 0];
      const double uny = solver->u[2*idn + 1];
      // neighbor elements
      int elmE = msh->N2N[4*idn + 0];
      int elmN = msh->N2N[4*idn + 1];
      int elmW = msh->N2N[4*idn + 2];
      int elmS = msh->N2N[4*idn + 3];

      // neighbor velocities
      double uxE = solver->u[2*elmE + 0];
      double uyN = solver->u[2*elmN + 1];
      double uxW = solver->u[2*elmW + 0];
      double uyS = solver->u[2*elmS + 1];

      // Find spacing just in case it is not uniform
      double hip1 = fabs(msh->x[elmE] - msh->x[idn] ); 
      double him1 = fabs(msh->x[idn]  - msh->x[elmW]);  

      double hjp1 = fabs(msh->y[elmN] - msh->y[idn] );  
      double hjm1 = fabs(msh->y[idn]  - msh->y[elmS]);  

      double dfqdx = unx> 0 ? (unx*solver->q[idn]- uxW*solver->q[elmW])/him1 :  (uxE*solver->q[elmE]- unx*solver->q[idn])/hip1;
      double dfqdy = uny> 0 ? (uny*solver->q[idn]- uyS*solver->q[elmS])/hjm1 :  (uyN*solver->q[elmN]- uny*solver->q[idn])/hjp1;

      double rhsq   = -(dfqdx +dfqdy);
      // Time integration i.e. resq = rk4a(stage)* resq + dt*rhsq//

      double resq = tstep->rk4a[stage]*tstep->resq[idn] + tstep->dt*rhsq;
      // Update q i.e. q = q 6 rk4b(stage)*resq
     
      {
        solver->q[idn]  +=  tstep->rk4b[stage]*resq;
      tstep->resq[idn] = resq; 
      tstep->rhsq[idn] = rhsq; 
      }
          }
           }
          } 
      
/* ************************************************************************** */
void initialCondition(solver_t *solver){
  mesh_t *msh = &(solver->msh); 

  solver->q = (double *)malloc(msh->Nnodes*sizeof(double)); 
  solver->u = (double *)malloc(2*msh->Nnodes*sizeof(double));

  for(int j=0; j<msh->NY; j++){
    for(int i=0; i<msh->NX; i++){
      const int idn = j*msh->NX + i; 
      const double xn = msh->x[idn]; 
      const double yn = msh->y[idn]; 
      const double rn = 0.15; 
      const double xc = 0.50; 
      const double yc = 0.75; 

      solver->q[idn] = sqrt((xn-xc)*(xn-xc) + (yn-yc)*(yn-yc)) -rn;
      solver->u[2*idn + 0] =  sin(4.0*M_PI*(xn + 0.5))*sin(4.0*M_PI*(yn + 0.5)); 
      solver->u[2*idn + 1] =  cos(4.0*M_PI*(xn + 0.5))*cos(4.0*M_PI*(yn + 0.5)); 

    }
  }

}

/*
solver->u[2*idn + 0] = -sin(2.0*M_PI*yn)*sin(M_PI*xn)*sin(M_PI*xn); 
solver->u[2*idn + 1] =  sin(2.0*M_PI*xn)*sin(M_PI*yn)*sin(M_PI*yn); 
*/

/* ************************************************************************** */
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

/* ************************************************************************** */
// void createMesh(struct mesh *msh){
mesh_t createMesh(char* inputFile){

  mesh_t msh; 

  msh.NX   = readInputFile(inputFile, "NX");
  msh.NY   = readInputFile(inputFile, "NY");
  msh.xmin = readInputFile(inputFile, "XMIN");
  msh.xmax = readInputFile(inputFile, "XMAX");
  msh.ymin = readInputFile(inputFile, "YMIN");
  msh.ymax = readInputFile(inputFile, "YMAX");


  // msh.dx = (msh.xmax - msh.xmin) / ( msh.NX - 1 );
  // msh.dy = (msh.ymax - msh.ymin) / ( msh.NY - 1 );

#if DEBUG==1
  printf("  The number of interior X grid points is %d\n", msh.NX);
  printf("  The number of interior Y grid points is %d\n", msh.NY); 
  printf("  The x grid spacing is %.4f\n", msh.dx);
  printf("  The y grid spacing is %.4f\n", msh.dy);
#endif

  msh.Nnodes = msh.NX*msh.NY;
  msh.x = (double *) malloc(msh.Nnodes*sizeof(double));
  msh.y = (double *) malloc(msh.Nnodes*sizeof(double));
  for(int j=0; j<msh.NY; j++){
    for(int i=0; i<msh.NX; i++){
      const int idn = j*msh.NX + i; 
      msh.x[idn] = i*(msh.xmax - msh.xmin)/(msh.NX-1);
      msh.y[idn] = j*(msh.ymax - msh.ymin)/(msh.NY-1);
    }
  }

  // Create periodic connectivity
  msh.N2N = (int *)malloc(4*msh.Nnodes*sizeof(int)); 
  for(int j=0; j<msh.NY; j++){
    for(int i=0; i<msh.NX; i++){
      const int idn = j*msh.NX + i; 
      msh.N2N[4*idn + 0] = j*msh.NX + i + 1;  
      msh.N2N[4*idn + 1] = (j+1)*msh.NX +i;  
      msh.N2N[4*idn + 2] = j*msh.NX +i -1;  
      msh.N2N[4*idn + 3] = (j-1)*msh.NX +i;  

      if(j==0){
        msh.N2N[4*idn + 3] = (msh.NY-1)*msh.NX +i;  
      }
      if(j==(msh.NY-1)){
        msh.N2N[4*idn + 1] = 0*msh.NX+i;  
      }
      if(i==0){
        msh.N2N[4*idn + 2] = (j+1)*msh.NX + 0 -1 ; ;   
      }
      if(i==(msh.NX-1)){
        msh.N2N[4*idn + 0] = j*msh.NX + i -msh.NX +1; ;   
      }
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
    float max;
    for(int n=0; n< msh->Nnodes; n++){
      fprintf(fp, "%.8f, %.8f,%.8f,%.8f\n", msh->x[n], msh->y[n], 0.0, Q[n]);
    } 
    for(int n=0; n+1< msh->Nnodes; n++){
      if (Q[n+1]>= Q[n]){
        max = Q[n+1];
        }
      }
    fprintf(fp,"Maximum value is:%.8f",max);
  }


/* ************************************************************************** */
double readInputFile(char *fileName, char* tag){
  FILE *fp = fopen(fileName, "r");
  if (fp == NULL) {
    printf("Error opening the input file\n");
    return -1;
  }

  int sk = 0; 
  double result; 
  char buffer[BUFSIZE];
  char fileTag[BUFSIZE]; 
  while(fgets(buffer, BUFSIZE, fp) != NULL){
    sscanf(buffer, "%s", fileTag);
    if(strstr(fileTag, tag)){
      fgets(buffer, BUFSIZE, fp);
      sscanf(buffer, "%lf", &result); 
      return result;
    }
    sk++;
  }

  if(sk==0){
    printf("could not find the tag: %s in the file %s\n", tag, fileName);
  }
}


