#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>
#include "petsc.h"
#include "petscksp.h"
#include "gel.h"
#include "silo.h"

typedef struct {
     Mat       LSinv;
     Mat       LCLinv;
     Mat       LCAinv;

//matrix related to disolved species
     Mat       L;
     Mat       LS;
     Mat       LCL;
     Mat       LCA;

//matrix related to Laplacian of electric potential
     Mat       LPS;
     Mat       LPCL;
     Mat       LPCA;

     Mat       DI; //diagonal matrix 
     GEL       *gel;

     Vec       tmp1;
     Vec       tmp2;
     Vec       tmp[4];

} AppCtx;

/* Declare routines for user-provided matrix*/
extern PetscErrorCode UserMultLSinv(Mat,Vec,Vec);
extern PetscErrorCode UserMultLCLinv(Mat,Vec,Vec);
extern PetscErrorCode UserMultLCAinv(Mat,Vec,Vec);
extern PetscErrorCode UserMultSRR(Mat,Vec,Vec);

static void   init_gel_struct(GEL*,int,char*[]);
static void   main_time_step_loop(GEL*);
static int    update_time_step(double,GEL*,int*);
static int    update_time_step_old(double,GEL*,int*);

static void   update_volume_fraction(GEL*,int*);
static void   update_volume_fraction_zerodelta(GEL*,int*);
static void   update_tao_and_z(GEL*,int*);
static void   update_tao_and_znew(GEL*,int*);

static void   update_taoz_flux(GEL*,int*);
static void   update_taoz_fluxnew(GEL*,int*);

static void   update_ghost_taoz_sn(GEL*,int*);
static void   update_ghost_taoz_we(GEL*,int*);

static void   increase_network_nozero(GEL*,int*);
static void   decrease_network_zero(GEL*,int*);

static void   update_bdion_advection(GEL*,int*);
static void   update_bdion_advection1(GEL*,int*);

static void   extrapolate_ion_concentration(GEL*,int*);
static void   extrapolate_ion_concentration1(GEL*,int*);

static void   update_dion_concentration(GEL*,int*);
static void   update_chemical_potential(GEL*,int*);

static double minmod(double,double,double);
static void   update_edge_flux_1st(GEL*,int*,double,double);
static void   update_edge_flux_2ndv1(GEL*,int*,double,double);
static void   update_edge_flux_2ndv2(GEL*,int*,double,double);

static void   update_edge_flux_bion(GEL*,int*,double,double);
static void   update_edge_flux_dion(GEL*,int*,double,double);
static void   update_edge_flux_dion1(GEL*,int*,double,double);


static void   interpolate_coarse_thn(GEL*,int,char*);
static void   update_buffer_estate(GEL*,int*);
static void   update_ghost_estate_sn(GEL*,int*);
static void   update_ghost_estate_we(GEL*,int*);

static void   update_bion_reaction(GEL*,int*);

static void   update_buffer_testate(GEL*,int*);
static void   update_ghost_testate_sn(GEL*,int*);
static void   update_ghost_testate_we(GEL*,int*);
static void   update_buffer_tao_and_z(GEL*,int*);
static void   F1_cycle(GEL*,int*,int);
static void   V1_cycle(GEL*,int*,int);
static void   rbgs_relaxation(GEL*,int*);
static void   residual_evaluation1(GEL*,int*);
static void   next_coarse_grid1(GEL*,GEL*);
static void   next_fine_grid1(GEL*,GEL*);

static PetscErrorCode SampleShellPCApply1(PC pc,Vec x,Vec y);

#define PI    3.1415926535897932

static  int   comm_size;

int main(argc,argv)
   int  argc;
   char *argv[];
{
    GEL     gel;
    double  start,finish;

    MPI_Init(&argc,&argv);
    PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);
    init_gel_struct(&gel,argc,argv);
    init_gel_state(&gel,gel.rgr.gmax);
    update_ghost_cell_states_we(&gel,gel.rgr.gmax);
    update_ghost_cell_states_sn(&gel,gel.rgr.gmax);

    update_avg_theta(&gel,gel.rgr.gmax);
  
    main_time_step_loop(&gel);

    free(gel.st);
    MPI_Barrier(MPI_COMM_WORLD);
    finish = MPI_Wtime();

    //system("hostname");    
    PetscFinalize();
    MPI_Finalize();
}

static void main_time_step_loop(
       GEL    *gel)
{
      int     i,j,step;
      int     *lbuf = gel->rgr.lbuf;
      int     *ubuf = gel->rgr.ubuf;
      double  h = gel->rgr.h;
      double  time,start,finish,x,y;
      bool    last = NO; 

      int     Nx = gel->rgr.gmax[0]-lbuf[0]-ubuf[0];
      int     Ny = gel->rgr.gmax[1]-lbuf[1]-ubuf[1];

      gel->step = step = 0;

      for(time=gel->stime; time<gel->etime; time+=gel->dt)
      {
         gel->time = time;

         gel->print = update_time_step(time,gel,gel->rgr.gmax); 

//get the velocity at t_k
         increase_network_nozero(gel,gel->rgr.gmax);
         update_chemical_potential(gel,gel->rgr.gmax);
         gel->gel_system_solver(gel,gel->rgr.gmax,0);
         decrease_network_zero(gel,gel->rgr.gmax);

         if(gel->step % 1 == 0)  
         {
	     float          *vx2,*vy2,*vx1,*vy1,*x,*y,*vx,*vy;
             float          *gpx,*gpy,*dx,*dy,*ftx,*fty,*vftx,*vfty;

	     float          *value1,*value2,*var[2],*var1[2],*var2[2],*var3[2],*var4[2],*var5[2],*var6[2];
	     float          *value5,*value6,*value7,*value8,*value3,*value4,*coords[2];
	     float          *value9,*value10,*value11,*value12,*value13,*value14,*value15;
             float          *value16,*value17,*value18,*value19,*value20,*value21;
             float          *value22,*value23,*value24,*value25,*value26,*value27,*value28;
             float          *value29,*value30,*value31,*value32,*value33,*value34,*value35;
             float          *value36,*value37,*value38,*value39,*value40;

             int            dims[2],dimm[2],total;
	     char           fname[100],filename[100];
	     char           *finput,*fmatlab;
             
	     DBfile         *dbfile;
	     char           s[100];
	     char           *varname[2]  = {"X-Velocity","Y-Velocity"};
             char           *varname1[2] = {"X1-Velocity","Y1-Velocity"};

             State          *st;

             sprintf(filename,"./visit_output/plot_%s_node%d_step%d.silo",gel->output,gel->my_id,step);

             dbfile = DBCreate(filename, 0, DB_LOCAL, "The Test", DB_PDB);

	     dims[0] = Nx; dims[1] = Ny;
             dimm[0] = Nx+1; dimm[1] = Ny+1; 
              
	     x = malloc((Nx+1)*sizeof(float));
	     y = malloc((Ny+1)*sizeof(float));
	     coords[0] = x; coords[1] = y;

             value1 = malloc(Nx*Ny*sizeof(float));
	     value2 = malloc(Nx*Ny*sizeof(float));
             value3 = malloc(Nx*Ny*sizeof(float));
	     value4 = malloc(Nx*Ny*sizeof(float));
             value5 = malloc(Nx*Ny*sizeof(float));
             value6 = malloc(Nx*Ny*sizeof(float));
             value7 = malloc(Nx*Ny*sizeof(float));
             value8 = malloc(Nx*Ny*sizeof(float));
             value9 = malloc(Nx*Ny*sizeof(float));

             value10 = malloc(Nx*Ny*sizeof(float));
             value11 = malloc(Nx*Ny*sizeof(float));
             value12 = malloc(Nx*Ny*sizeof(float));
             value13 = malloc(Nx*Ny*sizeof(float));
             value14 = malloc(Nx*Ny*sizeof(float));
             value15 = malloc(Nx*Ny*sizeof(float));

             value16 = malloc(Nx*Ny*sizeof(float));
             value17 = malloc(Nx*Ny*sizeof(float));
             value18 = malloc(Nx*Ny*sizeof(float));
             value19 = malloc(Nx*Ny*sizeof(float));
             value20 = malloc(Nx*Ny*sizeof(float));
             value21 = malloc(Nx*Ny*sizeof(float));
             value22 = malloc(Nx*Ny*sizeof(float));
              
             value23 = malloc(Nx*Ny*sizeof(float));
             value24 = malloc(Nx*Ny*sizeof(float));
             value25 = malloc(Nx*Ny*sizeof(float));
             value26 = malloc(Nx*Ny*sizeof(float));
             value27 = malloc(Nx*Ny*sizeof(float));
             value28 = malloc(Nx*Ny*sizeof(float));

             value29 = malloc(Nx*Ny*sizeof(float));
             value30 = malloc(Nx*Ny*sizeof(float));
             value31 = malloc(Nx*Ny*sizeof(float));
             value32 = malloc(Nx*Ny*sizeof(float));
             value33 = malloc(Nx*Ny*sizeof(float));
             value34 = malloc(Nx*Ny*sizeof(float));

             value35 = malloc(Nx*Ny*sizeof(float));
             value36 = malloc(Nx*Ny*sizeof(float));
             value37 = malloc(Nx*Ny*sizeof(float));
             value38 = malloc(Nx*Ny*sizeof(float));
             value39 = malloc(Nx*Ny*sizeof(float));
             value40 = malloc(Nx*Ny*sizeof(float));

	     vx  = malloc(Nx*Ny*sizeof(float));
       	     vy  = malloc(Nx*Ny*sizeof(float));
             vx1 = malloc(Nx*Ny*sizeof(float));
             vy1 = malloc(Nx*Ny*sizeof(float));

             for(i=lbuf[0]-1; i<gel->rgr.gmax[0]-ubuf[0];++i)
	        x[i-lbuf[0]+1] = Rect_coord(i,gel->rgr,0)+0.5*h;
             for(i=lbuf[1]-1; i<gel->rgr.gmax[1]-ubuf[1];++i)
	        y[i-lbuf[1]+1] = Rect_coord(i,gel->rgr,1)+0.5*h;
               
	     for(i=lbuf[0]; i<gel->rgr.gmax[0]-1; ++i)
	     {
	        for(j=lbuf[1]; j<gel->rgr.gmax[1]-1; ++j)
		{
                   st = &(Rect_state(i,j,gel));

                   value1[(j-lbuf[1])*Nx+(i-lbuf[0])] = Thn(gel,i,j);
                   value2[(j-lbuf[1])*Nx+(i-lbuf[0])] = S(gel,i,j);
                   value3[(j-lbuf[1])*Nx+(i-lbuf[0])] = CL(gel,i,j);
                   value4[(j-lbuf[1])*Nx+(i-lbuf[0])] = CA(gel,i,j);
                   value5[(j-lbuf[1])*Nx+(i-lbuf[0])] = BS(gel,i,j);
                   value6[(j-lbuf[1])*Nx+(i-lbuf[0])] = BCA(gel,i,j);
                   value7[(j-lbuf[1])*Nx+(i-lbuf[0])] = BC2(gel,i,j);
                   value8[(j-lbuf[1])*Nx+(i-lbuf[0])] = -st->bun;
                   value9[(j-lbuf[1])*Nx+(i-lbuf[0])] = Phi(gel,i,j);
                   
                   value10[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->gunx;
                   value11[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guny;
                   value12[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->gusx;
                   value13[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->gusy;
                   
                   value14[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guyn[0];
                   value15[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guyn[1];
                   value16[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guyn[2];
                   value17[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guyn[3];
                   value18[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guyn[4];
                  
                   value36[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guxn[0];
                   value37[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guxn[1];
                   value38[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guxn[2];
                   value39[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guxn[3];
                   value40[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guxn[4];
 
                   value19[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guys[0];
                   value20[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guys[1];
                   value21[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guys[2];
                   value22[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guys[3];

                   value32[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guxs[0];
                   value33[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guxs[1];
                   value34[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guxs[2];
                   value35[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guxs[3];

                   value23[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guy[0];
                   value24[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guy[1];
                   value25[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->guy[2];

                   value26[(j-lbuf[1])*Nx+(i-lbuf[0])] = -st->bvn;
                   value27[(j-lbuf[1])*Nx+(i-lbuf[0])] = -st->bvs;
                   value28[(j-lbuf[1])*Nx+(i-lbuf[0])] = -st->bus;

                   value29[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->gux[0];
                   value30[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->gux[1];
                   value31[(j-lbuf[1])*Nx+(i-lbuf[0])] = st->gux[2];

                   
                   vx[(j-lbuf[1])*Nx+(i-lbuf[0])]  = Un(gel,i,j);
                   vy[(j-lbuf[1])*Nx+(i-lbuf[0])]  = Vn(gel,i,j);

                   vx1[(j-lbuf[1])*Nx+(i-lbuf[0])]  = Us(gel,i,j);
                   vy1[(j-lbuf[1])*Nx+(i-lbuf[0])]  = Vs(gel,i,j);
                 }
	     }
	     var[0] = vx; var[1] = vy;
	     var1[0] = vx1; var1[1] = vy1;

	     DBPutQuadmesh(dbfile, "Mesh", NULL, coords, dimm, 2, DB_FLOAT,DB_COLLINEAR, NULL);

             DBPutQuadvar1(dbfile, "Thn", "Mesh", value1, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "Sodium", "Mesh", value2, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "Chloride", "Mesh", value3, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "Calcium", "Mesh", value4, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "BoundSodium", "Mesh", value5, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "BoundCalcium", "Mesh", value6, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "DBoundCalcium", "Mesh", value7, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "bun", "Mesh", value8, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "Psi", "Mesh", value9, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             
             DBPutQuadvar1(dbfile, "gunx", "Mesh", value10, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guny", "Mesh", value11, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "gusx", "Mesh", value12, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "gusy", "Mesh", value13, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             
             DBPutQuadvar1(dbfile, "guyn0", "Mesh", value14, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guyn1", "Mesh", value15, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guyn2", "Mesh", value16, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guyn3", "Mesh", value17, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guyn4", "Mesh", value18, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
            
             DBPutQuadvar1(dbfile, "guxn0", "Mesh", value36, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guxn1", "Mesh", value37, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guxn2", "Mesh", value38, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guxn3", "Mesh", value39, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guxn4", "Mesh", value40, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
 
             DBPutQuadvar1(dbfile, "guys0", "Mesh", value19, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guys1", "Mesh", value20, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guys2", "Mesh", value21, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guys3", "Mesh", value22, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
           
             DBPutQuadvar1(dbfile, "guxs0", "Mesh", value32, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guxs1", "Mesh", value33, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guxs2", "Mesh", value34, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guxs3", "Mesh", value35, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
 
             DBPutQuadvar1(dbfile, "guyS", "Mesh", value23, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guyCL", "Mesh", value24, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guyCA", "Mesh", value25, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);

             DBPutQuadvar1(dbfile, "bvn", "Mesh", value26, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "bvs", "Mesh", value27, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "bus", "Mesh", value28, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);

             DBPutQuadvar1(dbfile, "guxS", "Mesh", value29, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guxCL", "Mesh", value30, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBPutQuadvar1(dbfile, "guxCA", "Mesh", value31, dims, 2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
 
             DBPutQuadvar(dbfile, "netV","Mesh",2,varname,var,dims,2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL); 
	     DBPutQuadvar(dbfile, "solV","Mesh",2,varname1,var1,dims,2,NULL, 0, DB_FLOAT, DB_ZONECENT, NULL);
             DBClose(dbfile);

             free(x); free(y);
	     free(vx); free(vy);
	     free(vx1); free(vy1); 

             free(value1); free(value2);
	     free(value3); free(value4);
             free(value5); free(value6);
             free(value7); free(value8);
             free(value9); free(value10);
             free(value11); free(value12);
             free(value13); free(value14);
             free(value15); free(value16);
             free(value17); free(value18);
             free(value19); free(value20);
             free(value21); free(value22); free(value23);
             free(value24); free(value25); free(value26);
             free(value27); free(value28); free(value29);
             free(value30); free(value31); free(value32);
             free(value33); free(value34); free(value35);
             free(value36); free(value37); free(value38);
             free(value39); free(value40);
	 }
  
//update volume fraction from k to k+1, with extraplated u_{k+1/2}
         update_volume_fraction_zerodelta(gel,gel->rgr.gmax);

         update_ghost_cell_states_we(gel,gel->rgr.gmax);
         update_ghost_cell_states_sn(gel,gel->rgr.gmax);
         update_avg_theta(gel,gel->rgr.gmax);

//account for advection, with extraplated u_{k+1/2}

         update_bdion_advection(gel,gel->rgr.gmax);

//extrapolate values to k+1
         
         extrapolate_ion_concentration(gel,gel->rgr.gmax);

         update_ghost_cell_states_we(gel,gel->rgr.gmax);
         update_ghost_cell_states_sn(gel,gel->rgr.gmax);

//account for diffusion and reaction
         update_bion_reaction(gel,gel->rgr.gmax);
         
         update_dion_concentration(gel,gel->rgr.gmax);

         update_ghost_cell_states_we(gel,gel->rgr.gmax);
         update_ghost_cell_states_sn(gel,gel->rgr.gmax);

         if(gel->my_id == 0)
	    printf("\n time step %d = %6.5e ",step,gel->time);
	 if(fabs(time-gel->etime) < 1.0e-6)
	    break;
	
         step++; gel->step = step;
      }
      if(gel->my_id == 0)
         fclose(gel->file);
}

static int update_time_step(
       double time,
       GEL    *gel,
       int    *gmax)
{
       static int  counter = 1;
       int      print,i,j;
       int      *lbuf = gel->rgr.lbuf;
       int      *ubuf = gel->rgr.ubuf;

       double   *n_dt;
       double   lam_n=gel->lam_n;
       double   lam_s=gel->lam_s;
       double   cfln = gel->cfln;
       double   cfls = gel->cfls;

       double   mu_n=gel->mu_n;
       double   mu_s=gel->mu_s;
       double   xi=gel->xi;
       double   eta_c=gel->eta_c;
       double   eta_l=gel->eta_l;
       double   th_os = gel->th_os;
       State    *st;
       double   h=gel->rgr.h;
       double   vnmax,vsmax,dt,tmp1,tmp2; 
 
       print = NO;
       tmp1 = tmp2 = 0.0; vnmax = vsmax = 0.0;
 
       n_dt = malloc(comm_size*sizeof(double));

       for(i=0; i<gmax[0]; ++i)
       {
          for(j=0; j<gmax[1]; ++j)
	  {
             st=&(Rect_state(i,j,gel));
       
             if(i>=lbuf[0] && j >= lbuf[1] && i<gmax[0]-1 && j<gmax[1]-1)
	     {
                tmp1 = max(tmp1,fabs(Un(gel,i,j)+sqrt(T11(gel,i,j)+Z(gel,i,j))));
                tmp1 = max(tmp1,fabs(Un(gel,i,j)-sqrt(T11(gel,i,j)+Z(gel,i,j))));
                tmp1 = max(tmp1,fabs(Vn(gel,i,j)+sqrt(T22(gel,i,j)+Z(gel,i,j))));
                tmp1 = max(tmp1,fabs(Vn(gel,i,j)-sqrt(T22(gel,i,j)+Z(gel,i,j))));
 
	        tmp2 = max(tmp2,fabs(Un(gel,i,j)));  tmp2 = max(tmp2,fabs(Us(gel,i,j)));
	        tmp2 = max(tmp2,fabs(Vn(gel,i,j)));  tmp2 = max(tmp2,fabs(Vs(gel,i,j))); 
                vnmax = max(vnmax,fabs(Un(gel,i,j))); vnmax = max(vnmax,fabs(Vn(gel,i,j)));
                vsmax = max(vsmax,fabs(Us(gel,i,j))); vsmax = max(vsmax,fabs(Vs(gel,i,j)));
             }
	  }
       }

       //tmp1 = cfln/tmp1;   tmp2 = cfls/tmp2; 
      
       dt = h*cfln/vnmax;

       printf("\n vnmax = %6.5e, vsmax = %6.5e",vnmax,vsmax); 
       //dt = h*min(tmp1,tmp2);
       MPI_Allgather(&dt,1,MPI_DOUBLE,n_dt,1,MPI_DOUBLE,PETSC_COMM_WORLD); 
       
       //store old time and oldold time step
       gel->oodt = gel->odt; 
       gel->odt = gel->dt; 

       gel->dt = 10000.0;

       for(i=0; i<comm_size; ++i)
          gel->dt=min(gel->dt,n_dt[i]);
      
       free(n_dt);
     
       gel->dt = min(1.2*gel->odt,gel->dt); 

       gel->dt = gel->dtmin;

       printf("\n dt is %6.5e",gel->dt);

       if(gel->step == 0) //always print initial states
       {
         gel->odt = gel->dt;
         return YES; 
       }
       if(time+gel->dt > counter*gel->outputf - 1.0e-10) //in case time step exact
       {
          gel->dt = counter*gel->outputf-time;
	  counter++;
          print = YES;
       }
       
       if(time+gel->dt > gel->etime)
       {
          print = YES;
          gel->dt = gel->etime-time;
       }
       
       return print;
}

static void update_tao_and_z(
       GEL    *gel,
       int    *gmax)
{
       int     m,n,p,q,i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;

       double  h = gel->rgr.h;
       double  dt=gel->dt;
       double  beta = gel->beta;
       double  lambda = gel->lambda;

       double  **A;  //4x4 system to solve for each cell
       double  div_un,det,a,b,c,tmpv[3],B[3][3],v1,v2;
       double  rhs[3],xold[3],xnew[3]; //old and new solution vection
       double  C[2][2],tmp[2][2]; //old and new formation term
       double  check,ux,uy,vx,vy,lp1,lp2,lam1,lam2;
       double  r1[2],r2[2],R[2][2],R_inv[2][2];
 
//calculate the advection term on edge on half step 
       update_taoz_flux(gel,gmax);

       A = malloc(3*sizeof(double*));

       for(i = 0; i < 3; i++)
	  A[i] = malloc(3*sizeof(double));
      
       for(i=0; i<gmax[0]; ++i)
       {
          for(j=0; j<gmax[1]; ++j)
          {
              //save old values first
            OT11(gel,i,j) = T11(gel,i,j);
            OT12(gel,i,j) = T12(gel,i,j);
            OT22(gel,i,j) = T22(gel,i,j);

            if(i>=lbuf[0] && i<gmax[0]-ubuf[0] && j>=lbuf[1] && j<gmax[1]-ubuf[1])
            {
              xold[0] = T11(gel,i,j); xold[1] = T12(gel,i,j);
              xold[2] = T22(gel,i,j); 

              tmpv[0] = tmpv[1] = tmpv[2] = 0.0;
              
              ux = (Un(gel,i,j)-Un(gel,i-1,j))/h;
              v1 = 0.5*(Un(gel,i,j+1)+Un(gel,i-1,j+1));
              v2 = 0.5*(Un(gel,i,j-1)+Un(gel,i-1,j-1));
              uy = (v1-v2)/(2*h);
              
              vy = (Vn(gel,i,j)-Vn(gel,i,j-1))/h;
              v1 = 0.5*(Vn(gel,i+1,j)+Vn(gel,i+1,j-1));
              v2 = 0.5*(Vn(gel,i-1,j)+Vn(gel,i-1,j-1));
              vx = (v1-v2)/(2*h); 


              div_un = ux+vy; //divergence of the network velocity

              B[0][0] = 2*ux;   B[0][1] = 2*uy;
              B[0][2] = 0;      

              B[1][0] = vx;     B[1][1] = ux+vy;
              B[1][2] = uy;    ;
 
              B[2][0] = 0;      B[2][1] = 2*vx;
              B[2][2] = 2*vy;   
                       
              for(p=0; p<3; ++p)
                 for(q=0; q<3; ++q)
                 {
                    if(p==q)
                       A[p][q] = 1.0+0.5*dt*beta;
                    else
                       A[p][q] = 0.0;
                  }
            
              for(p=0; p<3; ++p)
                 for(q=0; q<3; ++q)
                    A[p][q] -= 0.5*dt*B[p][q]; 

              for(m=0; m<3; ++m)
                 for(n=0; n<3; ++n)
                    tmpv[m] += B[m][n]*xold[n];
 
              for(p=0; p<3; ++p)
              {
                 rhs[p] = xold[p] -dt/h*(FTx(gel,i,j)[p]-FTx(gel,i-1,j)[p]+FTy(gel,i,j)[p]-FTy(gel,i,j-1)[p]);
                 rhs[p] += 0.5*dt*(tmpv[p]-beta*xold[p]);
              }

              rhs[0] += dt*2*ux*Z(gel,i,j); rhs[1] += dt*(vx+uy)*Z(gel,i,j);
              rhs[2] += dt*2*vy*Z(gel,i,j);

              rhs[0] += dt*lambda*div_un; rhs[2] += dt*lambda*div_un;
 
              linear_system_solver(A,rhs,3,xnew);

              if(isnan(xnew[0]) || isnan(xnew[1]) || isnan(xnew[2])  || isnan(xnew[3]) )
              {
                 printf("\n warning, taoz solver failed for cell[%d %d], with solution = %6.5e %6.5e %6.5e %6.5e",i,j,xnew[0],xnew[1],xnew[2],xnew[3]);
                 linear_system_solver(A,rhs,3,xnew);
              } 

              T11(gel,i,j) = xnew[0];  T12(gel,i,j) =  xnew[1];
              T22(gel,i,j) = xnew[2];   
              //printf("\n solved T11 for cell[%d %d] is %10.9e",i,j,T11(gel,i,j));
              
              //check for semi-positive-definite
              check = (T11(gel,i,j)+Z(gel,i,j))*(T22(gel,i,j)+Z(gel,i,j));
              if(check < T12(gel,i,j)*T12(gel,i,j))
              {
                  printf("\n warning!, in cell[%d %d] violation is found with value=%6.5e !",i,j,
                  check-T12(gel,i,j)*T12(gel,i,j));
          
                  a = T11(gel,i,j)+Z(gel,i,j); b = T12(gel,i,j); 
                  c = T22(gel,i,j)+Z(gel,i,j); 
                  lam1 = 0.5*(a+c+sqrt((a-c)*(a-c)+4*b*b));
                  lam2 = 0.5*(a+c-sqrt((a-c)*(a-c)+4*b*b));

                  printf("\n lam1=%6.5e,lam2=%6.5e",lam1,lam2);

                  r1[0] = 1.0; r1[1] = (lam1-a)/b; 
                  r2[0] = 1.0; r2[1] = (lam2-a)/b; 
                  
                  lp1 = 0.5*(lam1+fabs(lam1));
                  lp2 = 0.5*(lam2+fabs(lam2));
               
                  R[0][0] = r1[0]; R[1][0] = r1[1];
                  R[0][1] = r2[0]; R[1][1] = r2[1];                  
               
                  det = R[0][0]*R[1][1]-R[1][0]*R[0][1];
                  R_inv[0][0] = 1/det*R[1][1]; R_inv[0][1] = -1/det*R[0][1];
                  R_inv[1][0] = -1/det*R[1][0]; R_inv[1][1] = 1/det*R[0][0];
            
                  tmp[0][0] = lp1*R_inv[0][0]; tmp[0][1] = lp1*R_inv[0][1];
                  tmp[1][0] = lp2*R_inv[1][0]; tmp[1][1] = lp2*R_inv[1][1];

                  C[0][0] = R[0][0]*tmp[0][0]+R[0][1]*tmp[1][0];
                  C[0][1] = R[0][0]*tmp[0][1]+R[0][1]*tmp[1][1];
                  C[1][0] = R[1][0]*tmp[0][0]+R[1][1]*tmp[1][0];
                  C[1][1] = R[1][0]*tmp[0][1]+R[1][1]*tmp[1][1];
 
                  T11(gel,i,j) = C[0][0]-Z(gel,i,j); 
                  T22(gel,i,j) = C[1][1]-Z(gel,i,j);
                  T12(gel,i,j) = C[0][1];  
                 
                  printf("\n C=[%6.5e %6.5e],z = %6.5e",C[0][1],C[1][0],Z(gel,i,j)); 
                  //check again
                  check = (T11(gel,i,j)+Z(gel,i,j))*(T22(gel,i,j)+Z(gel,i,j));
                  printf("\n revised value is %6.5e",check-T12(gel,i,j)*T12(gel,i,j)); 
             }
            } 
          }
       } 
       
       for(i=0; i<3; ++i)
         free(A[i]);
       
       free(A); 
  
       update_ghost_taoz_we(gel,gmax);
       update_ghost_taoz_sn(gel,gmax);
       update_buffer_tao_and_z(gel,gmax);
}

static void update_volume_fraction_zerodelta(
       GEL    *gel,
       int    *gmax)
{
       int     p,q,i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;
       double  h = gel->rgr.h;
       double  dt=gel->dt;
       double  sum1,sum2;

       State         *st;

       update_edge_flux_2ndv1(gel,gmax,dt,h);

       for(i=0; i<gmax[0]; ++i)
       {
          for(j=0; j<gmax[1]; ++j) 
	  {
             st = &(Rect_state(i,j,gel));

              //first store the old values
             for(p=0; p<3; ++p)
             {
                for(q=0; q<3; ++q)
                {
                   st->othn[p][q] = st->thn[p][q];
                   st->oths[p][q] = 1-st->othn[p][q];
                }
             }

             if(i>=lbuf[0] && i<gmax[0]-ubuf[0] && j>=lbuf[1] && j<gmax[1]-ubuf[1])
             {
                //printf("\n cell[%d %d] has Fy=%5.4e %5.4e",i,j,Fy(gel,i-1,j),Fy(gel,i,j));
                Thn(gel,i,j) += -dt*(Fx(gel,i,j)-Fx(gel,i-1,j)+Fy(gel,i,j)-Fy(gel,i,j-1))/h;
                if(Thn(gel,i,j) < 0.0)
                {
                   //printf("\ negative thn[%d %d]=%f",i,j,Thn(gel,i,j));
                   Thn(gel,i,j) = 0.0;
                }
                if(Thn(gel,i,j) > 1.0)
                {
                   printf("\ oversize thn[%d %d]=%f",i,j,Thn(gel,i,j));
                   Thn(gel,i,j) = 0.5;
                }
                Ths(gel,i,j) = 1-Thn(gel,i,j);
             }

	  }
       }
//spatial averaging of Thn
       /*for(j=lbuf[1]; j<gmax[1]-1; ++j)
       {
           sum1 = 0.0;
           for(i=lbuf[0]; i<gmax[0]-1; ++i)
           {
             sum1 += Thn(gel,i,j);
           }
           sum1 = sum1/(gmax[0]-3);

           for(i=lbuf[0]; i<gmax[0]-1; ++i)
           {
              Thn(gel,i,j) = sum1; Ths(gel,i,j) = 1.0-Thn(gel,i,j);
           }
       }*/     
}

static void init_gel_struct(
       GEL    *gel,
       int    argc,
       char   *argv[])
{
       FILE       *file;
       char       *finput;
       char       s[100];
       double     L[2],U[2];
       double     tmp,**array;
       int        pcrd[2],gmax[2];
       int        dim,me,i,j,tc;
       RECT_GRID  *rgr = &(gel->rgr); 
     
       for(i=0; i<argc; ++i)
       {
//specify parallel partition
	  if (strncmp(argv[i],"-pat",4) == 0)
	  {
	     gel->pgmax[0] = atoi(argv[i+1]);
	     gel->pgmax[1] = atoi(argv[i+2]);
          }
//specify input file
	  else if (strncmp(argv[i],"-inp",4) == 0)
	     finput = argv[i+1];
//specify basic outputfile name
	  else if (strncmp(argv[i],"-out",4) == 0)
	     gel->output = argv[i+1];
//how often to make output
          else if (strncmp(argv[i],"-fre",4) == 0)
	  {
	     gel->outputf = atof(argv[i+1]);
             printf("\n outputf = %f",gel->outputf);
	  }
       }
     
       //printf("\n parallel partition is %d %d",gel->pgmax[0],gel->pgmax[1]);
       //printf("\n input file name is %s",finput);
       //printf("\n output file name is %s",gel->output);

       file = fopen(finput,"r");
       if(file==NULL)
       {
          printf("\n WARNING!, no input file specified!");
	  exit(1);
       }
       fscanf(file,"%lf %lf %lf %lf",&(gel->gam),&(gel->xi),&(gel->beta),&(gel->z_0));
       fgets(s,50,file);
       fscanf(file,"%lf %lf",&(gel->mu_n),&(gel->mu_s));
       fgets(s,50,file);
       fscanf(file,"%lf %lf %lf",&(gel->lam_n),&(gel->lam_s),&(gel->lambda));
       fgets(s,50,file);
       
       printf("\n mun=%3.2e, mus=%3.2e,lamn=%f,lams=%f",gel->mu_n,gel->mu_s,gel->lam_n,gel->lam_s);
       printf("\n gam=%f,xi=%f,beta=%f,z0=%f",gel->gam,gel->xi,gel->beta,gel->z_0); 
       
       fgets(s,50,file);
       
       if(strncmp(s,"YES",3)==0)
          gel->diff = YES;
       else
          gel->diff = NO;
       fscanf(file,"%lf %lf %lf",&(gel->delta),&(gel->th_os),&(gel->dtmin));
       fgets(s,50,file);
       fscanf(file,"%lf",&(gel->omega));
       fgets(s,50,file);
       fscanf(file,"%lf %lf",&(gel->stime),&(gel->etime));
       fgets(s,50,file);
       fscanf(file,"%lf",&(gel->rtol));
       fgets(s,50,file);
       fscanf(file,"%d",&(gel->itmax));
       fgets(s,50,file);
//restriction factor for time step
       fscanf(file,"%lf %lf",&(gel->cfln),&(gel->cfls));
       fgets(s,50,file);
       fscanf(file,"%d",&dim);
       fgets(s,50,file);

       fgets(s,50,file);

       if(strncmp(s,"YES",3)==0)
          gel->harmonic = YES;
       else
          gel->harmonic = NO;

       fgets(s,50,file);
       
       if(strncmp(s,"YES",3)==0)
          gel->first_order = YES;
       else
	  gel->first_order = NO;

       fscanf(file,"%d %d",&(gel->clevel),&(gel->c1level));
       fgets(s,50,file);

       fscanf(file,"%d  %d",&(gel->prev),&(gel->post));
       fgets(s,50,file);

       fscanf(file,"%lf %lf",&(gel->epi),&(gel->wi));
       fgets(s,50,file);

       printf("\n clevel=%d %d,pre_post=%d %d,epi=%6.5e, wi=%6.5e, hamonic=%d",
        gel->clevel,gel->c1level,gel->prev,gel->post,gel->epi,gel->wi,gel->harmonic);
       
       for(i=0; i<dim; ++i)
       {
          fscanf(file,"%lf %lf %d",&L[i],&U[i],&gmax[i]);
	  rgr->GL[i] = L[i]; rgr->GU[i]=U[i];
          rgr->ggmax[i] = gmax[i]+3;
	 
	  fgets(s,50,file);
       }

       rgr->h = (U[0]-L[0])/gmax[0];
       rgr->dim = dim;

       fgets(s,50,file);
       
       if(strncmp(s,"GMRES",5)==0)
          gel->gel_system_solver = GMRES_solver;
       else if(strncmp(s,"MGRID",5)==0)
          gel->gel_system_solver = MG_solver;

       fscanf(file,"%lf %lf %lf %lf %lf",
         &(gel->s1),&(gel->s2),&(gel->s3),&(gel->s4),&(gel->s5));

       printf("\n spring constant is %6.5e %6.5e %6.5e %6.5e %6.5e",gel->s1,gel->s2,gel->s3,gel->s4,gel->s5);

       fgets(s,50,file);
       fscanf(file,"%lf %lf" ,&(gel->thn0),&(gel->thn1));

       printf("\n dtmin = %f, thn = %f %f",gel->dtmin,gel->thn0,gel->thn1);

       fgets(s,50,file);

       fgets(s,50,file);
       if(strncmp(s,"YES",3)==0)
          gel->cv = YES;
       else
          gel->cv = NO;

       printf("\n cv = %d",gel->cv);
 
       MPI_Comm_rank(PETSC_COMM_WORLD,&me);

       gel->my_id=me;

       if(gel->my_id == 0)
         gel->file = fopen(gel->output,"w");

       find_pp_grid(me,gel->pgmax,pcrd,dim);

       for(i=0; i<dim; ++i)
       {
          rgr->ubuf[i]=1;
	  rgr->lbuf[i] = 2;
	  
	  gel->pcrd[i] = pcrd[i];
          tmp = (U[i]-L[i])/(gel->pgmax[i]);
	  rgr->L[i]=L[i]+pcrd[i]*tmp;
	  rgr->U[i]=L[i]+(pcrd[i]+1)*tmp;
          rgr->gmax[i]=gmax[i]/(gel->pgmax[i]) + rgr->lbuf[i] + rgr->ubuf[i];
       }
       
       //printf("\n node %d as L=[%f %f],U=[%f %f],gmax=[%d %d]",me,rgr->L[0],rgr->L[1],rgr->U[0],
             //rgr->U[1],rgr->gmax[0],rgr->gmax[1]);
       gel->a_s = 2*(gel->mu_s)+(gel->lam_s);
       gel->a_n = 2*(gel->mu_n)+(gel->lam_n);
       
       tc = 1;  //total number of local cells
       for(i=0; i<2; ++i)
          tc *= gel->rgr.gmax[i];
       
       gel->st = (State *)malloc(tc*sizeof(State));
       if(gel->st == NULL)
       {
           printf("\n could not allocate memory!");
	   exit(1);
       }
       fclose(file);
       return;
}

static void update_edge_flux_1st(
       GEL     *gel,
       int     *gmax,
       double  dt,
       double  h)
{
       int     i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;

       printf("\n use first order flux");

       for(i=lbuf[0]-1; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]-1; j<gmax[1]-ubuf[1]; ++j)
          {
             Fx(gel,i,j) = (Un(gel,i,j) > 0) ? Thn(gel,i,j) : Thn(gel,i+1,j);
	     Fy(gel,i,j) = (Vn(gel,i,j) > 0) ? Thn(gel,i,j) : Thn(gel,i,j+1);

	     Fx(gel,i,j) *= Un(gel,i,j);
	     Fy(gel,i,j) *= Vn(gel,i,j);
	  }
       }
       return;
}


static void update_taoz_flux(
       GEL     *gel,
       int     *gmax)
{
       int     m,n,p,i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;
 
       double   h = gel->rgr.h;
       double   q[4],B[4][4];
       double   tmp[4],a[4],bq[4],betaq[4];
       double   qxl[4],qxr[4],qxc[4],qx[4];
       double   xold[4],qyt[4],qyb[4],qyc[4],qy[4];
       double   aold[4],ql[4],qr[4],qb[4],qt[4]; //all the neighbors
       double   v1,v2,uc,vc; //cell centered velocity
       double   div_un,ux,uy,vx,vy; 
       double   beta = gel->beta*PI; 
       double   dt = gel->dt;
       double   uq_y[4],uq_x[4]; //transverse derivative
       double   lambda = gel->lambda;
 
       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
             //calculate half step value on right edge
              tmp[0] = tmp[1] = tmp[2] = tmp[3] = 0.0;

              xold[0] = q[0] = T11(gel,i,j); betaq[0] = beta*T11(gel,i,j);
              xold[1] = q[1] = T12(gel,i,j); betaq[1] = beta*T12(gel,i,j);
              xold[2] = q[2] = T22(gel,i,j); betaq[2] = beta*T22(gel,i,j);
              xold[3] = q[3] = Z(gel,i,j);   betaq[3] = beta*Z(gel,i,j);

              ql[0] = T11(gel,i-1,j); qr[0] = T11(gel,i+1,j);
              qb[0] = T11(gel,i,j-1); qt[0] = T11(gel,i,j+1);

              ql[1] = T12(gel,i-1,j); qr[1] = T12(gel,i+1,j);
              qb[1] = T12(gel,i,j-1); qt[1] = T12(gel,i,j+1);

              ql[2] = T22(gel,i-1,j); qr[2] = T22(gel,i+1,j);
              qb[2] = T22(gel,i,j-1); qt[2] = T22(gel,i,j+1);

              ql[3] = Z(gel,i-1,j); qr[3] = Z(gel,i+1,j);
              qb[3] = Z(gel,i,j-1); qt[3] = Z(gel,i,j+1);

              ux = (Un(gel,i,j)-Un(gel,i-1,j))/h;
              v1 = 0.5*(Un(gel,i,j+1)+Un(gel,i-1,j+1));
              v2 = 0.5*(Un(gel,i,j-1)+Un(gel,i-1,j-1));
              uy = (v1-v2)/(2*h);
             
              vy = (Vn(gel,i,j)-Vn(gel,i,j-1))/h;
              v1 = 0.5*(Vn(gel,i+1,j)+Vn(gel,i+1,j-1));
              v2 = 0.5*(Vn(gel,i-1,j)+Vn(gel,i-1,j-1));
              vx = (v1-v2)/(2*h); 
             
              div_un = ux+vy; //divergence of the network velocity
 
              uc = 0.5*(Un(gel,i,j)+Un(gel,i-1,j));
              vc = 0.5*(Vn(gel,i,j)+Vn(gel,i,j-1));

              B[0][0] = 2*ux;   B[0][1] = 2*uy;
              B[0][2] = 0;      B[0][3] = 2*ux;

              B[1][0] = vx;     B[1][1] = ux+vy;
              B[1][2] = uy;     B[1][3] = vx+uy;
 
              B[2][0] = 0;      B[2][1] = 2*vx;
              B[2][2] = 2*vy;   B[2][3] = 2*vy;
  
              B[3][0] = 0;      B[3][1] = 0;
              B[3][2] = 0;      B[3][3] = 0;
                       
              for(m=0; m<4; ++m)
                 for(n=0; n<4; ++n)
                    tmp[m] += B[m][n]*xold[n];
             
             
//first on east and west edge
              
              for(p=0; p<4; ++p)
              {
                 qxl[p] = 2*(q[p]-ql[p])/h; qxr[p] = 2*(qr[p]-q[p])/h; 
                 qxc[p] = (qr[p]-ql[p])/(2*h); 
                 qx[p] = minmod(qxl[p],qxr[p],qxc[p]); 
             
                 if(vc > 0)
                    qy[p] = (q[p]-qb[p])/h;  
                 else
                    qy[p] = (qt[p]-q[p])/h;

                 VTL(gel,i,j)[p] = q[p]+(0.5*h-0.5*dt*uc)*qx[p]-0.5*dt*q[p]*ux+0.5*dt*(tmp[p]-betaq[p]); 
                                    -0.5*dt*(vy*q[p]+vc*qy[p]);  
                 VTR(gel,i-1,j)[p] = q[p]+(-0.5*h-0.5*dt*uc)*qx[p]-0.5*dt*q[p]*ux+0.5*dt*(tmp[p]-betaq[p]);
                                    -0.5*dt*(vy*q[p]+vc*qy[p]);   
                 if(p == 0 || p == 2)
                 { 
                    VTL(gel,i,j)[p] += 0.5*dt*lambda*div_un;
                    VTR(gel,i-1,j)[p] += 0.5*dt*lambda*div_un;
                 } 
//then on north and south edge
 
                 qyb[p] = 2*(q[p]-qb[p])/h; qyt[p] = 2*(qt[p]-q[p])/h; 
                 qyc[p] = (qt[p]-qb[p])/(2*h); 
                 qy[p] = minmod(qyb[p],qyt[p],qyc[p]); 
            
                 if(uc > 0)
                    qx[p] = (q[p]-ql[p])/h;
                 else
                    qx[p] = (qr[p]-q[p])/h;
 
                 VTB(gel,i,j)[p] = q[p]+(0.5*h-0.5*dt*vc)*qy[p]-0.5*dt*q[p]*vy+0.5*dt*(tmp[p]-betaq[p]);
                                    -0.5*dt*(ux*q[p]+uc*qx[p]);  
                 VTT(gel,i,j-1)[p] = q[p]+(-0.5*h-0.5*dt*vc)*qy[p]-0.5*dt*q[p]*vy+0.5*dt*(tmp[p]-betaq[p]);
                                    -0.5*dt*(ux*q[p]+vc*qx[p]);   
                 if(p == 0 || p == 2)
                 {
                    VTB(gel,i,j)[p] += 0.5*dt*lambda*div_un;
                    VTT(gel,i,j-1)[p] += 0.5*dt*lambda*div_un;
                 }

             }
          }
       }

       //update_ghost_testate_sn(gel,gmax);
       update_ghost_testate_we(gel,gmax);
       update_buffer_testate(gel,gmax);

       for(i=lbuf[0]-1; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]-1; j<gmax[1]-ubuf[1]; ++j)
          {
	     for(p=0; p<4; ++p)
             {
                FTx(gel,i,j)[p] = (Un(gel,i,j) > 0) ? VTL(gel,i,j)[p] : VTR(gel,i,j)[p];
	        FTy(gel,i,j)[p] = (Vn(gel,i,j) > 0) ? VTB(gel,i,j)[p] : VTT(gel,i,j)[p];

                if(gel->step == 0) //first step of updating volume fraction
                {
                  FTx(gel,i,j)[p] *= Un(gel,i,j);
                  FTy(gel,i,j)[p] *= Vn(gel,i,j);
                }
                else
                {
                   FTx(gel,i,j)[p] *= Un(gel,i,j)+(Un(gel,i,j)-OUn(gel,i,j))*dt/(2*gel->odt); //   0.5*(3*Un(gel,i,j)-OUn(gel,i,j));
                   FTy(gel,i,j)[p] *= Vn(gel,i,j)+(Vn(gel,i,j)-OVn(gel,i,j))*dt/(2*gel->odt); //   0.5*(3*Vn(gel,i,j)-OVn(gel,i,j));
                }
             }
          }
       }

       return;
}

//first version of colella's method to calculate flux
static void update_edge_flux_2ndv1(
       GEL     *gel,
       int     *gmax,
       double  dt,
       double  h)
{
       int     i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;

       double  hh = h*h;
       double  Ux,Vy,Tx,Ty; //partial derivatives 
       double  Uc,Vc; //cell centered velocity
       double  Txc,Txl,Txr; 
       double  Tyc,Tyl,Tyu,unh,vnh;

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
             Uc = 0.5*(Un(gel,i,j)+Un(gel,i-1,j));
             Vc = 0.5*(Vn(gel,i,j)+Vn(gel,i,j-1));

	     Ux = (Un(gel,i,j)-Un(gel,i-1,j))/h;
	     Vy = (Vn(gel,i,j)-Vn(gel,i,j-1))/h;

	     Txc = (Thn(gel,i+1,j)-Thn(gel,i-1,j))/(2*h);
	     Txl = 2*(Thn(gel,i,j)-Thn(gel,i-1,j))/h;
	     Txr = 2*(Thn(gel,i+1,j)-Thn(gel,i,j))/h;

	     Tx = minmod(Txc,Txl,Txr);
             
	     //upwinding for transverse derivative
	     if(Vc > 0)
	        Ty = (Thn(gel,i,j)-Thn(gel,i,j-1))/h;
             else
	        Ty = (Thn(gel,i,j+1)-Thn(gel,i,j))/h;

	     VL(gel,i,j) = Thn(gel,i,j)+0.5*dt*(-Tx*Uc-Ux*Thn(gel,i,j)-Thn(gel,i,j)*Vy-Vc*Ty) + 0.5*h*Tx;

             //calculate half step value on left edge
	      
	     VR(gel,i-1,j) = Thn(gel,i,j)+0.5*dt*(-Tx*Uc-Ux*Thn(gel,i,j)-Thn(gel,i,j)*Vy-Vc*Ty) - 0.5*h*Tx;

	      
             //calculate half step value on top edge

	     Tyc = (Thn(gel,i,j+1)-Thn(gel,i,j-1))/(2*h);
	     Tyl = 2*(Thn(gel,i,j)-Thn(gel,i,j-1))/h;
	     Tyu = 2*(Thn(gel,i,j+1)-Thn(gel,i,j))/h;

	     Ty = minmod(Tyc,Tyl,Tyu);
             
	     //upwinding for transverse derivative
	     
	     if(Uc > 0)
	        Tx = (Thn(gel,i,j)-Thn(gel,i-1,j))/h;
             else
	        Tx = (Thn(gel,i+1,j)-Thn(gel,i,j))/h;

	     VB(gel,i,j) = Thn(gel,i,j)+0.5*dt*(-Tx*Uc-Ux*Thn(gel,i,j)-Thn(gel,i,j)*Vy-Vc*Ty) + 0.5*h*Ty;
	     VT(gel,i,j-1) = Thn(gel,i,j)+0.5*dt*(-Tx*Uc-Ux*Thn(gel,i,j)-Thn(gel,i,j)*Vy-Vc*Ty) - 0.5*h*Ty; 
	  }
       }

       //update_ghost_estate_sn(gel,gmax);
       update_ghost_estate_we(gel,gmax);
 
       for(i=lbuf[0]-1; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]-1; j<gmax[1]-ubuf[1]; ++j)
          {
             if(gel->step == 0)
             {
               unh = Un(gel,i,j); vnh = Vn(gel,i,j);
             }
             else
             {
               unh = 0.5*(3*Un(gel,i,j)-OUn(gel,i,j));
               vnh = 0.5*(3*Vn(gel,i,j)-OVn(gel,i,j));
             }
             Fx(gel,i,j) = (unh > 0) ? VL(gel,i,j) : VR(gel,i,j);
	     Fy(gel,i,j) = (vnh > 0) ? VB(gel,i,j) : VT(gel,i,j);

             Fx(gel,i,j) *= unh;
             Fy(gel,i,j) *= vnh;
	  }
       }
       return;
}

//second version of colella's method to calculate flux
static void update_edge_flux_2ndv2(
       GEL     *gel,
       int     *gmax,
       double  dt,
       double  h)
{
       int     i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;

       double  hh = h*h;
       double  Ux,Vy,Tx,Ty; //partial derivatives 
       double  Uc,Vc; //cell centered velocity
       double  Txc,Txl,Txr; 
       double  Tyc,Tyl,Tyu;

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
             //calculate half step value on right edge

             Uc = 0.5*(Un(gel,i,j)+Un(gel,i-1,j));
             Vc = 0.5*(Vn(gel,i,j)+Vn(gel,i,j-1));

	     Ux = (Un(gel,i,j)-Un(gel,i-1,j))/h;
	     Vy = (Vn(gel,i,j)-Vn(gel,i,j-1))/h;

	     Txc = (Thn(gel,i+1,j)-Thn(gel,i-1,j))/(2*h);
	     Txl = 2*(Thn(gel,i,j)-Thn(gel,i-1,j))/h;
	     Txr = 2*(Thn(gel,i+1,j)-Thn(gel,i,j))/h;

	     Tx = minmod(Txc,Txl,Txr);
             
	     VL(gel,i,j) = Thn(gel,i,j)+0.5*dt*(-Tx*Uc-Ux*Thn(gel,i,j)) + 0.5*h*Tx;
	     VR(gel,i-1,j) = Thn(gel,i,j)+0.5*dt*(-Tx*Uc-Ux*Thn(gel,i,j)) - 0.5*h*Tx;

	      
             //calculate half step value on top edge

	     Tyc = (Thn(gel,i,j+1)-Thn(gel,i,j-1))/(2*h);
	     Tyl = 2*(Thn(gel,i,j)-Thn(gel,i,j-1))/h;
	     Tyu = 2*(Thn(gel,i,j+1)-Thn(gel,i,j))/h;

	     Ty = minmod(Tyc,Tyl,Tyu);
             
	     VB(gel,i,j) = Thn(gel,i,j)+0.5*dt*(-Thn(gel,i,j)*Vy-Vc*Ty) + 0.5*h*Ty;
	     VT(gel,i,j-1) = Thn(gel,i,j)+0.5*dt*(-Thn(gel,i,j)*Vy-Vc*Ty) - 0.5*h*Ty; 
	  }
       }

       for(i=lbuf[0]-1; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]-1; j<gmax[1]-ubuf[1]; ++j)
          {
	     if(i==lbuf[0]-1)
	        VL(gel,i,j) = VL(gel,gmax[0]-ubuf[0]-1,j);
	     if(i==gmax[0]-ubuf[0]-1)
	        VR(gel,i,j) = VR(gel,lbuf[0]-1,j);
           
             if(j==lbuf[1]-1)
                VB(gel,i,j) = VB(gel,i,gmax[1]-ubuf[1]-1);
             if(j==gmax[1]-ubuf[1]-1)
                VT(gel,i,j) = VT(gel,i,lbuf[1]-1); 
  
             RPX(gel,i,j) = (Un(gel,i,j) > 0) ? VL(gel,i,j) : VR(gel,i,j);
	     RPY(gel,i,j) = (Vn(gel,i,j) > 0) ? VB(gel,i,j) : VT(gel,i,j);
           }
       }

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
             Tx = (Un(gel,i,j)*RPX(gel,i,j)-Un(gel,i-1,j)*RPX(gel,i-1,j))/h; 
	     Ty = (Vn(gel,i,j)*RPY(gel,i,j)-Vn(gel,i,j-1)*RPY(gel,i,j-1))/h;

	     VL(gel,i,j) -=  0.5*dt*Ty; VR(gel,i-1,j) -= 0.5*dt*Ty;  
	     VB(gel,i,j) -= 0.5*dt*Tx;  VT(gel,i,j-1) -= 0.5*dt*Tx;
	  }
       }

       for(i=lbuf[0]-1; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]-1; j<gmax[1]-ubuf[1]; ++j)
          {
	     if(i==lbuf[0]-1)
	        VL(gel,i,j) = VL(gel,gmax[0]-ubuf[0]-1,j);
	     if(i==gmax[0]-ubuf[0]-1)
	        VR(gel,i,j) = VR(gel,lbuf[0]-1,j);
             
             if(j==lbuf[1]-1)
                VB(gel,i,j) = VB(gel,i,gmax[1]-ubuf[1]-1);
             if(j==gmax[1]-ubuf[1]-1)
                VT(gel,i,j) = VT(gel,i,lbuf[1]-1);

             Fx(gel,i,j) = (Un(gel,i,j) > 0) ? VL(gel,i,j) : VR(gel,i,j);
	     Fy(gel,i,j) = (Vn(gel,i,j) > 0) ? VB(gel,i,j) : VT(gel,i,j);

	     Fx(gel,i,j) *= Un(gel,i,j);
	     Fy(gel,i,j) *= Vn(gel,i,j);
	  }
       }
       return;
}

static double minmod(
       double  a,
       double  b,
       double  c)
{
      double  tmp;

      if(a*b < 0 || a*c < 0) 
         return 0.0;
      else
      {
         tmp = (fabs(a) < fabs(b)) ? a : b;
         tmp = (fabs(tmp) < fabs(c)) ? tmp : c; 

	 return tmp;
      }
}

extern void init_gel_state(
       GEL    *gel,
       int    *gmax)
{
       int        i,j;
       double     tmp,xc,yc,r;
       double     wi = gel->wi;

//value of valence
       gel->zcl = -1.0;
       gel->zs = 1.0; gel->zca = 2.0; gel->zt = 0.1; 

       gel->e1 = -45.0; gel->e2 = 25.0;
       gel->e3 = -0.5; gel->e4 = 0.0;

       gel->ds = 2.5e3; gel->dca = 2.5e3; gel->dcl = 2.5e3; 
  
       gel->kcon = 1.0e6; gel->kcoff = 1.0e3; 
       gel->kson = 5.0e6; gel->ksoff = 5.0e2; 
 
       for(i=0; i<gmax[0]; ++i)
       {
           xc = Rect_coord(i,gel->rgr,0);

           for(j=0; j<gmax[1]; ++j)
	   {
              yc = Rect_coord(j,gel->rgr,1);

              r = sqrt(xc*xc+yc*yc); 

              Ths(gel,i,j) = 0.25*tanh(wi*(yc-5.0))+0.75;
	      Thn(gel,i,j) = 1.0-Ths(gel,i,j);
             
//ion concentration
              OBC2(gel,i,j) = BC2(gel,i,j) = 0.01-0.01*tanh(wi*(yc-5.0));  
	      OBCA(gel,i,j) = BCA(gel,i,j)  = 0.0025-0.0025*tanh(wi*(yc-5.0)); 
              OBS(gel,i,j) = BS(gel,i,j)  = 0.0005-0.0005*tanh(wi*(yc-5.0)); 
              BH(gel,i,j)  = 0.0;

              OS(gel,i,j) = S(gel,i,j) = 0.01*tanh(wi*(yc-5.0))+0.01; 
              OCA(gel,i,j) = CA(gel,i,j) = 0.0002*tanh(wi*(yc-5.0)) + 0.0008; 

//choose chloride concentration to maintain neutrality 
              tmp = Ths(gel,i,j)*S(gel,i,j)+2.0*Ths(gel,i,j)*CA(gel,i,j);
              tmp += BS(gel,i,j) + 2.0*(BCA(gel,i,j)+BC2(gel,i,j));
 
              tmp -= gel->zt*Thn(gel,i,j);  
              OCL(gel,i,j) = CL(gel,i,j) = tmp/Ths(gel,i,j); 

              Pres(gel,i,j)=0.0; Phi(gel,i,j) = 0.0;
	      Us(gel,i,j)=Un(gel,i,j)=0.0;
	      Vs(gel,i,j)=Vn(gel,i,j)=0.0;
	      OUn(gel,i,j) = OUs(gel,i,j) = 0.0;
              OVn(gel,i,j) = OVs(gel,i,j) = 0.0;

              OT11(gel,i,j) = T11(gel,i,j) = 0.0;
              OT12(gel,i,j) = T12(gel,i,j) = 0.0;
              OT22(gel,i,j) = T22(gel,i,j) = 0.0;

              Un_indx(gel,i,j) = Us_indx(gel,i,j) = -1;
	      Vn_indx(gel,i,j) = Vs_indx(gel,i,j) = -1;
	      P_indx(gel,i,j) = -1;
          }
       }

       for(i=0; i<gmax[0]; ++i)
           for(j=0; j<gmax[1]; ++j)
              Z(gel,i,j) = gel->z_0; //a_0*Thn(gel,i,j)*Thn(gel,i,j)/gel->beta;
 
       gel->oodt = gel->odt = gel->dt = 0.0;
}

static void increase_network_nozero(
       GEL    *gel,
       int    *gmax)
{
       int        p,q,i,j;
       double     x,y,length;
       State      *st;

       for(i=0; i<gmax[0]; ++i)
       {
           for(j=0; j<gmax[1]; ++j)
	   {
              st = &(Rect_state(i,j,gel));

              for(p=0; p<3; ++p)
              {
                for(q=0; q<3; ++q)
                {
                   st->thn[p][q] += gel->epi; 
                   st->ths[p][q] = 1-st->thn[p][q];
                }
              } 
	   }
       }
}

static void decrease_network_zero(
       GEL    *gel,
       int    *gmax)
{
       int        p,q,i,j;
       double     x,y,length;
       State      *st;

       for(i=0; i<gmax[0]; ++i)
       {
           for(j=0; j<gmax[1]; ++j)
           {
              st = &(Rect_state(i,j,gel));

              for(p=0; p<3; ++p)
              {
                for(q=0; q<3; ++q)
                {
                   st->thn[p][q] -= gel->epi;
                   st->ths[p][q] = 1-st->thn[p][q];
                }
              }
           }
       }
}
/*update west and east ghost cell states for periodic boundary conditions*/
extern void update_ghost_cell_states_we(
       GEL    *gel,
       int    *gmax)
{
       double  *bfs,*bfr;
       int     *lbuf=gel->rgr.lbuf;
       int     *ubuf=gel->rgr.ubuf;

       int     *pcrd = gel->pcrd;
       int     Ny = gmax[1] ;
       int     his_id,ss,count,tag,i,j;

       if(gel->pgmax[0] != 1) //more than two subdomain need communicate
       {
          MPI_Status status;
       
          bfs = (double*)malloc(8*Ny*sizeof(double));
          bfr = (double*)malloc(8*Ny*sizeof(double));
 
          if(pcrd[0] == 0)
          {
            //pass to right boundary and receive left boundary
	     ss = lbuf[0];
	     count=0;
	     for(i=0; i<gmax[1]; i++)
	     {
	        bfs[count]      = Thn(gel,ss,i);
                bfs[count+Ny]   = Un(gel,ss,i);
	        bfs[count+2*Ny] = Us(gel,ss,i);
	        bfs[count+3*Ny] = Vn(gel,ss,i);
	        bfs[count+4*Ny] = Vs(gel,ss,i);
	        bfs[count+5*Ny] = Pres(gel,ss,i);
	        count++;
	     }

	     pcrd[0] += (gel->pgmax[0]-1);
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] -= (gel->pgmax[0]-1);

	     tag = 17; 
	     MPI_Sendrecv(bfs,6*Ny,MPI_DOUBLE,his_id,tag,
	       bfr,8*Ny,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
             count=0;
	     for (i=0; i<gmax[1]; i++)
	     {
	        Thn(gel,ss-1,i) = bfr[count];
	        Ths(gel,ss-1,i) = 1-Thn(gel,ss-1,i); 
	        
		Un(gel,ss-2,i)  = bfr[count+Ny];
		Us(gel,ss-2,i)  = bfr[count+2*Ny];
	        Un(gel,ss-1,i)   = bfr[count+3*Ny];
	        Us(gel,ss-1,i)   = bfr[count+4*Ny];
	        Vn(gel,ss-1,i)   = bfr[count+5*Ny];
	        Vs(gel,ss-1,i)   = bfr[count+6*Ny];
	        Pres(gel,ss-1,i) = bfr[count+7*Ny];
	        count++;
	     } 
          } 
       
          if(pcrd[0] == gel->pgmax[0]-1)
          {
	    /*pass to  left bound and also receive right bound*/
	     ss = gmax[0]-ubuf[0]-1;
	     count=0;
	     for(i=0; i<gmax[1]; i++)
	     {
	        bfs[count]      = Thn(gel,ss,i);
                
		bfs[count+Ny]   = Un(gel,ss-1,i);
		bfs[count+2*Ny] = Us(gel,ss-1,i);
		bfs[count+3*Ny] = Un(gel,ss,i);
	        bfs[count+4*Ny] = Us(gel,ss,i);
	        bfs[count+5*Ny] = Vn(gel,ss,i);
	        bfs[count+6*Ny] = Vs(gel,ss,i);
	        bfs[count+7*Ny] = Pres(gel,ss,i);
	        count++;
	     }

	     pcrd[0] -= (gel->pgmax[0]-1);
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] += (gel->pgmax[0]-1);

	     tag = 17; 
	     MPI_Sendrecv(bfs,8*Ny,MPI_DOUBLE,his_id,tag,
	       bfr,6*Ny,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
             count=0;
	     for (i=0; i<gmax[1]; i++)
	     {
	        Thn(gel,ss+1,i) = bfr[count];
	        Ths(gel,ss+1,i) = 1-Thn(gel,ss+1,i);
	     
	        Un(gel,ss+1,i)   = bfr[count+Ny];
	        Us(gel,ss+1,i)   = bfr[count+2*Ny];
	        Vn(gel,ss+1,i)   = bfr[count+3*Ny];
	        Vs(gel,ss+1,i)   = bfr[count+4*Ny];
	        Pres(gel,ss+1,i) = bfr[count+5*Ny];
	        count++;
	     } 
          } 
          free(bfs); free(bfr);
       }
       
       else  //only one subdomain in x-direction
       {
          for(i=0; i<gmax[1]; ++i)
          {
             //left side
	     Thn(gel,lbuf[0]-1,i) = Thn(gel,lbuf[0],i);  
             Ths(gel,lbuf[0]-1,i) = 1-Thn(gel,lbuf[0]-1,i);

             Phi(gel,lbuf[0]-1,i) = Phi(gel,lbuf[0],i);
             S(gel,lbuf[0]-1,i) = S(gel,lbuf[0],i);
             CL(gel,lbuf[0]-1,i) = CL(gel,lbuf[0],i); 
             CA(gel,lbuf[0]-1,i) = CA(gel,lbuf[0],i);

             ES(gel,lbuf[0]-1,i) = ES(gel,lbuf[0],i);
             ECL(gel,lbuf[0]-1,i) = ECL(gel,lbuf[0],i);
             ECA(gel,lbuf[0]-1,i) = ECA(gel,lbuf[0],i);

             OS(gel,lbuf[0]-1,i) = OS(gel,lbuf[0],i);
             OCL(gel,lbuf[0]-1,i) = OCL(gel,lbuf[0],i);
             OCA(gel,lbuf[0]-1,i) = OCA(gel,lbuf[0],i);

             BS(gel,lbuf[0]-1,i) = BS(gel,lbuf[0],i);
             BCA(gel,lbuf[0]-1,i) = BCA(gel,lbuf[0],i);
             BC2(gel,lbuf[0]-1,i) = BC2(gel,lbuf[0],i);

             EBS(gel,lbuf[0]-1,i) =  EBS(gel,lbuf[0],i);
             EBCA(gel,lbuf[0]-1,i) = EBCA(gel,lbuf[0],i);
             EBC2(gel,lbuf[0]-1,i) = EBC2(gel,lbuf[0],i);

	     Un(gel,lbuf[0]-1,i)   = 0.0;
	     Us(gel,lbuf[0]-1,i)   = 0.0;
	     Vn(gel,lbuf[0]-1,i)   = Vn(gel,lbuf[0],i);
	     Vs(gel,lbuf[0]-1,i)   = Vs(gel,lbuf[0],i);
	     Pres(gel,lbuf[0]-1,i) = Pres(gel,lbuf[0],i);

	  //right side
	     Thn(gel,gmax[0]-ubuf[0],i) = Thn(gel,gmax[0]-ubuf[0]-1,i);
	     Ths(gel,gmax[0]-ubuf[0],i) = 1-Thn(gel,gmax[0]-ubuf[0],i);
        
             Phi(gel,gmax[0]-ubuf[0],i) = Phi(gel,gmax[0]-ubuf[0]-1,i); 
             S(gel,gmax[0]-ubuf[0],i) = S(gel,gmax[0]-ubuf[0]-1,i);
             CL(gel,gmax[0]-ubuf[0],i) = CL(gel,gmax[0]-ubuf[0]-1,i);
             CA(gel,gmax[0]-ubuf[0],i) = CA(gel,gmax[0]-ubuf[0]-1,i);

             ES(gel,gmax[0]-ubuf[0],i) =  ES(gel,gmax[0]-ubuf[0]-1,i);
             ECL(gel,gmax[0]-ubuf[0],i) = ECL(gel,gmax[0]-ubuf[0]-1,i);
             ECA(gel,gmax[0]-ubuf[0],i) = ECA(gel,gmax[0]-ubuf[0]-1,i);

             OS(gel,gmax[0]-ubuf[0],i) = OS(gel,gmax[0]-ubuf[0]-1,i);
             OCL(gel,gmax[0]-ubuf[0],i) = OCL(gel,gmax[0]-ubuf[0]-1,i);
             OCA(gel,gmax[0]-ubuf[0],i) = OCA(gel,gmax[0]-ubuf[0]-1,i);

             BS(gel,gmax[0]-ubuf[0],i) =  BS(gel,gmax[0]-ubuf[0]-1,i);
             BCA(gel,gmax[0]-ubuf[0],i) = BCA(gel,gmax[0]-ubuf[0]-1,i);
             BC2(gel,gmax[0]-ubuf[0],i) = BC2(gel,gmax[0]-ubuf[0]-1,i);
          
             EBS(gel,gmax[0]-ubuf[0],i) =  EBS(gel,gmax[0]-ubuf[0]-1,i);
             EBCA(gel,gmax[0]-ubuf[0],i) = EBCA(gel,gmax[0]-ubuf[0]-1,i);
             EBC2(gel,gmax[0]-ubuf[0],i) = EBC2(gel,gmax[0]-ubuf[0]-1,i);
 
	     Un(gel,gmax[0]-ubuf[0]-1,i)   = 0.0;
	     Us(gel,gmax[0]-ubuf[0]-1,i)   = 0.0;
	     Vn(gel,gmax[0]-ubuf[0],i)   = Vn(gel,gmax[0]-ubuf[0]-1,i);
	     Vs(gel,gmax[0]-ubuf[0],i)   = Vs(gel,gmax[0]-ubuf[0]-1,i);
	     Pres(gel,gmax[0]-ubuf[0],i) = Pres(gel,gmax[0]-ubuf[0]-1,i);
          }
       }
       return;
}

static void interpolate_coarse_thn(
       GEL     *gel,
       int     fac,
       char    *fname)
{
       int          i,j,m,k;
       int          p,q,pf,qf;

       int          gmax[2];
       int          *lbuf=gel->rgr.lbuf;
       int          *ubuf=gel->rgr.ubuf;
       double       ivalue;
       State        *stn[4];
       FILE         *file;

       file = fopen(fname,"w");

       gmax[0] = lbuf[0]+ubuf[0]+(gel->rgr.gmax[0]-lbuf[0]-ubuf[0])/fac;
       gmax[1] = lbuf[1]+ubuf[1]+(gel->rgr.gmax[1]-lbuf[1]-ubuf[1])/fac;
       
       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
	  {
		if(fac != 1)
		{
		   p=fac*(i-lbuf[0])+lbuf[0]; q=fac*(j-lbuf[1])+lbuf[1]; //block indx of lower left corner of fine grid
	           pf = p+fac/2-1; qf = q+fac/2-1;
                   ivalue = 0.25*(Thn(gel,pf,qf)+Thn(gel,pf+1,qf)+Thn(gel,pf,qf+1)+Thn(gel,pf+1,qf+1));
	           fprintf(file,"%d %d %6.5e \n",i,j,ivalue);
	        }
		else
		{
		   ivalue = Thn(gel,i,j);
		   fprintf(file,"%d %d %6.5e \n",i,j,ivalue);
	        }
	  }
      }
      fclose(file);
}


static void update_buffer_estate(
	GEL    *gel,
	int    *gmax)
{
       double  *bfs,*bfr;
       int     *lbuf=gel->rgr.lbuf;
       int     *ubuf=gel->rgr.ubuf;

       int     *pcrd = gel->pcrd;
       int     Nx = (gmax[0]-lbuf[0]-ubuf[0]);
       int     Ny = (gmax[1]-lbuf[1]-ubuf[1]) ;
       int     his_id,ss,count,tag,i,j;

       MPI_Status status;
      
       bfs = malloc(Ny*sizeof(double));
       bfr = malloc(Ny*sizeof(double));

       if(pcrd[0]%2 == 0)
       {
	    /*first pass to  right and also receive righ side*/
	  if (pcrd[0] < gel->pgmax[0]-1)
	  {
	     count = 0;

             ss = gmax[0]-ubuf[0]-1;

	     for(i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	     {
		bfs[count]    = VL(gel,ss,i);
		count++;
	     }
	     
	     pcrd[0] += 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] -= 1;

	     tag = 9; 
	     MPI_Sendrecv(bfs,Ny,MPI_DOUBLE,his_id,tag,bfr,Ny,MPI_DOUBLE,
	          his_id,tag,PETSC_COMM_WORLD,&status);
	
             count=0;
	     
	     for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	     {
	        VR(gel,ss,i)   = bfr[count];
	        count++;
	     } 
	  }
            
	   /*then pass to left and receive left side*/
          if (pcrd[0] > 0)
          {
              pcrd[0] -= 1;
              his_id = domain_id(pcrd,gel->pgmax,2);
              pcrd[0] += 1;

              ss = lbuf[0]-1;
	      count = 0;

	      for(i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	      {
	         bfs[count]     = VR(gel,ss,i);
                 count++;
	      }

	      tag = 10;
              MPI_Sendrecv(bfs,Ny,MPI_DOUBLE,his_id,tag,bfr,Ny,MPI_DOUBLE,
	            his_id,tag,PETSC_COMM_WORLD,&status);

              count=0;
	      for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	      {
		 VL(gel,ss,i)   = bfr[count];
	         count++;
	      }
	  }        
       }
	
       if(pcrd[0]%2 != 0)
       {
            /*first receive left side and pass to left*/
          if (pcrd[0] > 0)
          {
	     pcrd[0] -= 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] += 1;
	     ss = lbuf[0]-1;
	     
	     tag = 9;
	     count=0;
	     for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	     {
	        bfs[count]      = VR(gel,ss,i);
		count++;
	     }
	     MPI_Sendrecv(bfs,Ny,MPI_DOUBLE,his_id,tag,bfr,Ny,MPI_DOUBLE,
	          his_id,tag,PETSC_COMM_WORLD,&status);
	     
	     count=0;
	     for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	     {
		VL(gel,ss,i) = bfr[count];
		count++;
	     }
             
	  }
	       /*then receive right side and pass to right*/
          if (pcrd[0] < gel->pgmax[0]-1)
	  {
	     pcrd[0] += 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] -= 1;

             ss = gmax[0]-ubuf[0]-1;

	     tag = 10;
	     count=0;
	     for (i=lbuf[1] ;i<gmax[1]-ubuf[1]; i++)
             {
                bfs[count]      = VL(gel,ss,i);
	        count++;
	     } 
	     MPI_Sendrecv(bfs,Ny,MPI_DOUBLE,his_id,tag,bfr,Ny,MPI_DOUBLE,
	         his_id,tag,PETSC_COMM_WORLD,&status);
             
	     count=0;
             for (i=lbuf[1] ;i<gmax[1]-ubuf[1]; i++)
	     {
	        VR(gel,ss,i)   = bfr[count];
	        count++;
	     } 
	  }
       }
       free(bfs); free(bfr);

//south and north, without corner (since corner value is not needed for restriction)
       bfs = (double *)malloc(Nx*sizeof(double));
       bfr = (double *)malloc(Nx*sizeof(double));

       if(pcrd[1]%2 == 0)
       {
    /*first pass to  upper and also receive upper side*/
          if (pcrd[1] < gel->pgmax[1]-1)
          {
	     ss = gmax[1]-ubuf[1]-1;

	     count=0;
	     for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	     {
	       bfs[count]      = VB(gel,i,ss);
               count++;
	     }
	     
	     pcrd[1] += 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[1] -= 1;

	     tag = 11; 
	     MPI_Sendrecv(bfs,Nx,MPI_DOUBLE,his_id,tag,bfr,Nx,MPI_DOUBLE,
	         his_id,tag,PETSC_COMM_WORLD,&status);

             count=0;
	     for (i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	     {
		VT(gel,i,ss) = bfr[count];
	        count++;
	     } 
	  } 
             
	     /*then pass to lower and receive lower side*/
          if (pcrd[1] > 0)
          {
             pcrd[1] -= 1;
             his_id = domain_id(pcrd,gel->pgmax,2);
             pcrd[1] += 1;

	     ss = lbuf[1]-1; 
	     count=0;
	     for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	     {
	        bfs[count]   = VT(gel,i,ss);
                count++;
	     }
	     tag = 12;
             MPI_Sendrecv(bfs,Nx,MPI_DOUBLE,his_id,tag,bfr,Nx,MPI_DOUBLE,
	         his_id,tag,PETSC_COMM_WORLD,&status);
             
	     count=0;
	     for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	     {
		VB(gel,i,ss) = bfr[count];
		count++;
	     }
          }
       }
       
       if(pcrd[1]%2 != 0)
       {
            /*first receive lower side and pass to lower*/
          if (pcrd[1] > 0)
	  {
	       pcrd[1] -= 1;
	       his_id = domain_id(pcrd,gel->pgmax,2);
	       pcrd[1] += 1;
	       ss = lbuf[1]-1;
	       
	       tag = 11;
	       count=0;
	       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	       {
	          bfs[count]    = VT(gel,i,ss);
		  count++;
	       }
	       MPI_Sendrecv(bfs,Nx,MPI_DOUBLE,his_id,tag,bfr,Nx,MPI_DOUBLE,
	           his_id,tag,PETSC_COMM_WORLD,&status);

               count=0;
	       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	       {
		  VB(gel,i,ss) = bfr[count];
		  count++;
	       }
	  }
	       /*then receive upper side and pass to upper*/
          if (pcrd[1] < gel->pgmax[1]-1)
	  {
	       pcrd[1] += 1;
	       his_id = domain_id(pcrd,gel->pgmax,2);
	       pcrd[1] -= 1;

               ss = gmax[1]-ubuf[1]-1;

	       tag = 12;
	       count=0;
	       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	       {
	          bfs[count]      = VB(gel,i,ss);
		  count++;
	       }
	       MPI_Sendrecv(bfs,Nx,MPI_DOUBLE,his_id,tag,bfr,Nx,MPI_DOUBLE,
	          his_id,tag,PETSC_COMM_WORLD,&status);

               count=0;
	       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	       {
		  VT(gel,i,ss) = bfr[count];
		  count++;
	       }
	  }
       }

       free(bfs); free(bfr);
}

static void update_ghost_estate_we(
        GEL    *gel,
        int    *gmax)
{
       double  *bfs,*bfr;
       int     *lbuf=gel->rgr.lbuf;
       int     *ubuf=gel->rgr.ubuf;

       int     *pcrd = gel->pcrd;
       int     Ny = (gmax[1]-lbuf[1]-ubuf[1]) ;
       int     his_id,ss,count,tag,i,j;
       MPI_Status status;
 
       if(gel->pgmax[0] != 1) //need communication
       {
         bfs = malloc(Ny*sizeof(double));
         bfr = malloc(Ny*sizeof(double));
       
        if(pcrd[0] == 0)
        {
	    /*pass to  right bound and also receive left bound*/
	  ss = lbuf[0]-1;
	  count=0;
	  for(i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	  {
	     bfs[count]     = VR(gel,ss,i);
             count++;
	  }

	  pcrd[0] += (gel->pgmax[0]-1);
	  his_id = domain_id(pcrd,gel->pgmax,2);
	  pcrd[0] -= (gel->pgmax[0]-1);

	  tag = 51; 
	  MPI_Sendrecv(bfs,Ny,MPI_DOUBLE,his_id,tag,
	    bfr,Ny,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
          count=0;
	  for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	  {
	     VL(gel,ss,i)  = bfr[count];
	     count++;
	  } 
        } 
       
        if(pcrd[0] == gel->pgmax[0]-1)
        {
	    /*pass to  left bound and also receive right bound*/
	  ss = gmax[0]-ubuf[0]-1;

	  count=0;
	  for(i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	  {
	     bfs[count]      = VL(gel,ss,i);
             count++;
	  }

	  pcrd[0] -= (gel->pgmax[0]-1);
	  his_id = domain_id(pcrd,gel->pgmax,2);
	  pcrd[0] += (gel->pgmax[0]-1);

	  tag = 51; 
	  MPI_Sendrecv(bfs,Ny,MPI_DOUBLE,his_id,tag,
	    bfr,Ny,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
          count=0;
	  for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	  {
	     VR(gel,ss,i)  = bfr[count];
	     count++;
	  } 
        } 
        free(bfs); free(bfr);
       }
       else
       {
          for(i=lbuf[1]; i<gmax[1]-ubuf[1]; ++i)
	  {
	     //left side
	     VL(gel,lbuf[0]-1,i) = 0.0; 
             SL(gel,lbuf[0]-1,i) = 0.0;
             CAL(gel,lbuf[0]-1,i) = 0.0; 
             C2L(gel,lbuf[0]-1,i) = 0.0;
             CLL(gel,lbuf[0]-1,i) = 0.0;   

 
	     //right side
	     VR(gel,gmax[0]-ubuf[0]-1,i) = 0.0; 
             SR(gel,gmax[0]-ubuf[0]-1,i) = 0.0;
             CAR(gel,gmax[0]-ubuf[0]-1,i) = 0.0;
             C2R(gel,gmax[0]-ubuf[0]-1,i) = 0.0;
             CLR(gel,gmax[0]-ubuf[0]-1,i) = 0.0;
	  }
       }
}

//update residual for periodic BC in south and north 

static void update_ghost_estate_sn(
        GEL    *gel,
        int    *gmax)
{
       double  *bfs,*bfr;
       int     *lbuf=gel->rgr.lbuf;
       int     *ubuf=gel->rgr.ubuf;

       int     *pcrd = gel->pcrd;
       int     Nx = (gmax[0]-lbuf[0]-ubuf[0]) ;
       int     his_id,ss,count,tag,i,j;
       MPI_Status status;

       if(gel->pgmax[1] != 1) //need communication
       {
         bfs = malloc(Nx*sizeof(double));
         bfr = malloc(Nx*sizeof(double));
       
        if(pcrd[1] == 0)
        {
	    /*pass to  upper bound and also receive lower bound*/
	  ss = lbuf[1]-1;
	  count=0;
	  for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	  {
	     bfs[count]     = VT(gel,i,ss);
             count++;
	  }

	  pcrd[1] += (gel->pgmax[1]-1);
	  his_id = domain_id(pcrd,gel->pgmax,2);
	  pcrd[1] -= (gel->pgmax[1]-1);

	  tag = 52; 
	  MPI_Sendrecv(bfs,Nx,MPI_DOUBLE,his_id,tag,
	    bfr,Nx,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
          count=0;
	  for (i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	  {
	     VB(gel,i,ss)  = bfr[count];
	     count++;
	  } 
        } 
       
        if(pcrd[1] == gel->pgmax[1]-1)
        {
	    /*pass to  lower bound and also receive upper bound*/
	  ss = gmax[1]-ubuf[1]-1;

	  count=0;
	  for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	  {
	     bfs[count]      = VB(gel,i,ss);
             count++;
	  }

	  pcrd[1] -= (gel->pgmax[1]-1);
	  his_id = domain_id(pcrd,gel->pgmax,2);
	  pcrd[1] += (gel->pgmax[1]-1);

	  tag = 52; 
	  MPI_Sendrecv(bfs,Nx,MPI_DOUBLE,his_id,tag,
	    bfr,Nx,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
          count=0;
	  for (i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	  {
	     VT(gel,i,ss)  = bfr[count];
	     count++;
	  } 
        } 
        free(bfs); free(bfr);
       }
       else
       {
          for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
	  {
	     //lower bound
	     VB(gel,i,lbuf[1]-1) = VB(gel,i,gmax[1]-ubuf[1]-1);
	  
	     //upper bound 
	     VT(gel,i,gmax[1]-ubuf[1]-1) = VT(gel,i,lbuf[1]-1);
	  }
       }
}



static void update_buffer_testate(
	GEL    *gel,
	int    *gmax)
{
       double  *bfs,*bfr;
       int     *lbuf=gel->rgr.lbuf;
       int     *ubuf=gel->rgr.ubuf;

       int     *pcrd = gel->pcrd;
       int     Nx = (gmax[0]-lbuf[0]-ubuf[0]);
       int     Ny = (gmax[1]-lbuf[1]-ubuf[1]) ;
       int     q,his_id,ss,count,tag,i,j;

       MPI_Status status;
      
       bfs = malloc(4*Ny*sizeof(double));
       bfr = malloc(4*Ny*sizeof(double));

       if(pcrd[0]%2 == 0)
       {
	    /*first pass to  right and also receive righ side*/
	  if (pcrd[0] < gel->pgmax[0]-1)
	  {
	     count = 0;

             ss = gmax[0]-ubuf[0]-1;

	     for(i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	     {
                for(q=0; q<4; ++q)
		{
                   bfs[count]    = VTL(gel,ss,i)[q];
		   count++;
	        }
             }
	     
	     pcrd[0] += 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] -= 1;

	     tag = 9; 
	     MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,bfr,4*Ny,MPI_DOUBLE,
	          his_id,tag,PETSC_COMM_WORLD,&status);
	
             count=0;
	     
	     for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	     {
                for(q=0; q<4; ++q)
	        {
                   VTR(gel,ss,i)[q]   = bfr[count];
	           count++;
	        }
             } 
	  }
            
	   /*then pass to left and receive left side*/
          if (pcrd[0] > 0)
          {
              pcrd[0] -= 1;
              his_id = domain_id(pcrd,gel->pgmax,2);
              pcrd[0] += 1;

              ss = lbuf[0]-1;
	      count = 0;

	      for(i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	      {
                 for(q=0; q<4; ++q)
	         {
                     bfs[count]     = VTR(gel,ss,i)[q];
                     count++;
	         }
              }

	      tag = 10;
              MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,bfr,4*Ny,MPI_DOUBLE,
	            his_id,tag,PETSC_COMM_WORLD,&status);

              count=0;
	      for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	      {
                 for(q=0; q<4; ++q)
		 {
                    VTL(gel,ss,i)[q]   = bfr[count];
	            count++;
	         }
              }
	  }        
       }
	
       if(pcrd[0]%2 != 0)
       {
            /*first receive left side and pass to left*/
          if (pcrd[0] > 0)
          {
	     pcrd[0] -= 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] += 1;
	     ss = lbuf[0]-1;
	     
	     tag = 9;
	     count=0;
	     for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	     {
                for(q=0; q<4; ++q)
	        {
                   bfs[count]      = VTR(gel,ss,i)[q];
		   count++;
	        }
             }
	     MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,bfr,4*Ny,MPI_DOUBLE,
	          his_id,tag,PETSC_COMM_WORLD,&status);
	     
	     count=0;
	     for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	     {
                for(q=0; q<4; ++q)
		{
                   VTL(gel,ss,i)[q] = bfr[count];
		   count++;
	        }
             }
             
	  }
	       /*then receive right side and pass to right*/
          if (pcrd[0] < gel->pgmax[0]-1)
	  {
	     pcrd[0] += 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] -= 1;

             ss = gmax[0]-ubuf[0]-1;

	     tag = 10;
	     count=0;
	     for (i=lbuf[1] ;i<gmax[1]-ubuf[1]; i++)
             {
                for(q=0; q<4; ++q)
                {
                   bfs[count]      = VTL(gel,ss,i)[q];
	           count++;
	        }
             } 
	     MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,bfr,4*Ny,MPI_DOUBLE,
	         his_id,tag,PETSC_COMM_WORLD,&status);
             
	     count=0;
             for (i=lbuf[1] ;i<gmax[1]-ubuf[1]; i++)
	     {
                for(q=0; q<4; ++q)
	        {
                   VTR(gel,ss,i)[q]   = bfr[count];
	           count++;
	        }
             } 
	  }
       }
       free(bfs); free(bfr);

//south and north, without corner (since corner value is not needed for restriction)
       bfs = (double *)malloc(4*Nx*sizeof(double));
       bfr = (double *)malloc(4*Nx*sizeof(double));

       if(pcrd[1]%2 == 0)
       {
    /*first pass to  upper and also receive upper side*/
          if (pcrd[1] < gel->pgmax[1]-1)
          {
	     ss = gmax[1]-ubuf[1]-1;

	     count=0;
	     for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	     {
               for(q=0; q<4; ++q)
	       {
                  bfs[count]      = VTB(gel,i,ss)[q];
                  count++;
	       }
             }
	     
	     pcrd[1] += 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[1] -= 1;

	     tag = 11; 
	     MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,bfr,4*Nx,MPI_DOUBLE,
	         his_id,tag,PETSC_COMM_WORLD,&status);

             count=0;
	     for (i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	     {
                for(q=0; q<4; ++q)
		{
                   VTT(gel,i,ss)[q] = bfr[count];
	           count++;
	        }
             } 
	  } 
             
	     /*then pass to lower and receive lower side*/
          if (pcrd[1] > 0)
          {
             pcrd[1] -= 1;
             his_id = domain_id(pcrd,gel->pgmax,2);
             pcrd[1] += 1;

	     ss = lbuf[1]-1; 
	     count=0;
	     for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	     { 
                for(q=0; q<4; ++q)
	        {
                   bfs[count]   = VTT(gel,i,ss)[q];
                   count++;
	        }
             }
	     tag = 12;
             MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,bfr,4*Nx,MPI_DOUBLE,
	         his_id,tag,PETSC_COMM_WORLD,&status);
             
	     count=0;
	     for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	     {
                for(q=0; q<4; ++q)
		{
                   VTB(gel,i,ss)[q] = bfr[count];
		   count++;
	        }
             }
          }
       }
       
       if(pcrd[1]%2 != 0)
       {
            /*first receive lower side and pass to lower*/
          if (pcrd[1] > 0)
	  {
	       pcrd[1] -= 1;
	       his_id = domain_id(pcrd,gel->pgmax,2);
	       pcrd[1] += 1;
	       ss = lbuf[1]-1;
	       
	       tag = 11;
	       count=0;
	       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	       {
                  for(q=0; q<4; ++q)
	          {
                     bfs[count]    = VTT(gel,i,ss)[q];
		     count++;
	          }
               }
	       MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,bfr,4*Nx,MPI_DOUBLE,
	           his_id,tag,PETSC_COMM_WORLD,&status);

               count=0;
	       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	       {
                  for(q=0; q<4; ++q) 
		  {
                     VTB(gel,i,ss)[q] = bfr[count];
		     count++;
	          }
               }
	  }
	       /*then receive upper side and pass to upper*/
          if (pcrd[1] < gel->pgmax[1]-1)
	  {
	       pcrd[1] += 1;
	       his_id = domain_id(pcrd,gel->pgmax,2);
	       pcrd[1] -= 1;

               ss = gmax[1]-ubuf[1]-1;

	       tag = 12;
	       count=0;
	       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	       {
                  for(q=0; q<4; ++q)
	          {
                     bfs[count]      = VTB(gel,i,ss)[q];
		     count++;
	          }
               }
	       MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,bfr,4*Nx,MPI_DOUBLE,
	          his_id,tag,PETSC_COMM_WORLD,&status);

               count=0;
	       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	       {
                  for(q=0; q<4; ++q)
		  {
                     VTT(gel,i,ss)[q] = bfr[count];
		     count++;
	          }
               }
	  }
       }

       free(bfs); free(bfr);
}

static void update_ghost_testate_we(
        GEL    *gel,
        int    *gmax)
{
       double  *bfs,*bfr;
       int     *lbuf=gel->rgr.lbuf;
       int     *ubuf=gel->rgr.ubuf;

       int     *pcrd = gel->pcrd;
       int     Ny = (gmax[1]-lbuf[1]-ubuf[1]) ;
       int     his_id,q,ss,count,tag,i,j;
       MPI_Status status;
 
       if(gel->pgmax[0] != 1) //need communication
       {
         bfs = malloc(4*Ny*sizeof(double));
         bfr = malloc(4*Ny*sizeof(double));
       
        if(pcrd[0] == 0)
        {
	    /*pass to  right bound and also receive left bound*/
	  ss = lbuf[0]-1;
	  count=0;
	  for(i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	  {
             for(q=0; q<4; ++q)
	     {
               bfs[count]     = VTR(gel,ss,i)[q];
               count++;
	     }
          }

	  pcrd[0] += (gel->pgmax[0]-1);
	  his_id = domain_id(pcrd,gel->pgmax,2);
	  pcrd[0] -= (gel->pgmax[0]-1);

	  tag = 51; 
	  MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,
	    bfr,4*Ny,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
          count=0;
	  for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	  {
             for(q=0; q<4; ++q)
	     {
               VTL(gel,ss,i)[q]  = bfr[count];
	       count++;
	     }
          } 
        } 
       
        if(pcrd[0] == gel->pgmax[0]-1)
        {
	    /*pass to  left bound and also receive right bound*/
	  ss = gmax[0]-ubuf[0]-1;

	  count=0;
	  for(i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	  {
             for(q=0; q<4; ++q)
	     {
                bfs[count]      = VTL(gel,ss,i)[q];
                count++;
	     }
          }

	  pcrd[0] -= (gel->pgmax[0]-1);
	  his_id = domain_id(pcrd,gel->pgmax,2);
	  pcrd[0] += (gel->pgmax[0]-1);

	  tag = 51; 
	  MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,
	    bfr,4*Ny,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
          count=0;
	  for (i=lbuf[1]; i<gmax[1]-ubuf[1]; i++)
	  {
	     for(q=0; q<4; ++q)
             {
                VTR(gel,ss,i)[q]  = bfr[count];
	        count++;
	     }
          } 
        } 
        free(bfs); free(bfr);
       }
       else
       {
          for(i=lbuf[1]; i<gmax[1]-ubuf[1]; ++i)
	  {
             for(q=0; q<4; ++q)
             {
	     //left side
	     VTL(gel,lbuf[0]-1,i)[q] = VTL(gel,gmax[0]-ubuf[0]-1,i)[q];
	  
	     //right side
	     VTR(gel,gmax[0]-ubuf[0]-1,i)[q] = VTR(gel,lbuf[0]-1,i)[q];
	     }
          }
       }
}

//update residual for periodic BC in south and north 

static void update_ghost_testate_sn(
        GEL    *gel,
        int    *gmax)
{
       double  *bfs,*bfr;
       int     *lbuf=gel->rgr.lbuf;
       int     *ubuf=gel->rgr.ubuf;

       int     *pcrd = gel->pcrd;
       int     Nx = (gmax[0]-lbuf[0]-ubuf[0]) ;
       int     his_id,ss,q,count,tag,i,j;
       MPI_Status status;

       if(gel->pgmax[1] != 1) //need communication
       {
         bfs = malloc(4*Nx*sizeof(double));
         bfr = malloc(4*Nx*sizeof(double));
       
        if(pcrd[1] == 0)
        {
	    /*pass to  upper bound and also receive lower bound*/
	  ss = lbuf[1]-1;
	  count=0;
	  for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	  {
             for(q=0; q<4; ++q)
	     {
               bfs[count]     = VTT(gel,i,ss)[q];
               count++;
	     }
          }

	  pcrd[1] += (gel->pgmax[1]-1);
	  his_id = domain_id(pcrd,gel->pgmax,2);
	  pcrd[1] -= (gel->pgmax[1]-1);

	  tag = 52; 
	  MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,
	    bfr,4*Nx,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
          count=0;
	  for (i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	  {
             for(q=0; q<4; ++q)
	     {
               VTB(gel,i,ss)[q]  = bfr[count];
	       count++;
	     }
          } 
        } 
       
        if(pcrd[1] == gel->pgmax[1]-1)
        {
	    /*pass to  lower bound and also receive upper bound*/
	  ss = gmax[1]-ubuf[1]-1;

	  count=0;
	  for(i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	  {
             for(q=0; q<4; ++q)
	     {
               bfs[count]      = VTB(gel,i,ss)[q];
               count++;
	     }
          }

	  pcrd[1] -= (gel->pgmax[1]-1);
	  his_id = domain_id(pcrd,gel->pgmax,2);
	  pcrd[1] += (gel->pgmax[1]-1);

	  tag = 52; 
	  MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,
	    bfr,4*Nx,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
          count=0;
	  for (i=lbuf[0]; i<gmax[0]-ubuf[0]; i++)
	  {
             for(q=0; q<4; ++q)
	     {
                VTT(gel,i,ss)[q]  = bfr[count];
	        count++;
	     }
          } 
        } 
        free(bfs); free(bfr);
       }
       else
       {
          for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
	  {
             for(q=0; q<4; ++q)
             {
	     //lower bound
	     VTB(gel,i,lbuf[1]-1)[q] = VTB(gel,i,gmax[1]-ubuf[1]-1)[q];
	  
	     //upper bound 
	     VTT(gel,i,gmax[1]-ubuf[1]-1)[q] = VTT(gel,i,lbuf[1]-1)[q];
             }
	  }
       }
}

static void update_buffer_tao_and_z(
	GEL    *gel,
	int    *gmax)
{
       double  *bfs,*bfr;
       int     *lbuf=gel->rgr.lbuf;
       int     *ubuf=gel->rgr.ubuf;

       int     *pcrd = gel->pcrd;
       int     Nx = gmax[0];
       int     Ny = gmax[1];
       int     his_id,ss,count,tag,i,j;

       MPI_Status status;
      
       bfs = malloc(4*Ny*sizeof(double));
       bfr = malloc(4*Ny*sizeof(double));

       if(pcrd[0]%2 == 0)
       {
	    /*first pass to  right and also receive righ side*/
	  if (pcrd[0] < gel->pgmax[0]-1)
	  {
	     count = 0;

             ss = gmax[0]-ubuf[0]-1;

	     for(i=0; i<gmax[1]; i++)
	     {
		bfs[count]      = T11(gel,ss,i);
		bfs[count+Ny]   = T22(gel,ss,i);
	        bfs[count+2*Ny] = T12(gel,ss,i);
		bfs[count+3*Ny] = Z(gel,ss,i);
		count++;
	     }
	     
	     pcrd[0] += 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] -= 1;

	     tag = 9; 
	     MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,bfr,4*Ny,MPI_DOUBLE,
	          his_id,tag,PETSC_COMM_WORLD,&status);
	
             count=0;
	     
	     for (i=0; i<gmax[1]; i++)
	     {
	        T11(gel,ss+1,i)   = bfr[count];
	        T22(gel,ss+1,i)   = bfr[count+Ny];
                T12(gel,ss+1,i)   = bfr[count+2*Ny];
                Z(gel,ss+1,i)     = bfr[count+3*Ny];
	        count++;
	     } 
	  }
            
	   /*then pass to left and receive left side*/
          if (pcrd[0] > 0)
          {
              pcrd[0] -= 1;
              his_id = domain_id(pcrd,gel->pgmax,2);
              pcrd[0] += 1;

              ss = lbuf[0];
	      count = 0;

	      for(i=0; i<gmax[1]; i++)
	      {
	         bfs[count]       = T11(gel,ss,i);
                 bfs[count+Ny]    = T22(gel,ss,i); 
                 bfs[count+2*Ny]  = T12(gel,ss,i);
                 bfs[count+3*Ny]  = Z(gel,ss,i);
                 count++;
	      }

	      tag = 10;
              MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,bfr,4*Ny,MPI_DOUBLE,
	            his_id,tag,PETSC_COMM_WORLD,&status);

              count=0;
	      for (i=0; i<gmax[1]; i++)
	      {
		 T11(gel,ss-1,i) = bfr[count];
		 T22(gel,ss-1,i) = bfr[count+Ny];
		 T12(gel,ss-1,i) = bfr[count+2*Ny];
		 Z(gel,ss-1,i)   = bfr[count+3*Ny];
	         count++;
	      }
	  }        
       }
	
       if(pcrd[0]%2 != 0)
       {
            /*first receive left side and pass to left*/
          if (pcrd[0] > 0)
          {
	     pcrd[0] -= 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] += 1;
	     ss = lbuf[0];
	     
	     tag = 9;
	     count=0;
	     for (i=0; i<gmax[1]; i++)
	     {
	        bfs[count]        = T11(gel,ss,i);
                bfs[count+Ny]     = T22(gel,ss,i); 
		bfs[count+2*Ny]   = T12(gel,ss,i); 
                bfs[count+3*Ny]   = Z(gel,ss,i);
                count++;
	     }
	     MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,bfr,4*Ny,MPI_DOUBLE,
	          his_id,tag,PETSC_COMM_WORLD,&status);
	     
	     count=0;
	     for (i=0; i<gmax[1]; i++)
	     {
		T11(gel,ss-1,i) = bfr[count];
		T22(gel,ss-1,i) = bfr[count+Ny];
		T12(gel,ss-1,i) = bfr[count+2*Ny];
		Z(gel,ss-1,i)   = bfr[count+3*Ny];
		count++;
	     }
             
	  }
	       /*then receive right side and pass to right*/
          if (pcrd[0] < gel->pgmax[0]-1)
	  {
	     pcrd[0] += 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] -= 1;

             ss = gmax[0]-ubuf[0]-1;

	     tag = 10;
	     count=0;
	     for (i=0 ;i<gmax[1]; i++)
             {
                bfs[count]      = T11(gel,ss,i);
		bfs[count+Ny]   = T22(gel,ss,i);
	        bfs[count+2*Ny] = T12(gel,ss,i);
                bfs[count+3*Ny] = Z(gel,ss,i); 
	        count++;
	     } 
	     MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,bfr,4*Ny,MPI_DOUBLE,
	         his_id,tag,PETSC_COMM_WORLD,&status);
             
	     count=0;
             for (i=0;i<gmax[1]; i++)
	     {
	        T11(gel,ss+1,i) = bfr[count];
		T22(gel,ss+1,i) = bfr[count+Ny];
	        T12(gel,ss+1,i) = bfr[count+2*Ny];
                Z(gel,ss+1,i)   = bfr[count+3*Ny];
                count++;
	     } 
	  }
       }
       free(bfs); free(bfr);

       bfs = (double *)malloc(4*Nx*sizeof(double));
       bfr = (double *)malloc(4*Nx*sizeof(double));

       if(pcrd[1]%2 == 0)
       {
    /*first pass to  upper and also receive upper side*/
          if (pcrd[1] < gel->pgmax[1]-1)
          {
	     ss = gmax[1]-ubuf[1]-1;

	     count=0;
	     for(i=0; i<gmax[0]; i++)
	     {
	       bfs[count]      = T11(gel,i,ss);
	       bfs[count+Nx]   = T22(gel,i,ss);
               bfs[count+2*Nx] = T12(gel,i,ss);
               bfs[count+3*Nx] = Z(gel,i,ss); 
               count++;
	     }
	     
	     pcrd[1] += 1;
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[1] -= 1;

	     tag = 11; 
	     MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,bfr,4*Nx,MPI_DOUBLE,
	         his_id,tag,PETSC_COMM_WORLD,&status);

             count=0;
	     for (i=0; i<gmax[0]; i++)
	     {
		T11(gel,i,ss+1) = bfr[count];
		T22(gel,i,ss+1) = bfr[count+Nx];
                T12(gel,i,ss+1) = bfr[count+2*Nx];
                Z(gel,i,ss+1) = bfr[count+3*Nx];
	        count++;
	     } 
	  } 
             
	     /*then pass to lower and receive lower side*/
          if (pcrd[1] > 0)
          {
             pcrd[1] -= 1;
             his_id = domain_id(pcrd,gel->pgmax,2);
             pcrd[1] += 1;

	     ss = lbuf[1]; 
	     count=0;
	     for(i=0; i<gmax[0]; i++)
	     {
	        bfs[count]   = T11(gel,i,ss);
                bfs[count+Nx]= T22(gel,i,ss); 
                bfs[count+2*Nx]= T12(gel,i,ss);
                bfs[count+3*Nx]= Z(gel,i,ss);
                count++;
	     }
	     tag = 12;
             MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,bfr,4*Nx,MPI_DOUBLE,
	         his_id,tag,PETSC_COMM_WORLD,&status);
             
	     count=0;
	     for(i=0; i<gmax[0]; i++)
	     {
		T11(gel,i,ss-1) = bfr[count];
		T22(gel,i,ss-1) = bfr[count+Nx];
		T12(gel,i,ss-1) = bfr[count+2*Nx];
		Z(gel,i,ss-1) = bfr[count+3*Nx];
		count++;
	     }
          }
       }
       
       if(pcrd[1]%2 != 0)
       {
            /*first receive lower side and pass to lower*/
          if (pcrd[1] > 0)
	  {
	       pcrd[1] -= 1;
	       his_id = domain_id(pcrd,gel->pgmax,2);
	       pcrd[1] += 1;
	       ss = lbuf[1];
	       
	       tag = 11;
	       count=0;
	       for(i=0; i<gmax[0]; i++)
	       {
	          bfs[count]    = T11(gel,i,ss);
                  bfs[count+Nx] = T22(gel,i,ss); 
		  bfs[count+2*Nx] = T12(gel,i,ss);
                  bfs[count+3*Nx] = Z(gel,i,ss);
                  count++;
	       }
	       MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,bfr,4*Nx,MPI_DOUBLE,
	           his_id,tag,PETSC_COMM_WORLD,&status);

               count=0;
	       for(i=0; i<gmax[0]; i++)
	       {
		  T11(gel,i,ss-1) = bfr[count];
		  T22(gel,i,ss-1) = bfr[count+Nx];
		  T12(gel,i,ss-1) = bfr[count+2*Nx];
		  Z(gel,i,ss-1) = bfr[count+3*Nx];
		  count++;
	       }
	  }
	       /*then receive upper side and pass to upper*/
          if (pcrd[1] < gel->pgmax[1]-1)
	  {
	       pcrd[1] += 1;
	       his_id = domain_id(pcrd,gel->pgmax,2);
	       pcrd[1] -= 1;

               ss = gmax[1]-ubuf[1]-1;

	       tag = 12;
	       count=0;
	       for(i=0; i<gmax[0]; i++)
	       {
	          bfs[count]      = T11(gel,i,ss);
                  bfs[count+Nx]   = T22(gel,i,ss); 
		  bfs[count+2*Nx] = T12(gel,i,ss);
		  bfs[count+3*Nx] = Z(gel,i,ss);
		  count++;
	       }
	       MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,bfr,4*Nx,MPI_DOUBLE,
	          his_id,tag,PETSC_COMM_WORLD,&status);

               count=0;
	       for(i=0; i<gmax[0]; i++)
	       {
		  T11(gel,i,ss+1) = bfr[count];
		  T22(gel,i,ss+1) = bfr[count+Nx];
                  T12(gel,i,ss+1) = bfr[count+2*Nx];
                  Z(gel,i,ss+1) = bfr[count+3*Nx];
		  count++;
	       }
	  }
       }

       free(bfs); free(bfr);
}

//extrapolate state variables from interior 

static void update_ghost_taoz_sn(
       GEL    *gel,
       int    *gmax)
{
       double  *bfs,*bfr;
       int     *lbuf=gel->rgr.lbuf;
       int     *ubuf=gel->rgr.ubuf;

       int     *pcrd = gel->pcrd;
       int     Nx = gmax[0] ;
       int     his_id,ss,count,tag,i,j;

       if(gel->pgmax[1] != 1) //more than two subdomain need communicate
       {
          MPI_Status status;

	  bfs = malloc(4*Nx*sizeof(double));
	  bfr = malloc(4*Nx*sizeof(double));
          
          if(pcrd[1] == 0)
          {
          //pass to upper bound and receive lower bound
	     ss = lbuf[1];
	     count=0;
	     for(i=0; i<gmax[0]; i++)
	     {
	        bfs[count]  = T11(gel,i,ss);
                bfs[count+Nx]  = T22(gel,i,ss);	        
                bfs[count+2*Nx]  = T12(gel,i,ss);
                bfs[count+3*Nx]  = Z(gel,i,ss);
                count++; 
	     }
	     pcrd[1] += (gel->pgmax[1]-1);
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[1] -= (gel->pgmax[1]-1);

	     tag = 53; 
	     MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,
	        bfr,4*Nx,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
             count=0;
	     for (i=0; i<gmax[0]; i++)
	     {
	        T11(gel,i,ss-1) = bfr[count];
	        T22(gel,i,ss-1) = bfr[count+Nx];
                T12(gel,i,ss-1) = bfr[count+2*Nx];
                Z(gel,i,ss-1) = bfr[count+3*Nx];
                count++;
	     } 
	  } 
       
          if(pcrd[1] == gel->pgmax[1]-1)
          {
	    /*pass to  lower bound and also receive upper bound*/
	     ss = gmax[1]-ubuf[1]-1;
	     count=0;
	     for(i=0; i<gmax[0]; i++)
	     {
	        bfs[count]  = T11(gel,i,ss);
	        bfs[count+Nx]  = T22(gel,i,ss);
                bfs[count+2*Nx]  = T12(gel,i,ss);
                bfs[count+3*Nx]  = Z(gel,i,ss);
                count++;
	     }
	     pcrd[1] -= (gel->pgmax[1]-1);
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[1] += (gel->pgmax[1]-1);
	    
	     tag = 53; 
	     MPI_Sendrecv(bfs,4*Nx,MPI_DOUBLE,his_id,tag,
	       bfr,4*Nx,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
             count=0;
	     for (i=0; i<gmax[0]; i++)
	     {
	        T11(gel,i,ss+1) = bfr[count];
	        T22(gel,i,ss+1) = bfr[count+Nx];
                T12(gel,i,ss+1) = bfr[count+2*Nx];
                Z(gel,i,ss+1) = bfr[count+3*Nx];
                count++;
	     } 
          } 
          free(bfs); free(bfr);
       }
       else  //only one subdomain in x-direction
       {
          for(i=0; i<gmax[0]; ++i)
          {
             //lower side
	     T11(gel,i,lbuf[1]-1) = T11(gel,i,lbuf[1]);  
             T12(gel,i,lbuf[1]-1) = T12(gel,i,lbuf[1]);
             T22(gel,i,lbuf[1]-1) = T22(gel,i,lbuf[1]);
             Z(gel,i,lbuf[1]-1) = Z(gel,i,lbuf[1]);
 
	     //upper side
	     T11(gel,i,gmax[1]-ubuf[1]) = T11(gel,i,gmax[1]-ubuf[1]-1);
             T12(gel,i,gmax[1]-ubuf[1]) = T12(gel,i,gmax[1]-ubuf[1]-1);
             T22(gel,i,gmax[1]-ubuf[1]) = T22(gel,i,gmax[1]-ubuf[1]-1);
             Z(gel,i,gmax[1]-ubuf[1]) = Z(gel,i,gmax[1]-ubuf[1]-1);
          }
       }
       return;
}

static void update_ghost_taoz_we(
       GEL    *gel,
       int    *gmax)
{
       double  *bfs,*bfr;
       int     *lbuf=gel->rgr.lbuf;
       int     *ubuf=gel->rgr.ubuf;

       int     *pcrd = gel->pcrd;
       int     Ny = gmax[1] ;
       int     his_id,ss,count,tag,i,j;

       if(gel->pgmax[0] != 1) //more than two subdomain need communicate
       {
          MPI_Status status;

	  bfs = malloc(4*Ny*sizeof(double));
	  bfr = malloc(4*Ny*sizeof(double));
          if(pcrd[0] == 0)
          {
          //pass to right boundary and receive left boundary
	     ss = lbuf[0];
	     count=0;
	     for(i=0; i<gmax[1]; i++)
	     {
	        bfs[count]  = T11(gel,ss,i);
	        bfs[count+Ny]  = T22(gel,ss,i);
                bfs[count+2*Ny]  = T12(gel,ss,i);
                bfs[count+3*Ny]  = Z(gel,ss,i);
                count++;
	     }
	     pcrd[0] += (gel->pgmax[0]-1);
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] -= (gel->pgmax[0]-1);

	     tag = 54; 
	     MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,
	        bfr,4*Ny,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
             count=0;
	     for (i=0; i<gmax[1]; i++)
	     {
	        T11(gel,ss-1,i) = bfr[count];
	        T22(gel,ss-1,i) = bfr[count+Ny];
                T12(gel,ss-1,i) = bfr[count+2*Ny];
                Z(gel,ss-1,i) = bfr[count+3*Ny];
                count++;
	     } 
	  } 
       
          if(pcrd[0] == gel->pgmax[0]-1)
          {
	    /*pass to  left bound and also receive right bound*/
	     ss = gmax[0]-ubuf[0]-1;
	     count=0;
	     for(i=0; i<gmax[1]; i++)
	     {
	        bfs[count]  = T11(gel,ss,i);
	        bfs[count+Ny]  = T22(gel,ss,i);
                bfs[count+2*Ny]  = T12(gel,ss,i);
                bfs[count+3*Ny]  = Z(gel,ss,i);
                count++;
	     }
	     pcrd[0] -= (gel->pgmax[0]-1);
	     his_id = domain_id(pcrd,gel->pgmax,2);
	     pcrd[0] += (gel->pgmax[0]-1);
	    
	     tag = 54; 
	     MPI_Sendrecv(bfs,4*Ny,MPI_DOUBLE,his_id,tag,
	       bfr,4*Ny,MPI_DOUBLE,his_id,tag,PETSC_COMM_WORLD,&status);
	
             count=0;
	     for (i=0; i<gmax[1]; i++)
	     {
	        T11(gel,ss+1,i) = bfr[count];
	        T22(gel,ss+1,i) = bfr[count+Ny];
                T12(gel,ss+1,i) = bfr[count+2*Ny];
                Z(gel,ss+1,i) = bfr[count+3*Ny];
                count++;
	     } 
          } 
          free(bfs); free(bfr);
       }
       else  //only one subdomain in x-direction
       {
          for(i=0; i<gmax[1]; ++i)
          {
             //left side
	     T11(gel,lbuf[0]-1,i) = T11(gel,gmax[0]-ubuf[0]-1,i);  
             T22(gel,lbuf[0]-1,i) = T22(gel,gmax[0]-ubuf[0]-1,i);
             T12(gel,lbuf[0]-1,i) = T12(gel,gmax[0]-ubuf[0]-1,i);
             Z(gel,lbuf[0]-1,i) = Z(gel,gmax[0]-ubuf[0]-1,i);
	     //right side
	     T11(gel,gmax[0]-ubuf[0],i) = T11(gel,lbuf[0],i);
             T22(gel,gmax[0]-ubuf[0],i) = T22(gel,lbuf[0],i);
             T12(gel,gmax[0]-ubuf[0],i) = T12(gel,lbuf[0],i);
             Z(gel,gmax[0]-ubuf[0],i) = Z(gel,lbuf[0],i); 
          }
       }
       return;
}

static void update_tao_and_znew(
       GEL    *gel,
       int    *gmax)
{
       int     m,n,p,q,i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;

       double  h = gel->rgr.h;
       double  dt=gel->dt;
       double  beta = gel->beta;

       double  **A;  //4x4 system to solve for each cell
       double  det,a,b,c,tmpv[3],B[3][3],v1,v2;
       double  rhs[3],xold[3],xnew[3]; //old and new solution vection
       double  C[2][2],tmp[2][2]; //old and new formation term
       double  lam1,lam2,lp1,lp2,check,ux,uy,vx,vy,oux,ouy,ovx,ovy;
       double  r1[2],r2[2],R[2][2],R_inv[2][2];
 
//calculate the advection term on edge on half step 
       update_taoz_fluxnew(gel,gmax);

       A = malloc(3*sizeof(double*));

       for(i = 0; i < 3; i++)
	  A[i] = malloc(3*sizeof(double));
      
       for(i=0; i<gmax[0]; ++i)
       {
          for(j=0; j<gmax[1]; ++j)
          {
              //save old values first
            OT11(gel,i,j) = T11(gel,i,j);
            OT12(gel,i,j) = T12(gel,i,j);
            OT22(gel,i,j) = T22(gel,i,j);

            if(i>=lbuf[0] && i<gmax[0]-ubuf[0] && j>=lbuf[1] && j<gmax[1]-ubuf[1])
            {
              xold[0] = T11(gel,i,j); xold[1] = T12(gel,i,j);
              xold[2] = T22(gel,i,j); 

              tmpv[0] = tmpv[1] = tmpv[2] = 0.0;
             
//extrapolate velocity to k+1/2 

              ux = (Un(gel,i,j)-Un(gel,i-1,j))/h;
              oux = (OUn(gel,i,j)-OUn(gel,i-1,j))/h;
              ux = 1.5*ux-0.5*oux;

              v1 = 0.5*(Un(gel,i,j+1)+Un(gel,i-1,j+1));
              v2 = 0.5*(Un(gel,i,j-1)+Un(gel,i-1,j-1));
              uy = (v1-v2)/(2*h);
             
              v1 = 0.5*(OUn(gel,i,j+1)+OUn(gel,i-1,j+1));
              v2 = 0.5*(OUn(gel,i,j-1)+OUn(gel,i-1,j-1));
              ouy = (v1-v2)/(2*h);
              uy = 1.5*uy-0.5*ouy;
 
              vy = (Vn(gel,i,j)-Vn(gel,i,j-1))/h;
              ovy = (OVn(gel,i,j)-OVn(gel,i,j-1))/h;
              vy = 1.5*vy-0.5*ovy;
             
              v1 = 0.5*(Vn(gel,i+1,j)+Vn(gel,i+1,j-1));
              v2 = 0.5*(Vn(gel,i-1,j)+Vn(gel,i-1,j-1));
              vx = (v1-v2)/(2*h); 

              v1 = 0.5*(OVn(gel,i+1,j)+OVn(gel,i+1,j-1));
              v2 = 0.5*(OVn(gel,i-1,j)+OVn(gel,i-1,j-1));
              ovx = (v1-v2)/(2*h);
              vx = 1.5*vx-0.5*ovx;

              B[0][0] = 2*ux;   B[0][1] = 2*uy;
              B[0][2] = 0;      

              B[1][0] = vx;     B[1][1] = ux+vy;
              B[1][2] = uy;    ;
 
              B[2][0] = 0;      B[2][1] = 2*vx;
              B[2][2] = 2*vy;   
                       
              for(p=0; p<3; ++p)
                 for(q=0; q<3; ++q)
                 {
                    if(p==q)
                       A[p][q] = 1.0+0.5*dt*beta;
                    else
                       A[p][q] = 0.0;
                  }
            
              for(p=0; p<3; ++p)
                 for(q=0; q<3; ++q)
                    A[p][q] -= 0.5*dt*B[p][q]; 

              for(m=0; m<3; ++m)
                 for(n=0; n<3; ++n)
                    tmpv[m] += B[m][n]*xold[n];
 
              for(p=0; p<3; ++p)
              {
                 rhs[p] = xold[p] -dt/h*(FTx(gel,i,j)[p]-FTx(gel,i-1,j)[p]+FTy(gel,i,j)[p]-FTy(gel,i,j-1)[p]);
                 rhs[p] += 0.5*dt*(tmpv[p]-beta*xold[p]);
              }

              rhs[0] += dt*2*ux*Z(gel,i,j); rhs[1] += dt*(vx+uy)*Z(gel,i,j);
              rhs[2] += dt*2*vy*Z(gel,i,j);

              linear_system_solver(A,rhs,3,xnew);

              if(isnan(xnew[0]) || isnan(xnew[1]) || isnan(xnew[2])  || isnan(xnew[3]) )
              {
                 printf("\n warning, taoz solver failed for cell[%d %d], with solution = %6.5e %6.5e %6.5e %6.5e",i,j,xnew[0],xnew[1],xnew[2],xnew[3]);
                 linear_system_solver(A,rhs,3,xnew);
              } 

              T11(gel,i,j) = xnew[0];  T12(gel,i,j) =  xnew[1];
              T22(gel,i,j) = xnew[2];   
              //printf("\n solved T11 for cell[%d %d] is %10.9e",i,j,T11(gel,i,j));
              
              //check for semi-positive-definite
              check = (T11(gel,i,j)+Z(gel,i,j))*(T22(gel,i,j)+Z(gel,i,j));
              if(check < T12(gel,i,j)*T12(gel,i,j))
              {
                  printf("\n warning!, in cell[%d %d] violation is found with value=%6.5e !",i,j,
                  check-T12(gel,i,j)*T12(gel,i,j));
          
                  a = T11(gel,i,j)+Z(gel,i,j); b = T12(gel,i,j); 
                  c = T22(gel,i,j)+Z(gel,i,j); 
                  lam1 = 0.5*(a+c+sqrt((a-c)*(a-c)+4*b*b));
                  lam2 = 0.5*(a+c-sqrt((a-c)*(a-c)+4*b*b));

                  printf("\n lam1=%6.5e,lam2=%6.5e",lam1,lam2);

                  r1[0] = 1.0; r1[1] = (lam1-a)/b; 
                  r2[0] = 1.0; r2[1] = (lam2-a)/b; 
                  
                  lp1 = 0.5*(lam1+fabs(lam1));
                  lp2 = 0.5*(lam2+fabs(lam2));
               
                  R[0][0] = r1[0]; R[1][0] = r1[1];
                  R[0][1] = r2[0]; R[1][1] = r2[1];                  
               
                  det = R[0][0]*R[1][1]-R[1][0]*R[0][1];
                  R_inv[0][0] = 1/det*R[1][1]; R_inv[0][1] = -1/det*R[0][1];
                  R_inv[1][0] = -1/det*R[1][0]; R_inv[1][1] = 1/det*R[0][0];
            
                  tmp[0][0] = lp1*R_inv[0][0]; tmp[0][1] = lp1*R_inv[0][1];
                  tmp[1][0] = lp2*R_inv[1][0]; tmp[1][1] = lp2*R_inv[1][1];

                  C[0][0] = R[0][0]*tmp[0][0]+R[0][1]*tmp[1][0];
                  C[0][1] = R[0][0]*tmp[0][1]+R[0][1]*tmp[1][1];
                  C[1][0] = R[1][0]*tmp[0][0]+R[1][1]*tmp[1][0];
                  C[1][1] = R[1][0]*tmp[0][1]+R[1][1]*tmp[1][1];
 
                  T11(gel,i,j) = C[0][0]-Z(gel,i,j); 
                  T22(gel,i,j) = C[1][1]-Z(gel,i,j);
                  T12(gel,i,j) = C[0][1];  
                 
                  printf("\n C=[%6.5e %6.5e],z = %6.5e",C[0][1],C[1][0],Z(gel,i,j)); 
                  //check again
                  check = (T11(gel,i,j)+Z(gel,i,j))*(T22(gel,i,j)+Z(gel,i,j));
                  printf("\n revised value is %6.5e",check-T12(gel,i,j)*T12(gel,i,j)); 
             }
            } 
          }
       } 
       
       for(i=0; i<3; ++i)
         free(A[i]);
       
       free(A); 
  
       update_ghost_taoz_we(gel,gmax);
       update_ghost_taoz_sn(gel,gmax);
       update_buffer_tao_and_z(gel,gmax);
}

static void update_taoz_fluxnew(
       GEL     *gel,
       int     *gmax)
{
       int     m,n,p,i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;
 
       double   h = gel->rgr.h;
       double   q[4],B[4][4];
       double   tmp[4],a[4],bq[4],betaq[4];
       double   qxl[4],qxr[4],qxc[4],qx[4];
       double   xold[4],qyt[4],qyb[4],qyc[4],qy[4];
       double   aold[4],ql[4],qr[4],qb[4],qt[4]; //all the neighbors
       double   uh,vh,v1,v2,uc,vc,ouc,ovc; //cell centered velocity
       double   ux,uy,vx,vy,oux,ouy,ovx,ovy; 
       double   beta = gel->beta; 
       double   dt = gel->dt;
       double   uq_y[4],uq_x[4]; //transverse derivative
       double   lambda = gel->lambda;
 
       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
             //calculate half step value on right edge
              tmp[0] = tmp[1] = tmp[2] = tmp[3] = 0.0;

              xold[0] = q[0] = T11(gel,i,j); betaq[0] = beta*T11(gel,i,j);
              xold[1] = q[1] = T12(gel,i,j); betaq[1] = beta*T12(gel,i,j);
              xold[2] = q[2] = T22(gel,i,j); betaq[2] = beta*T22(gel,i,j);
              xold[3] = q[3] = Z(gel,i,j);   betaq[3] = beta*Z(gel,i,j);

              ql[0] = T11(gel,i-1,j); qr[0] = T11(gel,i+1,j);
              qb[0] = T11(gel,i,j-1); qt[0] = T11(gel,i,j+1);

              ql[1] = T12(gel,i-1,j); qr[1] = T12(gel,i+1,j);
              qb[1] = T12(gel,i,j-1); qt[1] = T12(gel,i,j+1);

              ql[2] = T22(gel,i-1,j); qr[2] = T22(gel,i+1,j);
              qb[2] = T22(gel,i,j-1); qt[2] = T22(gel,i,j+1);

              ql[3] = Z(gel,i-1,j); qr[3] = Z(gel,i+1,j);
              qb[3] = Z(gel,i,j-1); qt[3] = Z(gel,i,j+1);
 
              ux = (Un(gel,i,j)-Un(gel,i-1,j))/h;

              v1 = 0.5*(Un(gel,i,j+1)+Un(gel,i-1,j+1));
              v2 = 0.5*(Un(gel,i,j-1)+Un(gel,i-1,j-1));
              uy = (v1-v2)/(2*h);
             
              vy = (Vn(gel,i,j)-Vn(gel,i,j-1))/h;
             
              v1 = 0.5*(Vn(gel,i+1,j)+Vn(gel,i+1,j-1));
              v2 = 0.5*(Vn(gel,i-1,j)+Vn(gel,i-1,j-1));
              vx = (v1-v2)/(2*h); 

              uc = 0.5*(Un(gel,i,j)+Un(gel,i-1,j));
              vc = 0.5*(Vn(gel,i,j)+Vn(gel,i,j-1));

              B[0][0] = 2*ux;   B[0][1] = 2*uy;
              B[0][2] = 0;      B[0][3] = 2*ux;

              B[1][0] = vx;     B[1][1] = ux+vy;
              B[1][2] = uy;     B[1][3] = vx+uy;
 
              B[2][0] = 0;      B[2][1] = 2*vx;
              B[2][2] = 2*vy;   B[2][3] = 2*vy;
  
              B[3][0] = 0;      B[3][1] = 0;
              B[3][2] = 0;      B[3][3] = 0;
                       
              for(m=0; m<4; ++m)
                 for(n=0; n<4; ++n)
                    tmp[m] += B[m][n]*xold[n];
             
//first on east and west edge
              
              for(p=0; p<4; ++p)
              {
                 qxl[p] = 2*(q[p]-ql[p])/h; qxr[p] = 2*(qr[p]-q[p])/h; 
                 qxc[p] = (qr[p]-ql[p])/(2*h); 
                 qx[p] = minmod(qxl[p],qxr[p],qxc[p]); 
             
                 if(vc > 0)
                    qy[p] = (q[p]-qb[p])/h;  
                 else
                    qy[p] = (qt[p]-q[p])/h;

                 VTL(gel,i,j)[p] = q[p]+(0.5*h-0.5*dt*uc)*qx[p]-0.5*dt*q[p]*ux+0.5*dt*(tmp[p]-betaq[p]); 
                                    -0.5*dt*(vy*q[p]+vc*qy[p]);  
                 VTR(gel,i-1,j)[p] = q[p]+(-0.5*h-0.5*dt*uc)*qx[p]-0.5*dt*q[p]*ux+0.5*dt*(tmp[p]-betaq[p]);
                                    -0.5*dt*(vy*q[p]+vc*qy[p]);   
//then on north and south edge
 
                 qyb[p] = 2*(q[p]-qb[p])/h; qyt[p] = 2*(qt[p]-q[p])/h; 
                 qyc[p] = (qt[p]-qb[p])/(2*h); 
                 qy[p] = minmod(qyb[p],qyt[p],qyc[p]); 
            
                 if(uc > 0)
                    qx[p] = (q[p]-ql[p])/h;
                 else
                    qx[p] = (qr[p]-q[p])/h;
 
                 VTB(gel,i,j)[p] = q[p]+(0.5*h-0.5*dt*vc)*qy[p]-0.5*dt*q[p]*vy+0.5*dt*(tmp[p]-betaq[p]);
                                    -0.5*dt*(ux*q[p]+uc*qx[p]);  
                 VTT(gel,i,j-1)[p] = q[p]+(-0.5*h-0.5*dt*vc)*qy[p]-0.5*dt*q[p]*vy+0.5*dt*(tmp[p]-betaq[p]);
                                    -0.5*dt*(ux*q[p]+vc*qx[p]);   
             }
          }
       }

       update_ghost_testate_we(gel,gmax);
       update_buffer_testate(gel,gmax);

       for(i=lbuf[0]-1; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]-1; j<gmax[1]-ubuf[1]; ++j)
          {
	     for(p=0; p<4; ++p)
             {

//extraplate velocity to step k+1/2
                uh = 0.5*(3*Un(gel,i,j)-OUn(gel,i,j));
                vh = 0.5*(3*Vn(gel,i,j)-OVn(gel,i,j));

                FTx(gel,i,j)[p] = (uh > 0) ? VTL(gel,i,j)[p] : VTR(gel,i,j)[p];
	        FTy(gel,i,j)[p] = (vh > 0) ? VTB(gel,i,j)[p] : VTT(gel,i,j)[p];

                FTx(gel,i,j)[p] *= uh; //0.5*(3*Un(gel,i,j)-OUn(gel,i,j));
                FTy(gel,i,j)[p] *= vh; //0.5*(3*Vn(gel,i,j)-OVn(gel,i,j));
             }
          }
       }

       return;
}

static void update_dion_concentration(
       GEL    *gel,
       int    *gmax)
{
       int     size,indx,i,j,k,its;
       int     p,q,ip,jp,nlocal,sz,sz1;
       int     mx,my,mz,xs,ys,ztmp;
       int     irhs[1],idxm[1],col[5],gmax1[2];
       int     *ggmax = gel->rgr.ggmax;
       int     *pcrd=gel->pcrd;
       int     *pgmax=gel->pgmax;
       int     *gidx;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;

//diffusion coeff.
       double  ds=gel->ds; 
       double  dcl=gel->dcl;
       double  dca=gel->dca;

       double  zs = gel->zs;
       double  zca = gel->zca;
       double  zcl = gel->zcl;
       double  zt = gel->zt;
 
       double  kson = gel->kson;
       double  ksoff = gel->ksoff;
       double  kcon = gel->kcon;
       double  kcoff = gel->kcoff; 
       double  h = gel->rgr.h;
       double  hh,d[1],tmp[5],tmp1[5],tmp2[5],tmp3[5],tmp4[5],tmp5[5];
       double  dt=gel->dt;
       double  abs,norm,normb; 
       PetscScalar  sum[4]={1.0,1.0,1.0,-1.0}; 
       double  rs[1],rcl[1],rca[1],rg[1];

       char    fname[100]; 
       AppCtx  user;
       State   *st;
            
	    /*PETSC Variables*/
       DM              da;
       KSP             ksp;
       Mat             SRR;
       PC              pc;

       MatNullSpace         nullsp;

       ISLocalToGlobalMapping ltogm;
     
       static Vec      phi = NULL;
       Vec             tsr,tmpsr[4],FSR;
       Vec             nphi,fs,fcl,fca,rhs,rhcl,rhca;
       Vec             CS,CCL,CCA; //concentration 
       PetscScalar     *values,*vs,*vcl,*vca;
       //PetscViewer     viewer;
       hh = h*h;

       MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullsp);
       user.gel = gel;
 
       gmax1[0] = gmax[0]-lbuf[0]-ubuf[0];
       gmax1[1] = gmax[1]-lbuf[1]-ubuf[1];

       DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,DMDA_STENCIL_STAR,
                ggmax[0]-lbuf[0]-ubuf[0],ggmax[1]-lbuf[1]-ubuf[1],pgmax[0],pgmax[1],1,1,0,0,&da);
       
       sz = gmax1[0]*gmax1[1];
       
       MatCreateAIJ(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,5,PETSC_NULL,5,PETSC_NULL,&user.L);
       MatCreateAIJ(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,5,PETSC_NULL,5,PETSC_NULL,&user.LS); 
       MatCreateAIJ(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,5,PETSC_NULL,5,PETSC_NULL,&user.LCL);
       MatCreateAIJ(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,5,PETSC_NULL,5,PETSC_NULL,&user.LCA);
       MatCreateAIJ(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,5,PETSC_NULL,5,PETSC_NULL,&user.LPS);
       MatCreateAIJ(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,5,PETSC_NULL,5,PETSC_NULL,&user.LPCL);
       MatCreateAIJ(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,5,PETSC_NULL,5,PETSC_NULL,&user.LPCA);
       MatCreateAIJ(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,1,PETSC_NULL,1,PETSC_NULL,&user.DI);

       MatCreateShell(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,&user,&user.LSinv);
       MatCreateShell(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,&user,&user.LCLinv);
       MatCreateShell(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,&user,&user.LCAinv);
      
       MatShellSetOperation(user.LSinv,MATOP_MULT,(void(*)(void))UserMultLSinv);   // To find LS inverse using GMRES
       MatShellSetOperation(user.LCLinv,MATOP_MULT,(void(*)(void))UserMultLCLinv);  // To find LCL inverse using GMRES
       MatShellSetOperation(user.LCAinv,MATOP_MULT,(void(*)(void))UserMultLCAinv); // To find LCA inverse using GMRES

       MatShellSetContext(user.LSinv,&user); MatShellSetContext(user.LCLinv,&user);
       MatShellSetContext(user.LCAinv,&user); 

       VecCreate(PETSC_COMM_WORLD,&fs);
       VecSetSizes(fs,sz,PETSC_DECIDE);
       VecSetFromOptions(fs);

       VecCreate(PETSC_COMM_WORLD,&fcl);
       VecSetSizes(fcl,sz,PETSC_DECIDE);
       VecSetFromOptions(fcl);

       VecCreate(PETSC_COMM_WORLD,&fca);
       VecSetSizes(fca,sz,PETSC_DECIDE);
       VecSetFromOptions(fca);

       if(phi == NULL)
          VecDuplicate(fca,&phi);

       VecDuplicate(fca,&tsr);
 
       VecCreate(PETSC_COMM_WORLD,&user.tmp1); VecSetSizes(user.tmp1,sz,PETSC_DECIDE); 
       VecCreate(PETSC_COMM_WORLD,&user.tmp2); VecSetSizes(user.tmp2,sz,PETSC_DECIDE);
       VecSetFromOptions(user.tmp1); VecSetFromOptions(user.tmp2);
       VecAssemblyBegin(user.tmp1);   VecAssemblyEnd(user.tmp1);
       VecAssemblyBegin(user.tmp2);   VecAssemblyEnd(user.tmp2);

       for (k=0; k<4; ++k)
       {
          VecCreate(PETSC_COMM_WORLD,&tmpsr[k]); VecSetSizes(tmpsr[k],sz,PETSC_DECIDE);
          VecSetFromOptions(tmpsr[k]);

          VecCreate(PETSC_COMM_WORLD,&user.tmp[k]); VecSetSizes(user.tmp[k],sz,PETSC_DECIDE);
          VecSetFromOptions(user.tmp[k]);
       }

//matrix for Schur Complement System

       MatCreateShell(PETSC_COMM_WORLD,sz,sz,PETSC_DECIDE,PETSC_DECIDE,&user,&SRR);
       MatShellSetOperation(SRR,MATOP_MULT,(void(*)(void))UserMultSRR);
       MatShellSetContext(SRR,&user);
       
       DMGetLocalToGlobalMapping(da,&ltogm);
       ISLocalToGlobalMappingGetIndices(ltogm,&gidx);
       DMDAGetGhostCorners(da,&xs,&ys,&ztmp,&mx,&my,&mz);

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          ip = i-lbuf[0]+1; //one layer of ghost cell 

	     for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
	     {
//ip and jp are local index of cell[i,j] with ghost_cell
	     jp = j-lbuf[1]+1;

	     st = &(Rect_state(i,j,gel));
            
             d[0] = st->ths[1][1];
             tmp[0] = -st->ths[0][1]-st->ths[2][1]-st->ths[1][0]-st->ths[1][2]; // Coefficient of C(i,j)  
             tmp[1] = st->ths[0][1]; // Coefficient of C(i-1,j)  
			 tmp[2] = st->ths[2][1]; // Coefficient of C(i+1,j)
             tmp[3] = st->ths[1][0]; // Coefficient of C(i,j-1)
			 tmp[4] = st->ths[1][2]; // Coefficient of C(i,j+1)

             ///printf("\n cell[%d %d] with theta=[%f %f %f %f]",i,j,st->ths[0][1],st->ths[2][1],st->ths[1][0],st->ths[1][2]);
             // Elements of LPS matrix
             tmp1[0] = -st->ths[0][1]*0.5*(ES(gel,i,j)+ES(gel,i-1,j));   // what is tmp1[0] the cofficient of?
             tmp1[0] += -st->ths[2][1]*(ES(gel,i,j)+ES(gel,i+1,j));
             tmp1[0] += -st->ths[1][0]*(ES(gel,i,j)+ES(gel,i,j-1));
             tmp1[0] += -st->ths[1][2]*(ES(gel,i,j)+ES(gel,i,j+1));

			 // Coefficients of phi(i-0,5,j), phi(i+0.5,j)....
             tmp1[1] = st->ths[0][1]*0.5*(ES(gel,i,j)+ES(gel,i-1,j));   // why is there no minus signs in front of theta(i-1/2,j) and theta(i,j-1/2)?
             tmp1[2] = st->ths[2][1]*0.5*(ES(gel,i,j)+ES(gel,i+1,j));
             tmp1[3] = st->ths[1][0]*0.5*(ES(gel,i,j)+ES(gel,i,j-1));
             tmp1[4] = st->ths[1][2]*0.5*(ES(gel,i,j)+ES(gel,i,j+1));

			 // Elements of LPCL matrix
             tmp2[0] = -st->ths[0][1]*0.5*(ECL(gel,i,j)+ECL(gel,i-1,j));
             tmp2[0] += -st->ths[2][1]*(ECL(gel,i,j)+ECL(gel,i+1,j));
             tmp2[0] += -st->ths[1][0]*(ECL(gel,i,j)+ECL(gel,i,j-1));
             tmp2[0] += -st->ths[1][2]*(ECL(gel,i,j)+ECL(gel,i,j+1));

             tmp2[1] = st->ths[0][1]*0.5*(ECL(gel,i,j)+ECL(gel,i-1,j));
             tmp2[2] = st->ths[2][1]*0.5*(ECL(gel,i,j)+ECL(gel,i+1,j));
             tmp2[3] = st->ths[1][0]*0.5*(ECL(gel,i,j)+ECL(gel,i,j-1));
             tmp2[4] = st->ths[1][2]*0.5*(ECL(gel,i,j)+ECL(gel,i,j+1));

			 // Elements of LPCA matrix
             tmp3[0] = -st->ths[0][1]*0.5*(ECA(gel,i,j)+ECA(gel,i-1,j));
             tmp3[0] += -st->ths[2][1]*(ECA(gel,i,j)+ECA(gel,i+1,j));
             tmp3[0] += -st->ths[1][0]*(ECA(gel,i,j)+ECA(gel,i,j-1));
             tmp3[0] += -st->ths[1][2]*(ECA(gel,i,j)+ECA(gel,i,j+1));

             tmp3[1] = st->ths[0][1]*0.5*(ECA(gel,i,j)+ECA(gel,i-1,j));
             tmp3[2] = st->ths[2][1]*0.5*(ECA(gel,i,j)+ECA(gel,i+1,j));
             tmp3[3] = st->ths[1][0]*0.5*(ECA(gel,i,j)+ECA(gel,i,j-1));
             tmp3[4] = st->ths[1][2]*0.5*(ECA(gel,i,j)+ECA(gel,i,j+1));

             irhs[0] = idxm[0] = gidx[jp*mx+ip]; //row index in global matrix
	     
             col[0] = idxm[0];
	     col[1] = gidx[jp*mx+ip-1];
             col[2] = gidx[jp*mx+ip+1];
   	     col[3] = gidx[jp*mx+ip-mx];
	     col[4] = gidx[jp*mx+ip+mx];

//zero-flux boundary conditions	    

             if(j==lbuf[1]) //lower wall 
             {
               tmp[0] += st->ths[1][0];
               tmp1[0] += st->ths[1][0]*0.5*(ES(gel,i,j)+ES(gel,i,j-1));
               tmp2[0] += st->ths[1][0]*0.5*(ECL(gel,i,j)+ECL(gel,i,j-1));
               tmp3[0] += st->ths[1][0]*0.5*(ECA(gel,i,j)+ECA(gel,i,j-1));
             }
             if(j==gmax[1]-2) //upper wall 
             {
               tmp[0] += st->ths[1][2];
               tmp1[0] += st->ths[1][2]*0.5*(ES(gel,i,j)+ES(gel,i,j+1));
               tmp2[0] += st->ths[1][2]*0.5*(ECL(gel,i,j)+ECL(gel,i,j+1));
               tmp3[0] += st->ths[1][2]*0.5*(ECA(gel,i,j)+ECA(gel,i,j+1)); 
             }
             if(i==lbuf[0]) //left wall
             {
               tmp[0] += st->ths[0][1];
               tmp1[0] += st->ths[0][1]*0.5*(ES(gel,i,j)+ES(gel,i-1,j));
               tmp2[0] += st->ths[0][1]*0.5*(ECL(gel,i,j)+ECL(gel,i-1,j));
               tmp3[0] += st->ths[0][1]*0.5*(ECA(gel,i,j)+ECA(gel,i-1,j));
             }
             if(i==gmax[0]-2) //right wall
             {
               tmp[0] += st->ths[2][1];
               tmp1[0] += st->ths[2][1]*0.5*(ES(gel,i,j)+ES(gel,i+1,j));
               tmp2[0] += st->ths[2][1]*0.5*(ECL(gel,i,j)+ECL(gel,i+1,j));
               tmp3[0] += st->ths[2][1]*0.5*(ECA(gel,i,j)+ECA(gel,i+1,j));
             }
  
             for (k=0; k<5; ++k)
             {
                tmp1[k] *= -dt*ds*zs/(hh*st->ths[1][1]); tmp2[k] *= -dt*dcl*zcl/(hh*st->ths[1][1]);
                tmp3[k] *= -dt*dca*zca/(hh*st->ths[1][1]);
             }

             MatSetValues(user.L,1,idxm,5,col,tmp,INSERT_VALUES);
             MatSetValues(user.DI,1,idxm,1,idxm,d,INSERT_VALUES);

			//matrix wrt to potential
             MatSetValues(user.LPS,1,idxm,5,col,tmp1,INSERT_VALUES);     // matrices that act on Phi
             MatSetValues(user.LPCL,1,idxm,5,col,tmp2,INSERT_VALUES);
             MatSetValues(user.LPCA,1,idxm,5,col,tmp3,INSERT_VALUES);

//now use tmp to set tmp1-5
			 // The following sets up L_Na, L_Ca, L_C2
             for(k=0; k<5; ++k)
             {
                tmp1[k] = -dt*ds*tmp[k]/(Ths(gel,i,j)*hh);  // note: ds = Diffusion constant for sodium	
                tmp2[k] = -dt*dcl*tmp[k]/(Ths(gel,i,j)*hh); // note: dcl  = Diffusion constant for chloride	
                tmp3[k] = -dt*dca*tmp[k]/(Ths(gel,i,j)*hh); // note: dca  = Diffusion constant for calcium
             }
//add terms other than the laplacian

             //available binding sites
             abs = zt*Thn(gel,i,j)-BS(gel,i,j)-BCA(gel,i,j)-2.0*BC2(gel,i,j);
             abs = max(0.0, abs);
             ABS(gel,i,j) = abs; 

			
			// These are different than tmp1[0] than above. These correspond to the diagonal entries of the LS,LCL,LCA matrices. 
			// They act on the unknowns C_Na, C_Ca, C_Cl at the next time step (k+1) 
            //sodium
             tmp1[0] += 1.0 + dt*kson*abs;
             //chloride
             tmp2[0] += 1.0;
             //calcium
             tmp3[0] += 1.0 + dt*kcon*abs;

             //printf("\n cell[%d %d] with tmp=[%5.4e %5.4e %5.4e %5.4e]",i,j,tmp1[1],tmp1[2],tmp1[3],tmp1[4]);
            
			 //right hand side of the vector
             rs[0] = S(gel,i,j) + dt*ksoff*Ths(gel,i,j)*BS(gel,i,j);
             rcl[0] = CL(gel,i,j);
             rca[0] = CA(gel,i,j) + dt*kcoff*Ths(gel,i,j)*BCA(gel,i,j); 
             
//right hand side of neutrality condition
             rg[0] = zt*Thn(gel,i,j)-BS(gel,i,j)-2.0*BCA(gel,i,j)-2.0*BC2(gel,i,j); 

             MatSetValues(user.LS,1,idxm,5,col,tmp1,INSERT_VALUES);
             MatSetValues(user.LCL,1,idxm,5,col,tmp2,INSERT_VALUES);
             MatSetValues(user.LCA,1,idxm,5,col,tmp3,INSERT_VALUES); 
 
             VecSetValues(fs,1,idxm,rs,INSERT_VALUES);
             VecSetValues(fcl,1,idxm,rcl,INSERT_VALUES);
             VecSetValues(fca,1,idxm,rca,INSERT_VALUES);
             VecSetValues(tmpsr[3],1,idxm,rg,INSERT_VALUES); 
	     }
       }
    
       MatAssemblyBegin(user.L,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(user.L,MAT_FINAL_ASSEMBLY);

       MatAssemblyBegin(user.DI,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(user.DI,MAT_FINAL_ASSEMBLY);

       MatAssemblyBegin(user.LS,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(user.LS,MAT_FINAL_ASSEMBLY);

       MatAssemblyBegin(user.LPS,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(user.LPS,MAT_FINAL_ASSEMBLY);

       MatAssemblyBegin(user.LCL,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(user.LCL,MAT_FINAL_ASSEMBLY);

       MatAssemblyBegin(user.LPCL,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(user.LPCL,MAT_FINAL_ASSEMBLY);

       MatAssemblyBegin(user.LCA,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(user.LCA,MAT_FINAL_ASSEMBLY);

       MatAssemblyBegin(user.LPCA,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(user.LPCA,MAT_FINAL_ASSEMBLY);

       MatAssemblyBegin(user.LSinv,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(user.LSinv,MAT_FINAL_ASSEMBLY);

       MatAssemblyBegin(user.LCLinv,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(user.LCLinv,MAT_FINAL_ASSEMBLY);

       MatAssemblyBegin(user.LCAinv,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(user.LCAinv,MAT_FINAL_ASSEMBLY);

       MatAssemblyBegin(SRR,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(SRR,MAT_FINAL_ASSEMBLY);

       VecAssemblyBegin(fs);   VecAssemblyEnd(fs);
       VecAssemblyBegin(fcl);   VecAssemblyEnd(fcl); 
       VecAssemblyBegin(fca);   VecAssemblyEnd(fca);
 
       for(k=0; k<4; ++k)
       {
         VecAssemblyBegin(user.tmp[k]);   VecAssemblyEnd(user.tmp[k]);
         VecAssemblyBegin(tmpsr[k]);   VecAssemblyEnd(tmpsr[k]);
       }
       ISLocalToGlobalMappingRestoreIndices(ltogm,&gidx);

       MatMult(user.LSinv,fs,tsr);

       MatMult(user.DI,tsr,tmpsr[0]); VecScale(tmpsr[0],zs);
 
       MatMult(user.LCLinv,fcl,tsr);
       MatMult(user.DI,tsr,tmpsr[1]); VecScale(tmpsr[1],zcl);

       MatMult(user.LCAinv,fca,tsr);
       MatMult(user.DI,tsr,tmpsr[2]); VecScale(tmpsr[2],zca);

       VecDuplicate(fca,&FSR); VecSet(FSR,0.0); 

       /*sprintf(fname,"vecfsr");
       PetscViewerASCIIOpen(PETSC_COMM_WORLD,fname,&viewer);
       VecView(FSR,viewer);
       exit(0);*/

       VecMAXPY(FSR,4,sum,tmpsr);

       KSPCreate(PETSC_COMM_WORLD,&ksp);
       KSPGetPC(ksp,&pc);
       PCSetType(pc,PCNONE);

       MatSetNullSpace(SRR,nullsp);
       KSPSetOperators(ksp,SRR,SRR);

       KSPSetType(ksp,KSPGMRES); KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);

       KSPSetTolerances(ksp,1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,30);

       KSPSetFromOptions(ksp);
       KSPSetUp(ksp);

       KSPSolve(ksp,FSR,phi);

       KSPGetIterationNumber(ksp,&its);
       KSPGetResidualNorm(ksp,&norm);
       VecNorm(FSR,NORM_2,&normb);
 
       if(gel->my_id ==0)
         printf("\n GMRES solve potential,iter=%d with rhs=%6.5e, relative residual=%6.5e",its,normb,norm/normb);

       VecGetArray(phi,&values);

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j) 
	  {
	        indx = (j-lbuf[1])*gmax1[0]+(i-lbuf[0]);
	       
                Phi(gel,i,j) = values[indx];
          }
       }     

       VecDuplicate(fca,&rhs); VecDuplicate(fca,&nphi); 
       VecDuplicate(fca,&rhcl); VecDuplicate(fca,&rhca);
       VecDuplicate(fca,&CS); VecDuplicate(fca,&CCL); VecDuplicate(fca,&CCA);

//compute negative phi
       VecSet(nphi,0.0); VecAXPY(nphi,-1.0,phi);

       MatMultAdd(user.LPS,nphi,fs,rhs); MatMultAdd(user.LPCL,nphi,fcl,rhcl);
       MatMultAdd(user.LPCA,nphi,fca,rhca);

       MatMult(user.LSinv,rhs,CS); MatMult(user.LCLinv,rhcl,CCL);
       MatMult(user.LCAinv,rhca,CCA); 

       VecGetArray(CS,&vs); VecGetArray(CCL,&vcl); VecGetArray(CCA,&vca);

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j) 
	  {
	        indx = (j-lbuf[1])*gmax1[0]+(i-lbuf[0]);
	        S(gel,i,j) = max(0.0,vs[indx]);
                CL(gel,i,j) = max(0.0,vcl[indx]);
                CA(gel,i,j) = max(0.0,vca[indx]);     
          }
       }
   
       VecRestoreArray(CS,&vs); VecRestoreArray(CCL,&vcl);
       VecRestoreArray(phi,&values); VecRestoreArray(CCA,&vca);
       
       VecDestroy(&FSR);  VecDestroy(&nphi); VecDestroy(&rhs); VecDestroy(&rhcl);
       VecDestroy(&rhca); VecDestroy(&fs); VecDestroy(&fcl); VecDestroy(&fca);
       VecDestroy(&tsr); VecDestroy(&CS); VecDestroy(&CCL); VecDestroy(&CCA);
       for(i=0; i<4; ++i)
       {
          VecDestroy(&user.tmp[i]); VecDestroy(&tmpsr[i]);
       }
       VecDestroy(&user.tmp1); VecDestroy(&user.tmp2);
       
       MatDestroy(&user.L); MatDestroy(&user.LS); MatDestroy(&user.LCL); MatDestroy(&user.LCA);
       MatDestroy(&user.LPS); MatDestroy(&user.LPCL); MatDestroy(&user.LPCA); 
       MatDestroy(&user.LSinv); MatDestroy(&user.LCLinv); MatDestroy(&user.LCAinv);

       MatDestroy(&user.DI); MatDestroy(&SRR);

       KSPDestroy(&ksp); DMDestroy(&da);
       MatNullSpaceDestroy(&nullsp);
}

PetscErrorCode UserMultSRR(Mat SRR,Vec x,Vec y)
{
       AppCtx          *user;
       //char            fname[100];
       //PetscViewer     viewer;

       MatShellGetContext(SRR,(void**)&user);

       double          zs = user->gel->zs;
       double          zcl = user->gel->zcl;
       double          zca = user->gel->zca; 


       double          h = user->gel->rgr.h; 
       double          hh = h*h;
       PetscScalar     sum[3]={1.0,1.0,1.0};

       MatMult(user->LPS,x,user->tmp1);
       MatMult(user->LSinv,user->tmp1,user->tmp2);
       MatMult(user->DI,user->tmp2,user->tmp[0]);
       VecScale(user->tmp[0],zs);

       MatMult(user->LPCL,x,user->tmp1);
       MatMult(user->LCLinv,user->tmp1,user->tmp2);
       MatMult(user->DI,user->tmp2,user->tmp[1]);
       VecScale(user->tmp[1],zcl);
       
       MatMult(user->LPCA,x,user->tmp1);
       MatMult(user->LCAinv,user->tmp1,user->tmp2);
       MatMult(user->DI,user->tmp2,user->tmp[2]);
       VecScale(user->tmp[2],zca);

       /*{
         sprintf(fname,"vecypre");
         PetscViewerASCIIOpen(PETSC_COMM_WORLD,fname,&viewer);
         VecView(y,viewer);
       }*/

       VecSet(y,0.0); VecMAXPY(y,3,sum,user->tmp);
       return 0;
}

PetscErrorCode UserMultLSinv(Mat LSinv,Vec x,Vec y)
{
       AppCtx          *user;

       MatShellGetContext(LSinv,(void**)&user);

       int          its;
       double       norm,normb;

            /*PETSC Variables*/
       KSP                  ksp;
       PC                   pc;

       KSPCreate(PETSC_COMM_WORLD,&ksp);
       KSPSetOperators(ksp,user->LS,user->LS);

       KSPGetPC(ksp,&pc); PCSetType(pc,PCSHELL);

       user->gel->opt = 1; //with respect to sodium 

       PCShellSetApply(pc,SampleShellPCApply1);
       PCShellSetContext(pc,user);

       KSPSetType(ksp,KSPGMRES);
       KSPSetTolerances(ksp,1.0e-12,PETSC_DEFAULT,PETSC_DEFAULT,40);
       
       KSPSetFromOptions(ksp);  KSPSetUp(ksp);
    
       KSPSolve(ksp,x,y);
       KSPGetIterationNumber(ksp,&its);
       KSPGetResidualNorm(ksp,&norm);
       VecNorm(x,NORM_2,&normb);

       printf("\n LSinv GMRES its=%d with residual=%6.5e and relative residual=%6.5e",its,norm,norm/normb);
     
       KSPDestroy(&ksp); return 0;
}

PetscErrorCode UserMultLCLinv(Mat LCLinv,Vec x,Vec y)
{
       AppCtx          *user;

       MatShellGetContext(LCLinv,(void**)&user);

       int          its;
       double       norm,normb;

            /*PETSC Variables*/
       KSP                  ksp;
       PC                   pc;

       KSPCreate(PETSC_COMM_WORLD,&ksp);
       KSPSetOperators(ksp,user->LCL,user->LCL);

       KSPGetPC(ksp,&pc); PCSetType(pc,PCSHELL);

       user->gel->opt = 0; //with respect to chloride 

       PCShellSetApply(pc,SampleShellPCApply1);
       PCShellSetContext(pc,user);

       KSPSetType(ksp,KSPGMRES);
       KSPSetTolerances(ksp,1.0e-12,PETSC_DEFAULT,PETSC_DEFAULT,40);
       
       KSPSetFromOptions(ksp);  KSPSetUp(ksp);
    
       KSPSolve(ksp,x,y);
       KSPGetIterationNumber(ksp,&its);
       KSPGetResidualNorm(ksp,&norm);
       VecNorm(x,NORM_2,&normb);

       printf("\n LCLinv GMRES its=%d with residual=%6.5e and relative residual=%6.5e",its,norm,norm/normb);
      
       KSPDestroy(&ksp); return 0;
}

PetscErrorCode UserMultLCAinv(Mat LCAinv,Vec x,Vec y)
{
       AppCtx          *user;

       MatShellGetContext(LCAinv,(void**)&user);

       int          its;
       double       norm,normb;

            /*PETSC Variables*/
       KSP                  ksp;
       PC                   pc;

       KSPCreate(PETSC_COMM_WORLD,&ksp);
       KSPSetOperators(ksp,user->LCA,user->LCA);

       KSPGetPC(ksp,&pc); PCSetType(pc,PCSHELL);

       user->gel->opt = 2; //with respect to calcium 

       PCShellSetApply(pc,SampleShellPCApply1);
       PCShellSetContext(pc,user);

       KSPSetType(ksp,KSPGMRES);
       KSPSetTolerances(ksp,1.0e-12,PETSC_DEFAULT,PETSC_DEFAULT,40);
       
       KSPSetFromOptions(ksp);  KSPSetUp(ksp);
    
       KSPSolve(ksp,x,y);
       KSPGetIterationNumber(ksp,&its);
       KSPGetResidualNorm(ksp,&norm);
       VecNorm(x,NORM_2,&normb);

       printf("\n LCAinv GMRES its=%d with residual=%6.5e and relative residual=%6.5e",its,norm,norm/normb);
      
       KSPDestroy(&ksp); return 0;
}


static void update_edge_flux_dion1(
       GEL     *gel,
       int     *gmax,
       double  dt,
       double  h)
{
       int     i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;

       double  Tx,Ty; //partial derivatives 
       double  Ux,Vy,Uc,Vc; //cell centered velocity

       double  Sx,Sy,Cx,Cy,CAx,CAy;
       double  Sxc,Sxl,Sxr,Syc,Syl,Syu;
       double  Cxc,Cxl,Cxr,Cyc,Cyl,Cyu;
       double  ush,vsh,CAxc,CAxl,CAxr,CAyc,CAyl,CAyu;

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
             Uc = 0.5*(Us(gel,i,j)+Us(gel,i-1,j));
             Vc = 0.5*(Vs(gel,i,j)+Vs(gel,i,j-1));

	     Ux = (Us(gel,i,j)-Us(gel,i-1,j))/h;
	     Vy = (Vs(gel,i,j)-Vs(gel,i,j-1))/h;

             Sxc = (S(gel,i+1,j)-S(gel,i-1,j))/(2*h);
             Sxl = 2*(S(gel,i,j)-S(gel,i-1,j))/h;
             Sxr = 2*(S(gel,i+1,j)-S(gel,i,j))/h;

             CAxc = (CA(gel,i+1,j)-CA(gel,i-1,j))/(2*h);
             CAxl = 2*(CA(gel,i,j)-CA(gel,i-1,j))/h;
             CAxr = 2*(CA(gel,i+1,j)-CA(gel,i,j))/h;

             Cxc = (CL(gel,i+1,j)-CL(gel,i-1,j))/(2*h);
             Cxl = 2*(CL(gel,i,j)-CL(gel,i-1,j))/h;
             Cxr = 2*(CL(gel,i+1,j)-CL(gel,i,j))/h;

             Sx = minmod(Sxc,Sxl,Sxr);
             Cx = minmod(Cxc,Cxl,Cxr);
             CAx = minmod(CAxc,CAxl,CAxr);
 
	     //upwinding for transverse derivative
	     if(Vc > 0)
             {
                Sy = (S(gel,i,j)-S(gel,i,j-1))/h;
                Cy = (CL(gel,i,j)-CL(gel,i,j-1))/h;
                CAy = (CA(gel,i,j)-CA(gel,i,j-1))/h;
             }
             else
             {
                Sy = (S(gel,i,j+1)-S(gel,i,j))/h;
                Cy = (CL(gel,i,j+1)-CL(gel,i,j))/h;
                CAy = (CA(gel,i,j+1)-CA(gel,i,j))/h;
             }
	    
             CLL(gel,i,j) = CL(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*CL(gel,i,j)-CL(gel,i,j)*Vy-Vc*Cy) + 0.5*h*Cx;
             SL(gel,i,j) = S(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*S(gel,i,j)-S(gel,i,j)*Vy-Vc*Sy)+0.5*h*Sx;
             CAL(gel,i,j) = CA(gel,i,j)+0.5*dt*(-CAx*Uc-Ux*CA(gel,i,j)-CA(gel,i,j)*Vy-Vc*CAy)+0.5*h*CAx;
	     
             CLR(gel,i-1,j) = CL(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*CL(gel,i,j)-CL(gel,i,j)*Vy-Vc*Cy) - 0.5*h*Cx;
             SR(gel,i-1,j) = S(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*S(gel,i,j)-S(gel,i,j)*Vy-Vc*Sy)-0.5*h*Sx;
             CAR(gel,i-1,j) = CA(gel,i,j)+0.5*dt*(-CAx*Uc-Ux*CA(gel,i,j)-CA(gel,i,j)*Vy-Vc*CAy)-0.5*h*CAx;

//the other direction 
             Syc = (S(gel,i,j+1)-S(gel,i,j-1))/(2*h);
             Syl = 2*(S(gel,i,j)-S(gel,i,j-1))/h;
             Syu = 2*(S(gel,i,j+1)-S(gel,i,j))/h;

             CAyc = (CA(gel,i,j+1)-CA(gel,i,j-1))/(2*h);
             CAyl = 2*(CA(gel,i,j)-CA(gel,i,j-1))/h;
             CAyu = 2*(CA(gel,i,j+1)-CA(gel,i,j))/h;

             Cyc = (CL(gel,i,j+1)-CL(gel,i,j-1))/(2*h);
             Cyl = 2*(CL(gel,i,j)-CL(gel,i,j-1))/h;
             Cyu = 2*(CL(gel,i,j+1)-CL(gel,i,j))/h;

             Sy = minmod(Syc,Syl,Syu);
             CAy = minmod(CAyc,CAyl,CAyu);
             Cy = minmod(Cyc,Cyl,Cyu);
           
	     //upwinding for transverse derivative
	     
	     if(Uc > 0)
             {
                Cx = (CL(gel,i,j)-CL(gel,i-1,j))/h;
                Sx = (S(gel,i,j)-S(gel,i-1,j))/h;
                CAx = (CA(gel,i,j)-CA(gel,i-1,j))/h;
             }
             else
             {
                Cx = (CL(gel,i+1,j)-CL(gel,i,j))/h;
                Sx = (S(gel,i+1,j)-S(gel,i,j))/h;
                CAx = (CA(gel,i+1,j)-CA(gel,i,j))/h;
             }

             SB(gel,i,j) = S(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*S(gel,i,j)-S(gel,i,j)*Vy-Vc*Sy)+0.5*h*Sy;
             ST(gel,i,j-1) = S(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*S(gel,i,j)-S(gel,i,j)*Vy-Vc*Sy)-0.5*h*Sy;

             CAB(gel,i,j) = CA(gel,i,j)+0.5*dt*(-CAx*Uc-Ux*CA(gel,i,j)-CA(gel,i,j)*Vy-Vc*CAy)+0.5*h*CAy;
             CAT(gel,i,j-1) = CA(gel,i,j)+0.5*dt*(-CAx*Uc-Ux*CA(gel,i,j)-CA(gel,i,j)*Vy-Vc*CAy)-0.5*h*CAy;

             CLB(gel,i,j) = CL(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*CL(gel,i,j)-CL(gel,i,j)*Vy-Vc*Cy) + 0.5*h*Cy;
             CLT(gel,i,j-1) = CL(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*CL(gel,i,j)-CL(gel,i,j)*Vy-Vc*Cy) - 0.5*h*Cy;
	  }
       }

       //parallel buffer edge state 
       //update_ghost_estate_sn(gel,gmax);
       update_ghost_estate_we(gel,gmax);
 
       for(i=lbuf[0]-1; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]-1; j<gmax[1]-ubuf[1]; ++j)
          {
             if(gel->step == 0)
             {
               ush = Us(gel,i,j); vsh = Vs(gel,i,j);
             }
             else
             {
               ush = 0.5*(3*Us(gel,i,j)-OUs(gel,i,j));
               vsh = 0.5*(3*Vs(gel,i,j)-OVs(gel,i,j));
             }

             FSx(gel,i,j) = (ush > 0) ? SL(gel,i,j) : SR(gel,i,j);
             FSy(gel,i,j) = (vsh > 0) ? SB(gel,i,j) : ST(gel,i,j);

             FSx(gel,i,j) *= ush;
             FSy(gel,i,j) *= vsh;

             FCLx(gel,i,j) = (ush > 0) ? CLL(gel,i,j) : CLR(gel,i,j);
             FCLy(gel,i,j) = (vsh > 0) ? CLB(gel,i,j) : CLT(gel,i,j);

             FCLx(gel,i,j) *= ush;

             FCLy(gel,i,j) *= vsh;

             FCAx(gel,i,j) = (ush > 0) ? CAL(gel,i,j) : CAR(gel,i,j);
             FCAy(gel,i,j) = (vsh > 0) ? CAB(gel,i,j) : CAT(gel,i,j);

             FCAx(gel,i,j) *= ush;
             FCAy(gel,i,j) *= vsh;
	  }
       }
       return;
}

//flux for dissolved ion 
static void update_edge_flux_dion(
       GEL     *gel,
       int     *gmax,
       double  dt,
       double  h)
{
       int     i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;

       double  Tx,Ty; //partial derivatives 
       double  Ux,Vy,Uc,Vc; //cell centered velocity

       double  Sx,Sy,Cx,Cy,CAx,CAy;
       double  Sxc,Sxl,Sxr,Syc,Syl,Syu;
       double  Cxc,Cxl,Cxr,Cyc,Cyl,Cyu;
       double  ush,vsh,CAxc,CAxl,CAxr,CAyc,CAyl,CAyu;

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
             Uc = 0.5*(Us(gel,i,j)+Us(gel,i-1,j));
             Vc = 0.5*(Vs(gel,i,j)+Vs(gel,i,j-1));

	     Ux = (Us(gel,i,j)-Us(gel,i-1,j))/h;
	     Vy = (Vs(gel,i,j)-Vs(gel,i,j-1))/h;

             Sxc = (OThs(gel,i+1,j)*S(gel,i+1,j)-OThs(gel,i-1,j)*S(gel,i-1,j))/(2*h);
             Sxl = 2*(OThs(gel,i,j)*S(gel,i,j)-OThs(gel,i-1,j)*S(gel,i-1,j))/h;
             Sxr = 2*(OThs(gel,i+1,j)*S(gel,i+1,j)-OThs(gel,i,j)*S(gel,i,j))/h;

             CAxc = (OThs(gel,i+1,j)*CA(gel,i+1,j)-OThs(gel,i-1,j)*CA(gel,i-1,j))/(2*h);
             CAxl = 2*(OThs(gel,i,j)*CA(gel,i,j)-OThs(gel,i-1,j)*CA(gel,i-1,j))/h;
             CAxr = 2*(OThs(gel,i+1,j)*CA(gel,i+1,j)-OThs(gel,i,j)*CA(gel,i,j))/h;

             Cxc = (OThs(gel,i+1,j)*CL(gel,i+1,j)-OThs(gel,i-1,j)*CL(gel,i-1,j))/(2*h);
             Cxl = 2*(OThs(gel,i,j)*CL(gel,i,j)-OThs(gel,i-1,j)*CL(gel,i-1,j))/h;
             Cxr = 2*(OThs(gel,i+1,j)*CL(gel,i+1,j)-OThs(gel,i,j)*CL(gel,i,j))/h;

             Sx = minmod(Sxc,Sxl,Sxr);
             Cx = minmod(Cxc,Cxl,Cxr);
             CAx = minmod(CAxc,CAxl,CAxr);
 
	     //upwinding for transverse derivative
	     if(Vc > 0)
             {
                Sy = (OThs(gel,i,j)*S(gel,i,j)-OThs(gel,i,j-1)*S(gel,i,j-1))/h;
                Cy = (OThs(gel,i,j)*CL(gel,i,j)-OThs(gel,i,j-1)*CL(gel,i,j-1))/h;
                CAy = (OThs(gel,i,j)*CA(gel,i,j)-OThs(gel,i,j-1)*CA(gel,i,j-1))/h;
             }
             else
             {
                Sy = (OThs(gel,i,j+1)*S(gel,i,j+1)-OThs(gel,i,j)*S(gel,i,j))/h;
                Cy = (OThs(gel,i,j+1)*CL(gel,i,j+1)-OThs(gel,i,j)*CL(gel,i,j))/h;
                CAy = (OThs(gel,i,j+1)*CA(gel,i,j+1)-OThs(gel,i,j)*CA(gel,i,j))/h;
             }
	    
             CLL(gel,i,j) = OThs(gel,i,j)*CL(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*OThs(gel,i,j)*CL(gel,i,j)-OThs(gel,i,j)*CL(gel,i,j)*Vy-Vc*Cy) + 0.5*h*Cx;
             SL(gel,i,j) = OThs(gel,i,j)*S(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*OThs(gel,i,j)*S(gel,i,j)-OThs(gel,i,j)*S(gel,i,j)*Vy-Vc*Sy)+0.5*h*Sx;
             CAL(gel,i,j) = OThs(gel,i,j)*CA(gel,i,j)+0.5*dt*(-CAx*Uc-Ux*OThs(gel,i,j)*CA(gel,i,j)-OThs(gel,i,j)*CA(gel,i,j)*Vy-Vc*CAy)+0.5*h*CAx;
	     
             CLR(gel,i-1,j) = OThs(gel,i,j)*CL(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*OThs(gel,i,j)*CL(gel,i,j)-OThs(gel,i,j)*CL(gel,i,j)*Vy-Vc*Cy) - 0.5*h*Cx;
             SR(gel,i-1,j) = OThs(gel,i,j)*S(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*OThs(gel,i,j)*S(gel,i,j)-OThs(gel,i,j)*S(gel,i,j)*Vy-Vc*Sy)-0.5*h*Sx;
             CAR(gel,i-1,j) = OThs(gel,i,j)*CA(gel,i,j)+0.5*dt*(-CAx*Uc-Ux*OThs(gel,i,j)*CA(gel,i,j)-OThs(gel,i,j)*CA(gel,i,j)*Vy-Vc*CAy)-0.5*h*CAx;

//the other direction 
             Syc = (OThs(gel,i,j+1)*S(gel,i,j+1)-OThs(gel,i,j-1)*S(gel,i,j-1))/(2*h);
             Syl = 2*(OThs(gel,i,j)*S(gel,i,j)-OThs(gel,i,j-1)*S(gel,i,j-1))/h;
             Syu = 2*(OThs(gel,i,j+1)*S(gel,i,j+1)-OThs(gel,i,j)*S(gel,i,j))/h;

             CAyc = (OThs(gel,i,j+1)*CA(gel,i,j+1)-OThs(gel,i,j-1)*CA(gel,i,j-1))/(2*h);
             CAyl = 2*(OThs(gel,i,j)*CA(gel,i,j)-OThs(gel,i,j-1)*CA(gel,i,j-1))/h;
             CAyu = 2*(OThs(gel,i,j+1)*CA(gel,i,j+1)-OThs(gel,i,j)*CA(gel,i,j))/h;

             Cyc = (OThs(gel,i,j+1)*CL(gel,i,j+1)-OThs(gel,i,j-1)*CL(gel,i,j-1))/(2*h);
             Cyl = 2*(OThs(gel,i,j)*CL(gel,i,j)-OThs(gel,i,j-1)*CL(gel,i,j-1))/h;
             Cyu = 2*(OThs(gel,i,j+1)*CL(gel,i,j+1)-OThs(gel,i,j)*CL(gel,i,j))/h;

             Sy = minmod(Syc,Syl,Syu);
             CAy = minmod(CAyc,CAyl,CAyu);
             Cy = minmod(Cyc,Cyl,Cyu);
           
	     //upwinding for transverse derivative
	     
	     if(Uc > 0)
             {
                Cx = (OThs(gel,i,j)*CL(gel,i,j)-OThs(gel,i-1,j)*CL(gel,i-1,j))/h;
                Sx = (OThs(gel,i,j)*S(gel,i,j)-OThs(gel,i-1,j)*S(gel,i-1,j))/h;
                CAx = (OThs(gel,i,j)*CA(gel,i,j)-OThs(gel,i-1,j)*CA(gel,i-1,j))/h;
             }
             else
             {
                Cx = (OThs(gel,i+1,j)*CL(gel,i+1,j)-OThs(gel,i,j)*CL(gel,i,j))/h;
                Sx = (OThs(gel,i+1,j)*S(gel,i+1,j)-OThs(gel,i,j)*S(gel,i,j))/h;
                CAx = (OThs(gel,i+1,j)*CA(gel,i+1,j)-OThs(gel,i,j)*CA(gel,i,j))/h;
             }

             SB(gel,i,j) = OThs(gel,i,j)*S(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*OThs(gel,i,j)*S(gel,i,j)-OThs(gel,i,j)*S(gel,i,j)*Vy-Vc*Sy)+0.5*h*Sy;
             ST(gel,i,j-1) = OThs(gel,i,j)*S(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*OThs(gel,i,j)*S(gel,i,j)-OThs(gel,i,j)*S(gel,i,j)*Vy-Vc*Sy)-0.5*h*Sy;

             CAB(gel,i,j) = OThs(gel,i,j)*CA(gel,i,j)+0.5*dt*(-CAx*Uc-Ux*OThs(gel,i,j)*CA(gel,i,j)-OThs(gel,i,j)*CA(gel,i,j)*Vy-Vc*CAy)+0.5*h*CAy;
             CAT(gel,i,j-1) = OThs(gel,i,j)*CA(gel,i,j)+0.5*dt*(-CAx*Uc-Ux*OThs(gel,i,j)*CA(gel,i,j)-OThs(gel,i,j)*CA(gel,i,j)*Vy-Vc*CAy)-0.5*h*CAy;

             CLB(gel,i,j) = OThs(gel,i,j)*CL(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*OThs(gel,i,j)*CL(gel,i,j)-OThs(gel,i,j)*CL(gel,i,j)*Vy-Vc*Cy) + 0.5*h*Cy;
             CLT(gel,i,j-1) = OThs(gel,i,j)*CL(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*OThs(gel,i,j)*CL(gel,i,j)-OThs(gel,i,j)*CL(gel,i,j)*Vy-Vc*Cy) - 0.5*h*Cy;
	  }
       }

       //parallel buffer edge state 
       //update_ghost_estate_sn(gel,gmax);
       update_ghost_estate_we(gel,gmax);
 
       for(i=lbuf[0]-1; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]-1; j<gmax[1]-ubuf[1]; ++j)
          {
             if(gel->step == 0)
             {
               ush = Us(gel,i,j); vsh = Vs(gel,i,j);
             }
             else
             {
               ush = 0.5*(3*Us(gel,i,j)-OUs(gel,i,j));
               vsh = 0.5*(3*Vs(gel,i,j)-OVs(gel,i,j));
             }

             FSx(gel,i,j) = (ush > 0) ? SL(gel,i,j) : SR(gel,i,j);
             FSy(gel,i,j) = (vsh > 0) ? SB(gel,i,j) : ST(gel,i,j);

             FSx(gel,i,j) *= ush;
             FSy(gel,i,j) *= vsh;

             FCLx(gel,i,j) = (ush > 0) ? CLL(gel,i,j) : CLR(gel,i,j);
             FCLy(gel,i,j) = (vsh > 0) ? CLB(gel,i,j) : CLT(gel,i,j);

             FCLx(gel,i,j) *= ush;
             FCLy(gel,i,j) *= vsh;

             FCAx(gel,i,j) = (ush > 0) ? CAL(gel,i,j) : CAR(gel,i,j);
             FCAy(gel,i,j) = (vsh > 0) ? CAB(gel,i,j) : CAT(gel,i,j);

             FCAx(gel,i,j) *= ush;
             FCAy(gel,i,j) *= vsh;
	  }
       }
       return;
}

//flux for bound ion 
static void update_edge_flux_bion(
       GEL     *gel,
       int     *gmax,
       double  dt,
       double  h)
{
       int     i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;

       double  hh = h*h;
       double  Tx,Ty; //partial derivatives 
       double  Ux,Vy,Uc,Vc; //cell centered velocity

       double  Sx,Sy,Cx,Cy,C2x,C2y;
       double  Sxc,Sxl,Sxr,Syc,Syl,Syu;
       double  unh,vnh,Cxc,Cxl,Cxr,Cyc,Cyl,Cyu;
       double  C2xc,C2xl,C2xr,C2yc,C2yl,C2yu;

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
             Uc = 0.5*(Un(gel,i,j)+Un(gel,i-1,j));
             Vc = 0.5*(Vn(gel,i,j)+Vn(gel,i,j-1));

	     Ux = (Un(gel,i,j)-Un(gel,i-1,j))/h;
	     Vy = (Vn(gel,i,j)-Vn(gel,i,j-1))/h;

             Sxc = (BS(gel,i+1,j)-BS(gel,i-1,j))/(2*h);
             Sxl = 2*(BS(gel,i,j)-BS(gel,i-1,j))/h;
             Sxr = 2*(BS(gel,i+1,j)-BS(gel,i,j))/h;

             C2xc = (BC2(gel,i+1,j)-BC2(gel,i-1,j))/(2*h);
             C2xl = 2*(BC2(gel,i,j)-BC2(gel,i-1,j))/h;
             C2xr = 2*(BC2(gel,i+1,j)-BC2(gel,i,j))/h;

             Cxc = (BCA(gel,i+1,j)-BCA(gel,i-1,j))/(2*h);
             Cxl = 2*(BCA(gel,i,j)-BCA(gel,i-1,j))/h;
             Cxr = 2*(BCA(gel,i+1,j)-BCA(gel,i,j))/h;

             Sx = minmod(Sxc,Sxl,Sxr);
             Cx = minmod(Cxc,Cxl,Cxr);
             C2x = minmod(C2xc,C2xl,C2xr);
 
	     //upwinding for transverse derivative
	     if(Vc > 0)
             {
                Sy = (BS(gel,i,j)-BS(gel,i,j-1))/h;
                Cy = (BCA(gel,i,j)-BCA(gel,i,j-1))/h;
                C2y = (BC2(gel,i,j)-BC2(gel,i,j-1))/h;
             }
             else
             {
                Sy = (BS(gel,i,j+1)-BS(gel,i,j))/h;
                Cy = (BCA(gel,i,j+1)-BCA(gel,i,j))/h;
                C2y = (BC2(gel,i,j+1)-BC2(gel,i,j))/h;
             }
	    
             CAL(gel,i,j) = BCA(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*BCA(gel,i,j)-BCA(gel,i,j)*Vy-Vc*Cy) + 0.5*h*Cx;
             SL(gel,i,j) = BS(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*BS(gel,i,j)-BS(gel,i,j)*Vy-Vc*Sy)+0.5*h*Sx;
             C2L(gel,i,j) = BC2(gel,i,j)+0.5*dt*(-C2x*Uc-Ux*BC2(gel,i,j)-BC2(gel,i,j)*Vy-Vc*C2y)+0.5*h*C2x;
	     
             CAR(gel,i-1,j) = BCA(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*BCA(gel,i,j)-BCA(gel,i,j)*Vy-Vc*Cy) - 0.5*h*Cx;
             SR(gel,i-1,j) = BS(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*BS(gel,i,j)-BS(gel,i,j)*Vy-Vc*Sy)-0.5*h*Sx;
             C2R(gel,i-1,j) = BC2(gel,i,j)+0.5*dt*(-C2x*Uc-Ux*BC2(gel,i,j)-BC2(gel,i,j)*Vy-Vc*C2y)-0.5*h*C2x;

//the other direction
             Syc = (BS(gel,i,j+1)-BS(gel,i,j-1))/(2*h);
             Syl = 2*(BS(gel,i,j)-BS(gel,i,j-1))/h;
             Syu = 2*(BS(gel,i,j+1)-BS(gel,i,j))/h;

             C2yc = (BC2(gel,i,j+1)-BC2(gel,i,j-1))/(2*h);
             C2yl = 2*(BC2(gel,i,j)-BC2(gel,i,j-1))/h;
             C2yu = 2*(BC2(gel,i,j+1)-BC2(gel,i,j))/h;

             Cyc = (BCA(gel,i,j+1)-BCA(gel,i,j-1))/(2*h);
             Cyl = 2*(BCA(gel,i,j)-BCA(gel,i,j-1))/h;
             Cyu = 2*(BCA(gel,i,j+1)-BCA(gel,i,j))/h;

             Sy = minmod(Syc,Syl,Syu);
             C2y = minmod(C2yc,C2yl,C2yu);
             Cy = minmod(Cyc,Cyl,Cyu);
           
	     //upwinding for transverse derivative
	     
	     if(Uc > 0)
             {
                Cx = (BCA(gel,i,j)-BCA(gel,i-1,j))/h;
                Sx = (BS(gel,i,j)-BS(gel,i-1,j))/h;
                C2x = (BC2(gel,i,j)-BC2(gel,i-1,j))/h;
             }
             else
             {
                Cx = (BCA(gel,i+1,j)-BCA(gel,i,j))/h;
                Sx = (BS(gel,i+1,j)-BS(gel,i,j))/h;
                C2x = (BC2(gel,i+1,j)-BC2(gel,i,j))/h;
             }

             SB(gel,i,j) = BS(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*BS(gel,i,j)-BS(gel,i,j)*Vy-Vc*Sy)+0.5*h*Sy;
             ST(gel,i,j-1) = BS(gel,i,j)+0.5*dt*(-Sx*Uc-Ux*BS(gel,i,j)-BS(gel,i,j)*Vy-Vc*Sy)-0.5*h*Sy;

             C2B(gel,i,j) = BC2(gel,i,j)+0.5*dt*(-C2x*Uc-Ux*BC2(gel,i,j)-BC2(gel,i,j)*Vy-Vc*C2y)+0.5*h*C2y;
             C2T(gel,i,j-1) = BC2(gel,i,j)+0.5*dt*(-C2x*Uc-Ux*BC2(gel,i,j)-BC2(gel,i,j)*Vy-Vc*C2y)-0.5*h*C2y;

             CAB(gel,i,j) = BCA(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*BCA(gel,i,j)-BCA(gel,i,j)*Vy-Vc*Cy) + 0.5*h*Cy;
             CAT(gel,i,j-1) = BCA(gel,i,j)+0.5*dt*(-Cx*Uc-Ux*BCA(gel,i,j)-BCA(gel,i,j)*Vy-Vc*Cy) - 0.5*h*Cy;
	  }
       }

       //update_ghost_estate_sn(gel,gmax);
       update_ghost_estate_we(gel,gmax);
 
       for(i=lbuf[0]-1; i<gmax[0]-ubuf[0]; ++i)
       {
          for(j=lbuf[1]-1; j<gmax[1]-ubuf[1]; ++j)
          {
             if(gel->step == 0)
             {
               unh = Un(gel,i,j); vnh = Vn(gel,i,j);
             }
             else
             {
               unh = 0.5*(3*Un(gel,i,j)-OUn(gel,i,j));
               vnh = 0.5*(3*Vn(gel,i,j)-OVn(gel,i,j));
             }

             FBSx(gel,i,j) = (unh > 0) ? SL(gel,i,j) : SR(gel,i,j);
             FBSy(gel,i,j) = (vnh > 0) ? SB(gel,i,j) : ST(gel,i,j);

             FBSx(gel,i,j) *= unh;
             FBSy(gel,i,j) *= vnh;

             FBCAx(gel,i,j) = (unh > 0) ? CAL(gel,i,j) : CAR(gel,i,j);
             FBCAy(gel,i,j) = (vnh > 0) ? CAB(gel,i,j) : CAT(gel,i,j);

             FBCAx(gel,i,j) *= unh;
             FBCAy(gel,i,j) *= vnh;

             FBC2x(gel,i,j) = (unh > 0) ? C2L(gel,i,j) : C2R(gel,i,j);
             FBC2y(gel,i,j) = (vnh > 0) ? C2B(gel,i,j) : C2T(gel,i,j);

             FBC2x(gel,i,j) *= unh;
             FBC2y(gel,i,j) *= vnh;
	  }
       }
       return;
}

static void update_bdion_advection(
       GEL    *gel,
       int    *gmax)
{
       int     p,q,i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;
       double  h = gel->rgr.h;
       double  dt=gel->dt;
       State   *st;  

       update_edge_flux_bion(gel,gmax,dt,h);
       
       update_edge_flux_dion(gel,gmax,dt,h);

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
//save the old values
             OS(gel,i,j) = S(gel,i,j); OCL(gel,i,j) = CL(gel,i,j);
             OCA(gel,i,j) = CA(gel,i,j); OBS(gel,i,j) = BS(gel,i,j);
             OBCA(gel,i,j) = BCA(gel,i,j); OBC2(gel,i,j) = BC2(gel,i,j); 

             S(gel,i,j) = OThs(gel,i,j)*S(gel,i,j)-dt*(FSx(gel,i,j)-FSx(gel,i-1,j)+FSy(gel,i,j)-FSy(gel,i,j-1))/h;
             CL(gel,i,j) = OThs(gel,i,j)*CL(gel,i,j)-dt*(FCLx(gel,i,j)-FCLx(gel,i-1,j)+FCLy(gel,i,j)-FCLy(gel,i,j-1))/h;
             CA(gel,i,j) = OThs(gel,i,j)*CA(gel,i,j)-dt*(FCAx(gel,i,j)-FCAx(gel,i-1,j)+FCAy(gel,i,j)-FCAy(gel,i,j-1))/h;

             S(gel,i,j) /= Ths(gel,i,j);
             CL(gel,i,j) /= Ths(gel,i,j);
             CA(gel,i,j) /= Ths(gel,i,j);

             S(gel,i,j) = max(0.0, S(gel,i,j)); CL(gel,i,j) = max(0.0, CL(gel,i,j));
             CA(gel,i,j) = max(0.0, CA(gel,i,j));
 
             BS(gel,i,j) += -dt*(FBSx(gel,i,j)-FBSx(gel,i-1,j)+FBSy(gel,i,j)-FBSy(gel,i,j-1))/h;
             BCA(gel,i,j) += -dt*(FBCAx(gel,i,j)-FBCAx(gel,i-1,j)+FBCAy(gel,i,j)-FBCAy(gel,i,j-1))/h;
             BC2(gel,i,j) += -dt*(FBC2x(gel,i,j)-FBC2x(gel,i-1,j)+FBC2y(gel,i,j)-FBC2y(gel,i,j-1))/h;

             BS(gel,i,j) = max(0.0, BS(gel,i,j)); BCA(gel,i,j) = max(0.0, BCA(gel,i,j));
             BC2(gel,i,j) = max(0.0, BC2(gel,i,j));
          }
}

static void update_bdion_advection1(
       GEL    *gel,
       int    *gmax)
{
       int     p,q,i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;
       double  h = gel->rgr.h;
       double  dt=gel->dt;
       State   *st;  

       update_edge_flux_bion(gel,gmax,dt,h);
       
       update_edge_flux_dion1(gel,gmax,dt,h);

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
//save the old values
             OS(gel,i,j) = S(gel,i,j); OCL(gel,i,j) = CL(gel,i,j);
             OCA(gel,i,j) = CA(gel,i,j); OBS(gel,i,j) = BS(gel,i,j);
             OBCA(gel,i,j) = BCA(gel,i,j); OBC2(gel,i,j) = BC2(gel,i,j); 

             S(gel,i,j) = S(gel,i,j)-dt*(FSx(gel,i,j)-FSx(gel,i-1,j)+FSy(gel,i,j)-FSy(gel,i,j-1))/h;
             CL(gel,i,j) = CL(gel,i,j)-dt*(FCLx(gel,i,j)-FCLx(gel,i-1,j)+FCLy(gel,i,j)-FCLy(gel,i,j-1))/h;
             CA(gel,i,j) = CA(gel,i,j)-dt*(FCAx(gel,i,j)-FCAx(gel,i-1,j)+FCAy(gel,i,j)-FCAy(gel,i,j-1))/h;

             S(gel,i,j) = max(0.0, S(gel,i,j)); CL(gel,i,j) = max(0.0, CL(gel,i,j));
             CA(gel,i,j) = max(0.0, CA(gel,i,j));
 
             BS(gel,i,j) += -dt*(FBSx(gel,i,j)-FBSx(gel,i-1,j)+FBSy(gel,i,j)-FBSy(gel,i,j-1))/h;
             BCA(gel,i,j) += -dt*(FBCAx(gel,i,j)-FBCAx(gel,i-1,j)+FBCAy(gel,i,j)-FBCAy(gel,i,j-1))/h;
             BC2(gel,i,j) += -dt*(FBC2x(gel,i,j)-FBC2x(gel,i-1,j)+FBC2y(gel,i,j)-FBC2y(gel,i,j-1))/h;

             BS(gel,i,j) = max(0.0, BS(gel,i,j)); BCA(gel,i,j) = max(0.0, BCA(gel,i,j));
             BC2(gel,i,j) = max(0.0, BC2(gel,i,j));
          }
}

// which function are we using?
static void extrapolate_ion_concentration(
       GEL    *gel,
       int    *gmax)
{
       int     i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
             ES(gel,i,j)  = 2.0*S(gel,i,j)-OS(gel,i,j);  
             ECL(gel,i,j)  = 2.0*CL(gel,i,j)-OCL(gel,i,j); 
             ECA(gel,i,j) = 2.0*CA(gel,i,j)-OCA(gel,i,j);

             EBS(gel,i,j)  = 2.0*BS(gel,i,j)-OBS(gel,i,j);
             EBCA(gel,i,j)  = 2.0*BCA(gel,i,j)-OBCA(gel,i,j);
             EBC2(gel,i,j) = 2.0*BC2(gel,i,j)-OBC2(gel,i,j);
          }
}

static void extrapolate_ion_concentration1(
       GEL    *gel,
       int    *gmax)
{
       int     i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
             ES(gel,i,j)  = S(gel,i,j);  
             ECL(gel,i,j) = CL(gel,i,j); 
             ECA(gel,i,j) = CA(gel,i,j);

             EBS(gel,i,j)  = BS(gel,i,j);
             EBCA(gel,i,j) = BCA(gel,i,j);
             EBC2(gel,i,j) = BC2(gel,i,j);
          }
}

static void update_bion_reaction(
       GEL    *gel,
       int    *gmax)
{
       int     i,j;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;
       double  dt = gel->dt;
       double  kson = gel->kson;
       double  kcon = gel->kcon;
       double  ksoff = gel->ksoff;
       double  kcoff = gel->kcoff;

       double  **A,x[3],rhs[3];
       double  zt = gel->zt;

       A = malloc(3*sizeof(double*));

       for(i = 0; i < 3; i++)
          A[i] = malloc(3*sizeof(double));

//solve a 3-by-3 linear system from backward Euler

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {

		// coming from eqns 74,75 and 76 in Nov22_2019 notes (including the new corrections)
	    // Note: Solution vector is : [B_na; B_Ca; B_C2]
//bound sodium
             A[0][0] = 1.0 + dt*(kson*Ths(gel,i,j)*ES(gel,i,j)+ksoff*Ths(gel,i,j)*Ths(gel,i,j)); // coeff of B_Na
             A[0][1] = dt*kson*Ths(gel,i,j)*ES(gel,i,j); // coeff of B_Ca
			 A[0][2] = 2.0*dt*kson*Ths(gel,i,j)*ES(gel,i,j); // coeff of B_C2
             
             rhs[0] = BS(gel,i,j) + dt*kson*Ths(gel,i,j)*zt*Thn(gel,i,j)*ES(gel,i,j);

//bound calcium
             A[1][0] = dt*(kcon*Ths(gel,i,j)*ECA(gel,i,j) - 0.5*kcon*EBCA(gel,i,j)); // coeff of B_Na
             A[1][1] = 1.0 + dt*(kcon*Ths(gel,i,j)*ECA(gel,i,j)+kcoff*Ths(gel,i,j)*Ths(gel,i,j));
             A[1][1] -= 0.5*dt*kcon*EBCA(gel,i,j); // coeff of B_Ca
			 A[1][2] = 2.0*dt*kcon*Ths(gel,i,j)*ECA(gel,i,j);						// coeff of B_Ca
             A[1][2] -= dt*(kcon*EBCA(gel,i,j)+2.0*kcoff*Ths(gel,i,j)*Ths(gel,i,j)); // coeff of B_Ca
 
             rhs[1] = BCA(gel,i,j) +dt*kcon*Ths(gel,i,j)*zt*Thn(gel,i,j)*ECA(gel,i,j);
             rhs[1] -= 0.5*dt*kcon*zt*Thn(gel,i,j)*EBCA(gel,i,j);

//doubly bound calcium  
             A[2][0] = 0.5*dt*kcon*EBCA(gel,i,j);						// coeff of B_Na
             A[2][1] = 0.5*dt*kcon*EBCA(gel,i,j);						// coeff of B_Ca
             A[2][2] = 1.0 + dt*(kcon*EBCA(gel,i,j) + 2.0*kcoff*Ths(gel,i,j)*Ths(gel,i,j));   // coeff of B_C2;
             
             rhs[2] = BC2(gel,i,j) + 0.5*dt*kcon*zt*Thn(gel,i,j)*EBCA(gel,i,j);

             linear_system_solver(A,rhs,3,x);

             BS(gel,i,j) = max(0.0, x[0]); BCA(gel,i,j) = max(0.0, x[1]);
             BC2(gel,i,j) = max(0.0, x[2]); 
          }
         
          for(i=0; i<3; ++i)
            free(A[i]);

          free(A);
}

static void update_chemical_potential(
       GEL    *gel,
       int    *gmax)
{
       int     nbmin,iter,i,j,p,q;
       int     *lbuf = gel->rgr.lbuf;
       int     *ubuf = gel->rgr.ubuf;
       double  N = 6.0; // Number of monomers in a typical polymer chain of mucin
       double  dt = gel->dt;
       double  kbT = 42.821; 
       double  CW = 55.5;  // Standard molarity of water

       double  e1 = gel->e1;
       double  e2 = gel->e2; 
       double  zt = gel->zt; 
       double  e3 = gel->e3;
       double  a,I,sigI;

       double  zs = gel->zs;
       double  zcl = gel->zcl;
       double  zca = gel->zca;

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
          for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
          {
             a = 2.0*BC2(gel,i,j)/(zt*Thn(gel,i,j));
             I = 6.0*(e1+e2)-2.0*(1.0-1.0/N)*e1-e1*a; 

             Psinet(gel,i,j) = 1.0/N*log(Thn(gel,i,j)) + (1.0/N-1.0)*Ths(gel,i,j);             // note: e4 = 0
             Psinet(gel,i,j) += 0.5*I*Ths(gel,i,j)*Ths(gel,i,j) + 0.5*a*e3 - zt*Phi(gel,i,j);  // what about other terms in mu_n^0? e.g -3*e3
			 // constant terms can be removed as we are taking gradients
             Psinet(gel,i,j) *= kbT;
        
             sigI = (S(gel,i,j) + CL(gel,i,j) + CA(gel,i,j))/CW;
             Psisol(gel,i,j) = log(Ths(gel,i,j))+(1.0-1.0/N)*Thn(gel,i,j)-sigI;
             Psisol(gel,i,j) += 0.5*I*Thn(gel,i,j)*Thn(gel,i,j);
             Psisol(gel,i,j) *= kbT; 
             
             Psis(gel,i,j) = log((gel->epi+S(gel,i,j))/CW)+1.0-2.0*sigI+zs*Phi(gel,i,j); // what is epi? to make sure you dont have 0 inside
             Psis(gel,i,j) *= kbT;
 
             Psicl(gel,i,j) = log((gel->epi+CL(gel,i,j))/CW)+1.0-2.0*sigI+zcl*Phi(gel,i,j);
             Psicl(gel,i,j) *= kbT;

             Psica(gel,i,j) = log((gel->epi+CA(gel,i,j))/CW)+1.0-2.0*sigI+zca*Phi(gel,i,j);
             Psica(gel,i,j) *= kbT;
         }

         for(i=0; i<gmax[1]; ++i)
         {
	     Psinet(gel,lbuf[0]-1,i) = Psinet(gel,lbuf[0],i);  
             Psisol(gel,lbuf[0]-1,i) = Psisol(gel,lbuf[0],i);
             
             Psis(gel,lbuf[0]-1,i) = Psis(gel,lbuf[0],i);
             Psicl(gel,lbuf[0]-1,i) = Psicl(gel,lbuf[0],i);
             Psica(gel,lbuf[0]-1,i) = Psica(gel,lbuf[0],i);
 
             Psinet(gel,gmax[0]-ubuf[0],i) = Psinet(gel,gmax[0]-ubuf[0]-1,i);
             Psisol(gel,gmax[0]-ubuf[0],i) = Psisol(gel,gmax[0]-ubuf[0]-1,i);

             Psis(gel,gmax[0]-ubuf[0],i) = Psis(gel,gmax[0]-ubuf[0]-1,i);
             Psicl(gel,gmax[0]-ubuf[0],i) = Psicl(gel,gmax[0]-ubuf[0]-1,i);
             Psica(gel,gmax[0]-ubuf[0],i) = Psica(gel,gmax[0]-ubuf[0]-1,i);
         }

//top and bottom
         for(i=0; i<gmax[0]; ++i)
         {   
             Psinet(gel,i,lbuf[1]-1) = Psinet(gel,i,lbuf[1]);
             Psisol(gel,i,lbuf[1]-1) = Psisol(gel,i,lbuf[1]);
             
             Psis(gel,i,lbuf[1]-1)  = Psis(gel,i,lbuf[1]);
             Psicl(gel,i,lbuf[1]-1) = Psicl(gel,i,lbuf[1]);
             Psica(gel,i,lbuf[1]-1) = Psica(gel,i,lbuf[1]);
 
             Psinet(gel,i,gmax[1]-ubuf[1]) = Psinet(gel,i,gmax[1]-ubuf[1]-1);
             Psisol(gel,i,gmax[1]-ubuf[1]) = Psisol(gel,i,gmax[1]-ubuf[1]-1);
         
             Psis(gel,i,gmax[1]-ubuf[1])  = Psis(gel,i,gmax[1]-ubuf[1]-1);
             Psicl(gel,i,gmax[1]-ubuf[1]) = Psicl(gel,i,gmax[1]-ubuf[1]-1);
             Psica(gel,i,gmax[1]-ubuf[1]) = Psica(gel,i,gmax[1]-ubuf[1]-1);    
         }
}


//set up multigrid solver/preconditioner for L 
extern void MG1_solver(
       GEL    *gel,
       int    *gmax)
{
       int             i,j,tlevel;
       int             *lbuf=gel->rgr.lbuf;
       int             *ubuf=gel->rgr.ubuf;

       update_ghost_cell_states_we(gel,gmax);
       update_ghost_cell_states_sn(gel,gmax); //with periodic condition
       update_avg_theta(gel,gmax);
       
       //total level of refinement
       tlevel = floor(min(log2f(gmax[0]-lbuf[0]-ubuf[0]),log2f(gmax[1]-lbuf[1]-ubuf[1])));

       F1_cycle(gel,gmax,tlevel);
}


static void F1_cycle(
       GEL     *gel,
       int     *gmax,
       int     level)
{
       GEL          geln; //gel structure at next coarsest grid

       int          *lbuf = gel->rgr.lbuf;
       int          *ubuf = gel->rgr.ubuf;
       int          i;

       double       start,finish,residual;

       if(level == gel->c1level) //four grid center in any direction
       {
	 for (i=0; i<5; ++i)
         {
            rbgs_relaxation(gel,gmax);
	    update_ghost_cell_states_we(gel,gmax);
	    update_ghost_cell_states_sn(gel,gmax);
	 }
         return;
       }
       else
       {
	  for(i=0; i<gel->prev; ++i) //pre-smoothing
	  {
             rbgs_relaxation(gel,gmax);
	     update_ghost_cell_states_we(gel,gmax);
             update_ghost_cell_states_sn(gel,gmax);
          }
	  residual_evaluation1(gel,gmax);
	  init_next_gel(gel,&geln);
	  next_coarse_grid1(gel,&geln); //restriction of b-Ax' to next coarse grid
          update_ghost_cell_states_we(&geln,geln.rgr.gmax); //set volume fraction in buffer
	  update_ghost_cell_states_sn(&geln,geln.rgr.gmax);

          update_avg_theta(&geln,geln.rgr.gmax);
          
          F1_cycle(&geln,geln.rgr.gmax,floor(min(log2f(geln.rgr.gmax[0]-lbuf[0]-ubuf[0]),
                log2f(geln.rgr.gmax[1]-lbuf[1]-ubuf[1])))); 
	  
	  //recursive call F_cycle with zero initial guess
          next_fine_grid1(&geln,gel); //interpolate correction at next fine grid,correct solution
	  update_ghost_cell_states_we(gel,gmax);
          update_ghost_cell_states_sn(gel,gmax);
	  
	  for(i=0; i<gel->post; ++i) //post-smoothing
	  {
             rbgs_relaxation(gel,gmax);  //updated relaxation,problem occurs here
	     update_ghost_cell_states_we(gel,gmax);
             update_ghost_cell_states_sn(gel,gmax);
          }
          free(geln.st);
	  V1_cycle(gel,gmax,floor(min(log2f(gmax[0]-lbuf[0]-ubuf[0]),log2f(gmax[1]-lbuf[1]-ubuf[1]))));
       }
}

static void V1_cycle(
       GEL     *gel,
       int     *gmax,
       int     level)
{
       GEL          geln; //gel structure at next coarsest grid
       int          *lbuf = gel->rgr.lbuf;
       int          *ubuf = gel->rgr.ubuf;
       int          i;

       if(level == gel->c1level) 
       {
          for (i=0; i<5; ++i)
          {
            rbgs_relaxation(gel,gmax); 
	    update_ghost_cell_states_we(gel,gmax);
	    update_ghost_cell_states_sn(gel,gmax);
	  }
          return;
       }
       else
       {
	  for(i=0; i<gel->prev; ++i)
          {
	     rbgs_relaxation(gel,gmax);  //pre-smoothing
	     update_ghost_cell_states_we(gel,gmax);
             update_ghost_cell_states_sn(gel,gmax);
          }
          residual_evaluation1(gel,gmax);

	  init_next_gel(gel,&geln);
          next_coarse_grid1(gel,&geln); //restriction of b-Ax' to next coarse grid
	  update_ghost_cell_states_we(&geln,geln.rgr.gmax);
          update_ghost_cell_states_sn(&geln,geln.rgr.gmax);

	  update_avg_theta(&geln,geln.rgr.gmax);
 
          V1_cycle(&geln,geln.rgr.gmax,floor(min(log2f(geln.rgr.gmax[0]-lbuf[0]-ubuf[0]),
                   log2f(geln.rgr.gmax[1]-lbuf[1]-ubuf[1])))); 
	  //recursive call V_cycle with zero initial guess

          next_fine_grid1(&geln,gel); //interpolate correction at next fine grid,correct solution
          update_ghost_cell_states_we(gel,gmax);
          update_ghost_cell_states_sn(gel,gmax);
 
	  free(geln.st);
	  for(i=0; i<gel->post; ++i)
          {   
	     rbgs_relaxation(gel,gmax);  //post-smoothing
             update_ghost_cell_states_we(gel,gmax);
             update_ghost_cell_states_sn(gel,gmax);
          }
       }
}

static void rbgs_relaxation(
       GEL     *gel,
       int     *gmax)
{
       int           t,i,j,p,q;
       int           indx,m,n;
       int           dim = gel->rgr.dim;
       int           ig,jg;  //global block indx
       int           *pcrd=gel->pcrd;
       int           *pgmax=gel->pgmax;
       double        dt = gel->dt;
       double        kson = gel->kson;
       double        kcon = gel->kcon;

       double        h=gel->rgr.h;
       double        omega = gel->omega;
       double        hh,scale,div;
       double        Dc = gel->ds; 
       RECT_GRID     *rgr = &(gel->rgr);
       int           *lbuf=rgr->lbuf;
       int           *ubuf=rgr->ubuf;

       State         *st;

       hh = h*h; 
     
       for(indx=0; indx<2; ++indx)
       {
             for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
             {
	       for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
	       {
		 if((indx+(i-lbuf[0])+(j-lbuf[1]))%2 != 0)
		    continue;

                 st=&(Rect_state(i,j,gel)); scale = Dc*dt/(hh*Ths(gel,i,j));

                 S(gel,i,j) = BCC(gel,i,j) + scale*(st->ths[0][1]*S(gel,i-1,j) + st->ths[2][1]*S(gel,i+1,j)
                              + st->ths[1][0]*S(gel,i,j-1) + st->ths[1][2]*S(gel,i,j+1));
                 div = 1.0 + scale*(st->ths[0][1]+st->ths[2][1]+st->ths[1][0]+st->ths[1][2]);
 
                 if(gel->opt == 1) //for sodium
                    div +=  dt*kson*ABS(gel,i,j);
                 else if(gel->opt == 2) //for calcium
                    div +=  dt*kcon*ABS(gel,i,j);

                 S(gel,i,j) /= div;      
               }
             }
       }
}
static void residual_evaluation1(
       GEL     *gel,
       int     *gmax)
{
       int        i,j;
       State      *st;
       int        *lbuf=gel->rgr.lbuf;
       int        *ubuf=gel->rgr.ubuf;
       double     dt = gel->dt;
       double     h=gel->rgr.h;
       double     kson = gel->kson;
       double     kcon = gel->kcon;
       double     sub,scale;
       double     Dc = gel->ds;
       double     hh = h*h;

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
	   for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
	   {
	      st=&(Rect_state(i,j,gel)); scale = Dc*dt/(hh*Ths(gel,i,j));

              Rc(gel,i,j) = BCC(gel,i,j)+scale*(st->ths[0][1]*S(gel,i-1,j)+st->ths[2][1]*S(gel,i+1,j)
                            +st->ths[1][0]*S(gel,i,j-1) + st->ths[1][2]*S(gel,i,j+1));  
              Rc(gel,i,j) -= (1.0 + scale*(st->ths[0][1]+st->ths[2][1]+st->ths[1][0]+st->ths[1][2]))*S(gel,i,j);

              if(gel->opt == 1) //sodium
                 Rc(gel,i,j) -= dt*kson*ABS(gel,i,j)*S(gel,i,j);
              else if(gel->opt == 2) //calcium
                 Rc(gel,i,j) -= dt*kcon*ABS(gel,i,j)*S(gel,i,j); 
           }
       }
}

static void next_coarse_grid1(
       GEL     *gel,
       GEL     *geln)
{
       int          i,j,p,q;
       int          *gmax=geln->rgr.gmax;
       RECT_GRID    *rgr=&(geln->rgr);
       int          *lbuf=rgr->lbuf;
       int          *ubuf=rgr->ubuf;

       for(i=0; i<gmax[0]; ++i)
       {
          for(j=0; j<gmax[1]; ++j)
	  {
	     if(i >= lbuf[0] && j >= lbuf[1] && i < gmax[0]-ubuf[0] && j < gmax[1]-ubuf[1])
	     {
	        p=2*(i-lbuf[0])+lbuf[0]; q=2*(j-lbuf[1])+lbuf[1]; //block indx of lower left corner of fine grid

		ABS(geln,i,j)=0.25*(ABS(gel,p,q)+ABS(gel,p+1,q)+ABS(gel,p,q+1)+ABS(gel,p+1,q+1));

                BCC(geln,i,j)=0.25*(Rc(gel,p,q)+Rc(gel,p+1,q)+Rc(gel,p,q+1)+Rc(gel,p+1,q+1));
	        Ths(geln,i,j)=0.25*(Ths(gel,p,q)+Ths(gel,p+1,q)+Ths(gel,p,q+1)+Ths(gel,p+1,q+1));
	        Thn(geln,i,j)=1-Ths(geln,i,j);
	     
                //printf("\n coarse grid[%d %d] has thn=%f, with THN=%f %f %f %f",i,j,Thn(geln,i,j),
                  // Thn(gel,p,q),Thn(gel,p+1,q),Thn(gel,p,q+1),Thn(gel,p+1,q+1));
             }
	     S(geln,i,j) = 0.0;
	  }
       }
}

//interpolate from coarse grid gel to fine grid geln and correct solution
static void next_fine_grid1(
       GEL     *gel,   
       GEL     *geln)
{
       int          i,j,p,q;
       int          dir[2]; 
       int          *gmax=geln->rgr.gmax;
       RECT_GRID    *rgr=&(gel->rgr);
       int          *lbuf=rgr->lbuf;
       int          *ubuf=rgr->ubuf;
       int          *pcrd=gel->pcrd;

       for(i=lbuf[0]; i<gmax[0]-ubuf[0]; ++i)
       {
          p=floor((i-lbuf[0])/2)+lbuf[0];
	  dir[0]= (((i-lbuf[0])%2)==0 ? -1 :1);

	  for(j=lbuf[1]; j<gmax[1]-ubuf[1]; ++j)
	  {
	      q=floor((j-lbuf[1])/2)+lbuf[1]; 
	      dir[1]= (((j-lbuf[1])%2)==0 ? -1 :1);  //displacement within the coarse cell
	      
	      S(geln,i,j)+=0.0625*(9.0*S(gel,p,q)+3.0*S(gel,p+dir[0],q)+3.0*S(gel,p,q+dir[1])
	                     +S(gel,p+dir[0],q+dir[1]));
          }
       }
}

PetscErrorCode SampleShellPCApply1(PC pc,Vec x,Vec y)
{
       AppCtx          *user;

       PCShellGetContext(pc,(void**)&user);

       int     *gmax = user->gel->rgr.gmax;
       int     irow[1],gmax1[2];
       int     *lbuf=user->gel->rgr.lbuf;
       int     *ubuf=user->gel->rgr.ubuf;
       int     i,j,ip,jp;

       double               value[1];
       PetscScalar          *values;

//zero as initial guess
       for(i=0; i<gmax[0]; ++i)
       {
           for(j=0; j<gmax[1]; ++j)
	   {
	      S(user->gel,i,j)=0.0;
	   }
       }
       
       gmax1[0] = gmax[0]-lbuf[0]-ubuf[0];
       gmax1[1] = gmax[1]-lbuf[1]-ubuf[1];

       VecGetArray(x,&values);
      
//use input x as the right hand side
 
       for(i=0; i<gmax1[0]*gmax1[1]; ++i)
       {
          jp=floor(i/gmax1[0]) + lbuf[1]; 
          ip=i%gmax1[0] + lbuf[0]; 
    
          BCC(user->gel,ip,jp)=values[i];
       }

       VecRestoreArray(x,&values);

       MG1_solver(user->gel,gmax);
       
       for(i=0; i<gmax1[0]*gmax1[1]; ++i)
       {
          irow[0]=i;
	  
          jp=floor(i/gmax1[0]) + lbuf[1]; 
          ip=i%gmax1[0] + lbuf[0]; 
 
          value[0] = S(user->gel,ip,jp);
          VecSetValues(y,1,irow,value,INSERT_VALUES);
       }

       VecAssemblyBegin(y); VecAssemblyEnd(y);
       return 0;
}

