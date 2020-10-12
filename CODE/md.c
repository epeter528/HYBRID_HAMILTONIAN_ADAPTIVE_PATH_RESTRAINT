/* -*- mode: c; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; c-file-style: "stroustrup"; -*-
 *
 * 
 *                This source code is part of
 * 
 *                 G   R   O   M   A   C   S
 * 
 *          GROningen MAchine for Chemical Simulations
 * 
 *                        VERSION 3.2.0
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 * 
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 * 
 * For more info, check our website at http://www.gromacs.org
 * 
 * And Hey:
 * Gallium Rubidium Oxygen Manganese Argon Carbon Silicon
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#if ((defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64) && !defined __CYGWIN__ && !defined __CYGWIN32__)
/* _isnan() */
#include <float.h>
#endif

#include "typedefs.h"
#include "smalloc.h"
#include "sysstuff.h"
#include "vec.h"
#include "statutil.h"
#include "vcm.h"
#include "mdebin.h"
#include "nrnb.h"
#include "calcmu.h"
#include "index.h"
#include "vsite.h"
#include "update.h"
#include "ns.h"
#include "trnio.h"
#include "xtcio.h"
#include "mdrun.h"
#include "confio.h"
#include "network.h"
#include "pull.h"
#include "xvgr.h"
#include "physics.h"
#include "names.h"
#include "xmdrun.h"
#include "ionize.h"
#include "disre.h"
#include "orires.h"
#include "dihre.h"
#include "pppm.h"
#include "pme.h"
#include "mdatoms.h"
#include "repl_ex.h"
#include "qmmm.h"
#include "mpelogging.h"
#include "domdec.h"
#include "partdec.h"
#include "topsort.h"
#include "coulomb.h"
#include "constr.h"
#include "shellfc.h"
#include "compute_io.h"
#include "mvdata.h"
#include "checkpoint.h"
#include "mtop_util.h"
#include "sighandler.h"
#include "string2.h"

#ifdef GMX_LIB_MPI
#include <mpi.h>
#endif
#ifdef GMX_THREADS
#include "tmpi.h"
#endif

#ifdef GMX_FAHCORE
#include "corewrap.h"
#endif
#include <sys/stat.h>

#include "gmx_lapack.h"
#include "gmx_arpack.h"
#include "sparsematrix.h"
#include "eigensolver.h"

double do_md(FILE *fplog,t_commrec *cr,int nfile,const t_filenm fnm[],
             const output_env_t oenv, gmx_bool bVerbose,gmx_bool bCompact,
             int nstglobalcomm,
             gmx_vsite_t *vsite,gmx_constr_t constr,
             int stepout,t_inputrec *ir,
             gmx_mtop_t *top_global,
             t_fcdata *fcd,
             t_state *state_global,
             t_mdatoms *mdatoms,
             t_nrnb *nrnb,gmx_wallcycle_t wcycle,
             gmx_edsam_t ed,t_forcerec *fr,
             int repl_ex_nst,int repl_ex_seed,
             real cpt_period,real max_hours,
             const char *deviceOptions,
             unsigned long Flags,
             gmx_runtime_t *runtime)
{
    gmx_mdoutf_t *outf;
    gmx_large_int_t step,step_rel;
    double     run_time;
    double     t,t0,lam0;
    gmx_bool       bGStatEveryStep,bGStat,bNstEner,bCalcEnerPres;
    gmx_bool       bNS,bNStList,bSimAnn,bStopCM,bRerunMD,bNotLastFrame=FALSE,
               bFirstStep,bStateFromTPX,bInitStep,bLastStep,
               bBornRadii,bStartingFromCpt;
    gmx_bool       bDoDHDL=FALSE;
    gmx_bool       do_ene,do_log,do_verbose,bRerunWarnNoV=TRUE,
               bForceUpdate=FALSE,bCPT;
    int        mdof_flags;
    gmx_bool       bMasterState;
    int        force_flags,cglo_flags;
    tensor     force_vir,shake_vir,total_vir,tmp_vir,pres;
    int        i,m;
    t_trxstatus *status;
    rvec       mu_tot;
    t_vcm      *vcm;
    t_state    *bufstate=NULL;   
    matrix     *scale_tot,pcoupl_mu,M,ebox;
    gmx_nlheur_t nlh;
    t_trxframe rerun_fr;
    gmx_repl_ex_t repl_ex=NULL;
    int        nchkpt=1;

    gmx_localtop_t *top;	
    t_mdebin *mdebin=NULL;
    t_state    *state=NULL;
    rvec       *f_global=NULL;
    int        n_xtc=-1;
    rvec       *x_xtc=NULL;
    gmx_enerdata_t *enerd;
    rvec       *f=NULL;
    gmx_global_stat_t gstat;
    gmx_update_t upd=NULL;
    t_graph    *graph=NULL;
    globsig_t   gs;

    gmx_bool        bFFscan;
    gmx_groups_t *groups;
    gmx_ekindata_t *ekind, *ekind_save;
    gmx_shellfc_t shellfc;
    int         count,nconverged=0;
    real        timestep=0;
    double      tcount=0;
    gmx_bool        bIonize=FALSE;
    gmx_bool        bTCR=FALSE,bConverged=TRUE,bOK,bSumEkinhOld,bExchanged;
    gmx_bool        bAppend;
    gmx_bool        bResetCountersHalfMaxH=FALSE;
    gmx_bool        bVV,bIterations,bFirstIterate,bTemp,bPres,bTrotter;
    real        temp0,mu_aver=0,dvdl;
    int         a0,a1,gnx=0,ii;
    atom_id     *grpindex=NULL;
    char        *grpname;
    t_coupl_rec *tcr=NULL;
    rvec        *xcopy=NULL,*vcopy=NULL,*cbuf=NULL;
    matrix      boxcopy={{0}},lastbox;
	tensor      tmpvir;
	real        fom,oldfom,veta_save,pcurr,scalevir,tracevir;
	real        vetanew = 0;
    double      cycles;
	real        saved_conserved_quantity = 0;
    real        last_ekin = 0;
	int         iter_i;
	t_extmass   MassQ;
// new vars
    double force_level;
    double max_prob;
    double min_prob;
    double f_diff_x,f_diff_y,f_diff_z;
    double *ener_g;
    double diff_x,diff_y,diff_z;
    double d_tot;
    double kmc_time = 0.0;
    double maxf,maxf0,gamma,maxf1=0.0;
   int dstart,dend;
   int *count_0;
   double box_length;
   int stepper=0;    
   int k;
   double *norm,norm1,norm2,r_g1;	
    int         **trotter_seq; 
    char        sbuf[STEPSTRSIZE],sbuf2[STEPSTRSIZE];
    int         handled_stop_condition=gmx_stop_cond_none; /* compare to get_stop_condition*/
    gmx_iterate_t iterate;
    gmx_large_int_t multisim_nsteps=-1; /* number of steps to do  before first multisim 
                                          simulation stops. If equal to zero, don't
                                          communicate any more between multisims.*/
#ifdef GMX_FAHCORE
    /* Temporary addition for FAHCORE checkpointing */
    int chkpt_ret;
#endif


    double WA,WB;
    double *sigma_xx,*sigma_yx,*sigma_zx;
    double **sigma_strich_zx,**sigma_strich_xx,**sigma_strich_yx;
    double **grad_meta_x,**grad_meta_y,**grad_meta_z;
    double *df_x,**df_x_s,*df_y,**df_y_s,*df_z,**df_z_s;
    double df_x_tot,df_y_tot,df_z_tot;
    double PI,min_fluct_par;
    double wt_meta_delta_t,s_factor=0.0,accept=0.0;
    double df_xx,df_yx,df_zx;

    double *f_x1,*f_y1,*f_z1;
    double *f_x2,*f_y2,*f_z2; 

    double *f_x1_s,*f_y1_s,*f_z1_s;
    double *f_x2_s,*f_y2_s,*f_z2_s;

    double *delta_fx1,*delta_fy1,*delta_fz1;
    double *delta_fx2,*delta_fy2,*delta_fz2;
    double *delta_fx3,*delta_fy3,*delta_fz3;
    double **delta_fx4,**delta_fy4,**delta_fz4;
    double **delta_fx5,**delta_fy5,**delta_fz5;    
    double *delta_fx,*delta_fy,*delta_fz;      
    double *df_x1,*df_y1,*df_z1;
    double *df_x1b,*df_y1b,*df_z1b;    
    double **d_x1,**d_x1_s,**d_y1;
    double **d_y1_s,**d_z1,**d_z1_s;
    double *x0,*y0,*z0;
    double *x1,*y1,*z1,*r_g; 
    double *x0_s,*y0_s,*z0_s;
    double *x1_s,*y1_s,*z1_s; 
    double *grad_x,*grad_y,*grad_z;   
    double *grad_x0,*grad_y0,*grad_z0; 
    double *grad_x1,*grad_y1,*grad_z1;    
    double diff_t=1.0,*diff_t2,beta,**delta_t2,d_t,*d_t_a;
    double *d_t_a0;
    double d_range,**diff_t_last,*h1,*h2,*h3,h4,h5;
    double pot_x=0.0,pot_y=0.0,pot_z=0.0;
    int    *count2,counter_test=0,r;
    double beta2=0.0,beta_sum=0.0,angle_phi,width_WA=0.0;
    float ran,ran2;
    int   **tau,**tau2;
    int   n,number_of_replicas,p,natoms1=0,*at_id;
    
    double d_max=0.0;
    double d_tot2,phi[101],*delta_s,replica_fraction=0.0;
    int    replica_fraction_integer=0,*step_father,nr;   
    double beta3,stepsize_alpha=0.05;
    double delta_alpha1,delta_alpha2;
    double delta_alpha1b,delta_alpha2b;
    double **delta_L_x,**delta_L_y,**delta_L_z;
    double **delta_L_x2,**delta_L_y2,**delta_L_z2;
    double **alpha_1_x,**alpha_1_y,**alpha_1_z;
    double **alpha_2_x,**alpha_2_y,**alpha_2_z;
    double *alpha_3_x,*alpha_3_y,*alpha_3_z;
    double alpha_bar_x=0.0,alpha_bar_y=0.0,alpha_bar_z=0.0;
    double **delta_alpha_1x,**delta_alpha_1y,**delta_alpha_1z;
    double **delta_alpha_2x,**delta_alpha_2y,**delta_alpha_2z; 
    double *beta_sum_x,*beta_sum_y,*beta_sum_z,d_x,d_y,d_z,d_t2;
    double range_alpha= 0.0;
    int tau2_a,tau_c,tau_a,**delta_tau,**delta_tau_2,**tau_1,tau_bar=0.0;
    int count_num_resolve=0;
   
    double ***fee_en,radius=0.0; 
    double *alpha_strich_1_x,*alpha_strich_1_y,*alpha_strich_1_z;
    double *alpha_strich_2_x,*alpha_strich_2_y,*alpha_strich_2_z; 
    double alpha_strich_bar_x=0.0,alpha_strich_bar_y=0.0,alpha_strich_bar_z=0.0;
    double *delta_rdf_x2,*delta_rdf_x,*delta_rdf_y2,*delta_rdf_y,*delta_rdf_z,*delta_rdf_z2;
    double *delta_alpha_strich_1_x,*delta_alpha_strich_1_y,*delta_alpha_strich_1_z;
    double *delta_alpha_strich_2_x,*delta_alpha_strich_2_y,*delta_alpha_strich_2_z; 
    double **sigma_strich_rdf;    
    int   ch,natoms2=0;
    int    *at_id_b,number_of_lines=0;
    double *a_x_b,*a_y_b,*a_z_b;
    double *a_x_b2,*a_y_b2,*a_z_b2;
    double *alpha_strich;
    double range_dist=0.0,dx=0.0,dy=0.0,dz=0.0;
    double **epsilon2,**epsilon3,*epsilon4,*epsilon5,**epsilon7,**epsilon6,epsilon=0.0,epsilon_md=0.0;   
    double cut_off_range=0.0; 

    double *sigma_zero_x,*sigma_zero_y,*sigma_zero_z;
    double *sigma_one_x,*sigma_one_y,*sigma_one_z;
    double *sum_sigma_one_x,*sum_sigma_one_y,*sum_sigma_one_z;
    double *sum_sigma_zero_x,*sum_sigma_zero_y,*sum_sigma_zero_z;
    double sum_d_sigma_x=0.0,sum_d_sigma_y=0.0,sum_d_sigma_z=0.0;
    double d_sigma_x=0.0,d_sigma_y=0.0,d_sigma_z=0.0;
    double diff_sigma_x=0.0,diff_sigma_y=0.0,diff_sigma_z=0.0;
    double sigma_constant=0.0;
    double sign=1.0;
    double av_a_x_b=0.0,av_a_y_b=0.0,av_a_z_b=0.0;
    double *counter_t_a;

    double sum_exp = 0.0;
    double *exp_of_funct;
    double sum_of_all_derivatives_x = 0.0;
    double sum_of_all_derivatives_y = 0.0;
    double sum_of_all_derivatives_z = 0.0;      
    double sum_of_all_derivatives_x3;
    double sum_of_all_derivatives_y3,sum_of_all_derivatives_z3;
    double sum_of_all_derivatives_x5,sum_of_all_derivatives_y5;
    double sum_of_all_derivatives_z5;
    double exp_of_funct2;      
    double *x_1; 
    double *y_1; 
    double *z_1;
    double *charge_at_site; 
    double probability_input=1.0;
    double rho_sys=0.0;
    double **dL_abs,sum_dL=0.0;
    double **rho,**sum_rho,**prob_rho;
    double counter_exchange = 0.0;
    double rf_factor=0.0;  
    int num_grid=0;
    int density_boolean=0;
    double friction_constant=0.0;
    double *path_x,*path_y,*path_z;
    double *path_x2,*path_y2,*path_z2;
    double *path_x3,*path_y3,*path_z3;    
    int maximal_search=0;
    double probability_input2=0.0;
    int *restraint_sampling;
    int *at_id_r,*at_id_r2;
    double *distance_restraint;
    int number_of_restraints = 0;
    double **velocity;
    double *d_x2,*d_y2,*d_z2;
    double *d_x3,*d_y3,*d_z3;
    
    double *delta_path_x,*delta_path_y,*delta_path_z;
    double *sum_f_x,*sum_f_y,*sum_f_z;
    double *sum_x,*sum_y,*sum_z;
    double *sum_delta_path_x,*sum_delta_path_y,*sum_delta_path_z;  
    double *sum_of_path_x,*sum_of_path_y,*sum_of_path_z;
    
    double *av_grad_x,*av_grad_y,*av_grad_z;
    double cos_val;
    int    timeframe_sol = 3626;
    int    stepper22=0,step_sol=0;
    double *df_x_imp,*df_y_imp,*df_z_imp;
    double *vec_dL_x,*vec_dL_y,*vec_dL_z;
    double *vec_dL_x2,*vec_dL_y2,*vec_dL_z2; 
    double *vec_dL_x3,*vec_dL_y3,*vec_dL_z3;     
    double d_t4=0.0,epsilon_sol=0.0,epsilon_sol1=0.0;
    double d_t0;
    
    double **dih_en;
    int    number_of_dihedrals=0;
    int    *at_id_d1,*at_id_d2,*at_id_d3,*at_id_d4;
    int    dih_rdf_sampling_on = 0;
    
    double *tmp_x,*tmp_y,*tmp_z;
    double *mat;
    real   *eigval;
    int    info=0,flag_diagonalize;
    int selector=0,l;  
    double var=0.0;     
    int  applied_imp=0;
    double delta1=0.0;
    
       // printf("%s\n","here-1");    
    struct stat *buf = malloc(sizeof(struct stat));
   
    PI = 3.14159265359;

    FILE *fp = fopen("minput","r");

    fscanf(fp,"%d\n",&dstart);
    fscanf(fp,"%d\n",&dend);
    fscanf(fp,"%lg\n",&WA);  
    fscanf(fp,"%d\n",&tau_a);
    fscanf(fp,"%d\n",&tau2_a); // if tau2_a == 0, replica based path-sampling switched off
    fscanf(fp,"%lg\n",&diff_t);
    fscanf(fp,"%d\n",&number_of_replicas);
    if(fscanf(fp,"%lg\n",&beta2)==1) printf("%s\t%lg\n","beta on ",beta2);
    if(fscanf(fp,"%lg\n",&beta3)==1) printf("%s\t%lg\n","beta3 ",beta3);
    fscanf(fp,"%lg\n",&stepsize_alpha);
    fscanf(fp,"%lg\n",&range_alpha);    
    if(fscanf(fp,"%lg",&wt_meta_delta_t)==1) 
    printf("%s\t%lg\n","wt-meta on",wt_meta_delta_t);
    if(fscanf(fp,"%d\n",&p)==1) printf("%s\t%d\n","p set",p); 
    fscanf(fp,"%lg\n",&epsilon);
    fscanf(fp,"%lg\n",&epsilon_md); 
    fscanf(fp,"%d\n",&tau_c); // if tau_c == 0, the search for rdf-based potentials of mean force is switched off
    fscanf(fp,"%lg\n",&probability_input);
    if(fscanf(fp,"%lg\n",&rf_factor)==1) printf("%lg\t%s\n",rf_factor," rf_factor");
    if(fscanf(fp,"%d\n",&num_grid)==1) printf("%d\t%s\n",num_grid," num_grid "); 
    if(fscanf(fp,"%lg\n",&epsilon_sol)==1) printf("%lg\t%s\n",epsilon_sol," epsilon_sol ");   
    if(fscanf(fp,"%d\n",&timeframe_sol)==1) printf("%s\t%d\n"," timeframe_sol ", timeframe_sol);
    if(fscanf(fp,"%d\n",&density_boolean)==1) {
       
     if(density_boolean == 1) { 
        printf("%s\n","density-sampling on - replica sampling switched off");
        };
    };     
    
    if(p > 0) natoms1 = (int)roundf((float)(dend-dstart+1)/(float)p);
    if(p == 0) natoms1 = (dend-dstart+1);   

    fclose(fp);
  
    snew(at_id,natoms1+1);     
       
   probability_input2 = probability_input;
 
   fp = fopen("atom_index.dat","r");    
   
   if(fp != NULL && tau_c == 0) {
   
   k = 0;
    
   while(!feof(fp))
   {
   ch = fgetc(fp);
    if(ch == '\n')
     {
       k++;
     };
   };    
   
   fclose(fp);
   
   natoms1 = k;
    
   
   k = 1;
   
   char char1[101];
   
    fp = fopen("atom_index.dat","r");
    
    while(fgets(char1,100,fp)!=0) {
        
        sscanf(char1,"%d",&at_id[k]);
        printf("%s\t%d\n","atom_id-dat ",at_id[k]);
        
        at_id[k] -= 1;
        
        k++;
        
    };

    fclose(fp);
    
   } else {    
    
    for(i=1;i<=natoms1;i++) 

    {

      if(p > 0.0) {
	      
      at_id[i] = ((int)roundf((float)(i-1)*(float)p)) - 1 + dstart;

      };

      if(p == 0.0) at_id[i] = i - 2 + dstart;

      printf("%d\t%s\n",at_id[i]," at-id ");

    };
   };    
    
    snew(restraint_sampling,dend-dstart+1);
    
    for(i=1;i<=natoms1;i++) {
        
        restraint_sampling[i] = 0.0;
        
    };
    
    snew(tmp_x,(natoms1+1)*3*(natoms1+1)*3);   
    snew(mat,(natoms1+1)*3*(natoms1+1)*3); 
    snew(eigval,(natoms1+1)*3);    
    
   if(stat("restraint_data.dat", buf) == 0) {
   
      fp = fopen("restraint_data.dat","r");
   
      char char1[100];
 
      k = 1;

      while(fgets(char1,100,fp)!=0) {

         k++;

      };

      fclose(fp);

      number_of_restraints = k-1;

      snew(at_id_r,number_of_restraints+1);
      snew(at_id_r2,number_of_restraints+1);
      snew(distance_restraint,number_of_restraints+1);

      fp = fopen("restraint_data.dat","r");
       
      int n1,n2;
      double d1,d2;
      
      k = 1; 
      
      while(fgets(char1,100,fp)!=0) {
        
        sscanf(char1,"%d%d%lg",&n1,&n2,&d1);
        printf("%s\t%d\t%d\t%lg\n","at_restr ",n1,n2,d1);        
        
                 at_id_r[k] = n1;
                 distance_restraint[k] = d1;         
                 at_id_r2[k] = n2;
                 restraint_sampling[n1] = 1;
                 restraint_sampling[n2] = 1;
                 
        number_of_restraints = k;
        k++;
        
      };
      
      fclose(fp);
      
   };
    
dih_rdf_sampling_on = 0;   
   
if(tau_c > 0 && stat("dihedral_rdf_index.dat", buf) == 0) {
    
            
      dih_rdf_sampling_on = 1;
       
      int n1,n2,n3,n4;
      char char1[101];
      double fee_en_value=0.0;
      char filechar[200],string[70]; 
      k = 1; 
      
      fp = fopen("dihedral_rdf_index.dat","r");      
      
      while(fgets(char1,100,fp)!=0) {
        
        sscanf(char1,"%d%d%d%d",&n1,&n2,&n3,&n4);      
        
                 k++;
                 
      }; 
      
      fclose(fp);
      
      number_of_dihedrals = k-1;      
  
      snew(at_id_d1,number_of_dihedrals+1);
      snew(at_id_d2,number_of_dihedrals+1);
      snew(at_id_d3,number_of_dihedrals+1);
      snew(at_id_d4,number_of_dihedrals+1);       
      
      k = 1;
      
      fp = fopen("dihedral_rdf_index.dat","r");      
      
      while(fgets(char1,100,fp)!=0) {
        
        sscanf(char1,"%d%d%d%d",&n1,&n2,&n3,&n4);
        printf("%s\t%d\t%d\t%d\t%d\n","dih-index ",n1,n2,n3,n4);        
        
                 at_id_d1[k] = n1;
                 at_id_d2[k] = n2;
                 at_id_d3[k] = n3;
                 at_id_d4[k] = n4;
        
                 k++;
                 
      };       
       
      dih_en = malloc(sizeof(double)*(number_of_dihedrals+1));
    
      for(i=1;i<=number_of_dihedrals;i++){
        
         dih_en[i] = malloc(sizeof(double)*(361)); 

      };       
      
      fclose(fp);
      
       for(k=1;k<=number_of_dihedrals;k++) {   
         
         sprintf(filechar,"%s%d%s%d%s%d%s%d%s","angpmf_",at_id_d1[k],"_",at_id_d2[k],"_",at_id_d3[k],"_",at_id_d4[k],".dat"); 
         
         printf("%s\n",filechar);         
         
         n = 1;
         
         fp = fopen(filechar,"r"); 
         
         while(fgets(string,70,fp)!=NULL){
             
                 sscanf(string,"%lg\t%lg\n",&radius,&fee_en_value);
                 dih_en[k][n] = fee_en_value;
                 n++;
                 
             };    
                         
         fclose(fp);         
         
         };
      
};
  
   
if(tau_c > 0) {    
    
   fp = fopen("atom_index.dat","r");    
   
   if(fp != NULL) {
   
   k = 0;
    
   while(!feof(fp))
   {
   ch = fgetc(fp);
    if(ch == '\n')
     {
       k++;
     };
   };    
   
   fclose(fp);
    
   snew(at_id_b,k+1);
   
   natoms2 = k;
   
   k = 1;
   
   char char1[101];
   
    fp = fopen("atom_index.dat","r");
    
    while(fgets(char1,100,fp)!=0) {
        
        sscanf(char1,"%d",&at_id_b[k]);
        
        at_id_b[k] -= 1;
        
        k++;
        
    };

    fclose(fp);
    
   };
};

if(tau_c > 0) {
    
    snew(a_x_b,natoms2+1);
    snew(a_y_b,natoms2+1);
    snew(a_z_b,natoms2+1);  
    snew(a_x_b2,natoms2+1);
    snew(a_y_b2,natoms2+1);
    snew(a_z_b2,natoms2+1);   
    snew(d_t_a,natoms2+1);
    snew(d_t_a0,natoms2+1);
    snew(counter_t_a,natoms2+1);
    snew(exp_of_funct,natoms2+1);
    
    fee_en = malloc(sizeof(double)*(natoms2+1));
    
    for(i=1;i<=natoms2;i++){
        
        fee_en[i] = malloc(sizeof(double)*(natoms2+1)); 

        
    };
    
    for(i=1;i<=natoms2;i++){

        for(k=1;k<=natoms2;k++){  
        
        fee_en[i][k] = malloc(sizeof(double)*(201)); 
        
         };
       };
       
    snew(sigma_zero_x,natoms2+1);
    snew(sigma_zero_y,natoms2+1);
    snew(sigma_zero_z,natoms2+1);
    snew(sigma_one_x,natoms2+1);
    snew(sigma_one_y,natoms2+1);
    snew(sigma_one_z,natoms2+1);
    snew(sum_sigma_zero_x,natoms2+1);
    snew(sum_sigma_zero_y,natoms2+1);
    snew(sum_sigma_zero_z,natoms2+1);
    snew(sum_sigma_one_x,natoms2+1);
    snew(sum_sigma_one_y,natoms2+1);
    snew(sum_sigma_one_z,natoms2+1);
    snew(epsilon5,natoms2+1);
    snew(epsilon4,natoms2+1); 
    snew(delta_rdf_x,natoms2+1);
    snew(delta_rdf_y,natoms2+1);
    snew(delta_rdf_z,natoms2+1);
    snew(delta_rdf_x2,natoms2+1);
    snew(delta_rdf_y2,natoms2+1);
    snew(delta_rdf_z2,natoms2+1);    
    snew(alpha_strich_1_x,natoms2+1);
    snew(alpha_strich_1_y,natoms2+1);
    snew(alpha_strich_1_z,natoms2+1);
    snew(alpha_strich_2_x,natoms2+1);
    snew(alpha_strich_2_y,natoms2+1);
    snew(alpha_strich_2_z,natoms2+1);
    snew(delta_alpha_strich_1_x,natoms2+1);
    snew(delta_alpha_strich_1_y,natoms2+1);
    snew(delta_alpha_strich_1_z,natoms2+1);
    snew(delta_alpha_strich_2_x,natoms2+1);
    snew(delta_alpha_strich_2_y,natoms2+1);
    snew(delta_alpha_strich_2_z,natoms2+1);        
       
    };
    
    beta2 = beta2/(double)number_of_replicas;

    delta_alpha1   = 1E-4;
    delta_alpha2   = 1E-4;    
    
    r_g1=1.0;
    
    snew(alpha_strich,natoms2+1);

    x_1 = malloc(sizeof(double)*(dend+1)*(dend+1));
    y_1 = malloc(sizeof(double)*(dend+1)*(dend+1));
    z_1 = malloc(sizeof(double)*(dend+1)*(dend+1));
       
    charge_at_site = malloc(sizeof(double)*(dend+1)*(dend+1));
   
    sigma_strich_rdf = malloc(sizeof(double)*(natoms2+1));
    
    for(i=1;i<=natoms2;i++){
        
        sigma_strich_rdf[i] = malloc(sizeof(double)*(natoms2+1)); 
        
    };    
    
    
    gmx_multisim_t *ms;

    ms = cr->ms;
    
    snew(d_x2,natoms1+1);
    snew(d_y2,natoms1+1);
    snew(d_z2,natoms1+1);
    snew(d_x3,natoms1+1);
    snew(d_y3,natoms1+1);
    snew(d_z3,natoms1+1);
    
    snew(h1,natoms1+1);
    snew(h2,natoms1+1);
    snew(h3,natoms1+1);
    snew(count2,number_of_replicas+1);
    
    snew(delta_path_x,natoms1+1);
    snew(delta_path_y,natoms1+1);
    snew(delta_path_z,natoms1+1);
    snew(sum_f_x,natoms1+1);
    snew(sum_f_y,natoms1+1);
    snew(sum_f_z,natoms1+1);   
    snew(sum_x,101);
    snew(sum_y,101);
    snew(sum_z,101);
    snew(sum_delta_path_x,natoms1+1);
    snew(sum_delta_path_y,natoms1+1);
    snew(sum_delta_path_z,natoms1+1);    
    snew(sum_of_path_x,natoms1+1);
    snew(sum_of_path_y,natoms1+1);
    snew(sum_of_path_z,natoms1+1);     
    snew(av_grad_x,natoms1+1);
    snew(av_grad_y,natoms1+1);
    snew(av_grad_z,natoms1+1);  
    
    snew(df_x_imp,natoms1+1);
    snew(df_y_imp,natoms1+1);
    snew(df_z_imp,natoms1+1);
    snew(vec_dL_x,natoms1+1);
    snew(vec_dL_y,natoms1+1);
    snew(vec_dL_z,natoms1+1);
    snew(vec_dL_x2,natoms1+1);
    snew(vec_dL_y2,natoms1+1);
    snew(vec_dL_z2,natoms1+1);    
    snew(vec_dL_x3,natoms1+1);
    snew(vec_dL_y3,natoms1+1);
    snew(vec_dL_z3,natoms1+1);     
      
    velocity = malloc(sizeof(double)*(num_grid+1));
    
    for(i=1;i<=num_grid+1;i++){
        
        velocity[i] = malloc(sizeof(double)*(3)); 
        
    };    
    
    
    grad_meta_x = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        grad_meta_x[i] = malloc(sizeof(double)*(101)); 
        
    };
    
    grad_meta_y = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        grad_meta_y[i] = malloc(sizeof(double)*(101)); 
        
    };    
    
    grad_meta_z = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        grad_meta_z[i] = malloc(sizeof(double)*(101)); 
        
    };     
    
    dL_abs = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        dL_abs[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };             
    
    rho = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        rho[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };             
    
    sum_rho = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        sum_rho[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };             
       
    prob_rho = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        prob_rho[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };     
    
    sigma_strich_xx = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        sigma_strich_xx[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };     
    
    sigma_strich_yx = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        sigma_strich_yx[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };        
           
    sigma_strich_zx = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        sigma_strich_zx[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };    
    
    delta_fx4 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_fx4[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };  
    
    delta_fy4 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_fy4[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };      
    
    delta_fz4 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_fz4[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };      
    
    delta_fx5 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_fx5[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };  
    
    delta_fy5 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_fy5[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };      
    
    delta_fz5 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_fz5[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };  
    
    delta_t2 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_t2[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };  
 
    diff_t_last = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        diff_t_last[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };      
    
    tau = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        tau[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };  
    
    tau2 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        tau2[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };     
    
    tau_1 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        tau_1[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };     
    
    delta_tau = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_tau[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    }; 
    
    delta_tau_2 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_tau_2[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };     

    d_x1 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        d_x1[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };     
    
    d_x1_s = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        d_x1_s[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };    
    
    d_y1 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        d_y1[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };     
    
    d_y1_s = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        d_y1_s[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };     
    
    d_z1 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        d_z1[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };     
    
    d_z1_s = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        d_z1_s[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };     

    df_x_s = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        df_x_s[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    }; 

    df_y_s = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        df_y_s[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    }; 

    df_z_s = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        df_z_s[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };

    delta_L_x = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_L_x[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };    
    delta_L_y = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_L_y[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };  
    delta_L_z = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_L_z[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };  
    delta_L_x2 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_L_x2[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };    
    delta_L_y2 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_L_y2[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };  
    delta_L_z2 = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_L_z2[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };     
    alpha_1_x = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        alpha_1_x[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    }; 
    alpha_1_y = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        alpha_1_y[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    }; 
    alpha_1_z = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        alpha_1_z[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };   
    alpha_2_x = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        alpha_2_x[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    }; 
    alpha_2_y = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        alpha_2_y[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    }; 
    alpha_2_z = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        alpha_2_z[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };
    
    delta_alpha_1x = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_alpha_1x[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    }; 
    delta_alpha_1y = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_alpha_1y[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    }; 
    delta_alpha_1z = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_alpha_1z[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };    
    delta_alpha_2x = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_alpha_2x[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    }; 
    delta_alpha_2y = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_alpha_2y[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    }; 
    delta_alpha_2z = malloc(sizeof(double)*(natoms1+1));
    
    for(i=1;i<=natoms1;i++){
        
        delta_alpha_2z[i] = malloc(sizeof(double)*(number_of_replicas+1)); 
        
    };             

    epsilon2 = malloc(sizeof(double)*(natoms1+1));

    for(i=1;i<=natoms1;i++){

        epsilon2[i] = malloc(sizeof(double)*(number_of_replicas+1));

    };   

    epsilon3 = malloc(sizeof(double)*(natoms1+1));

    for(i=1;i<=natoms1;i++){

        epsilon3[i] = malloc(sizeof(double)*(number_of_replicas+1));

    };

    epsilon6 = malloc(sizeof(double)*(natoms1+1));

    for(i=1;i<=natoms1;i++){

        epsilon6[i] = malloc(sizeof(double)*(number_of_replicas+1));

    };

    epsilon7 = malloc(sizeof(double)*(natoms1+1));

    for(i=1;i<=natoms1;i++){

        epsilon7[i] = malloc(sizeof(double)*(number_of_replicas+1));

    };
    
    snew(grad_x1,natoms1+1);
    snew(grad_y1,natoms1+1);
    snew(grad_z1,natoms1+1);
    snew(grad_x0,natoms1+1);
    snew(grad_y0,natoms1+1);
    snew(grad_z0,natoms1+1);    
    snew(delta_s,natoms1+1);
    snew(diff_t2,natoms1+1);
    snew(grad_x,natoms1+1);
    snew(grad_y,natoms1+1);
    snew(grad_z,natoms1+1);
    snew(x0_s,natoms1+1);
    snew(y0_s,natoms1+1);
    snew(z0_s,natoms1+1);
    snew(x1_s,natoms1+1);
    snew(y1_s,natoms1+1);
    snew(z1_s,natoms1+1);
    snew(x0,natoms1+1);
    snew(y0,natoms1+1);
    snew(z0,natoms1+1);
    snew(x1,natoms1+1);
    snew(y1,natoms1+1);
    snew(z1,natoms1+1);    
    snew(r_g,natoms1+1);
    snew(df_x1,natoms1+1);
    snew(df_y1,natoms1+1);
    snew(df_z1,natoms1+1);
    snew(df_x1b,natoms1+1);
    snew(df_y1b,natoms1+1);
    snew(df_z1b,natoms1+1);    
    snew(delta_fx,natoms1+1);
    snew(delta_fy,natoms1+1);
    snew(delta_fz,natoms1+1);
    snew(delta_fx1,natoms1+1);
    snew(delta_fy1,natoms1+1);
    snew(delta_fz1,natoms1+1);    
    snew(delta_fx2,natoms1+1);
    snew(delta_fy2,natoms1+1);
    snew(delta_fz2,natoms1+1);  
    snew(delta_fx3,natoms1+1);
    snew(delta_fy3,natoms1+1);
    snew(delta_fz3,natoms1+1);    
    snew(f_x1_s,natoms1+1);
    snew(f_y1_s,natoms1+1);
    snew(f_z1_s,natoms1+1);
    snew(f_x2_s,natoms1+1);
    snew(f_y2_s,natoms1+1);
    snew(f_z2_s,natoms1+1);    
    snew(f_x1,natoms1+1);
    snew(f_y1,natoms1+1);
    snew(f_z1,natoms1+1);
    snew(f_x2,natoms1+1);
    snew(f_y2,natoms1+1);
    snew(f_z2,natoms1+1);    
    snew(df_x,natoms1+1);
    snew(df_y,natoms1+1);
    snew(df_z,natoms1+1);    
    snew(sigma_xx,natoms1+1);
    snew(sigma_yx,natoms1+1);
    snew(sigma_zx,natoms1+1); 
    snew(step_father,number_of_replicas+1);
    snew(alpha_3_x,natoms1+1);
    snew(alpha_3_y,natoms1+1);
    snew(alpha_3_z,natoms1+1);        
    snew(beta_sum_x,natoms1+1);
    snew(beta_sum_y,natoms1+1);
    snew(beta_sum_z,natoms1+1); 
    
    snew(path_x,natoms1+1);
    snew(path_y,natoms1+1);
    snew(path_z,natoms1+1);
    snew(path_x2,natoms1+1);
    snew(path_y2,natoms1+1);
    snew(path_z2,natoms1+1);
    snew(path_x3,natoms1+1);
    snew(path_y3,natoms1+1);
    snew(path_z3,natoms1+1);
    
    for(i=1;i<=natoms1;i++){
        
      for(nr=1;nr<=number_of_replicas;nr++) {  
        
        tau[i][nr] = tau_a*nr;
        tau_1[i][nr] = tau_a*nr;
        tau2[i][nr] = tau2_a*nr;
        delta_tau[i][nr] = 1;
        delta_tau_2[i][nr] = 1;
        
      };
    };

if(tau_c > 0) {    
    
    if(MASTER(cr)) {
    
        
    for(i=1;i<=natoms2;i++){

       counter_t_a[i] = 0.0;
       d_t_a0[i]      = 0.0;

       for(k=1;k<=natoms2;k++){ 
        
         n = 1;  
           
         if(i != k) {  
           
         char filechar[200];  
         char string[70];
         double fee_en_value;
         
         sprintf(filechar,"%s%d%d%s","rdf_",at_id_b[i]+1,at_id_b[k]+1,".dat"); 
         
         printf("%s\n",filechar);         
         
         fp = fopen(filechar,"r"); 
         
         while(fgets(string,70,fp)!=NULL){
                 sscanf(string,"%lg\t%lg\n",&radius,&fee_en_value);
                // printf("%lg\t%lg\n",radius,fee_en_value);
                 fee_en[i][k][n] = fee_en_value;
                 n++;
             };    
             
           number_of_lines = n;
          range_dist      = radius;
                         
         fclose(fp); 

         };
       };
     };
    };
};    
    
    
    MPI_Barrier(cr->mpi_comm_mygroup); 
    
    if(MASTER(cr)) printf("%d\t%d\t%d\t%lg\t%lg\t%lg\t%lg\t%d\t%d\t%d\t%s\n",dstart,dend,natoms1,diff_t,stepsize_alpha,range_alpha,epsilon,tau_a,tau2_a,
           number_of_replicas,"dstart and dend, diff_t, stepsize_alpha, range_alpha, epsilon, tau,tau2 - number of rep. ");  

    /* Check for special mdrun options */
    bRerunMD = (Flags & MD_RERUN);
    bIonize  = (Flags & MD_IONIZE);
    bFFscan  = (Flags & MD_FFSCAN);
    bAppend  = (Flags & MD_APPENDFILES);
    if (Flags & MD_RESETCOUNTERSHALFWAY)
    {
        if (ir->nsteps > 0)
        {
            /* Signal to reset the counters half the simulation steps. */
            wcycle_set_reset_counters(wcycle,ir->nsteps/2);
        }
        /* Signal to reset the counters halfway the simulation time. */
        bResetCountersHalfMaxH = (max_hours > 0);
    }

    /* md-vv uses averaged full step velocities for T-control 
       md-vv-avek uses averaged half step velocities for T-control (but full step ekin for P control)
       md uses averaged half step kinetic energies to determine temperature unless defined otherwise by GMX_EKIN_AVE_VEL; */
    bVV = EI_VV(ir->eI);
    if (bVV) /* to store the initial velocities while computing virial */
    {
        snew(cbuf,top_global->natoms);
    }
    /* all the iteratative cases - only if there are constraints */ 
    bIterations = ((IR_NPT_TROTTER(ir)) && (constr) && (!bRerunMD));
    bTrotter = (bVV && (IR_NPT_TROTTER(ir) || (IR_NVT_TROTTER(ir))));        
    
    if (bRerunMD)
    {
        /* Since we don't know if the frames read are related in any way,
         * rebuild the neighborlist at every step.
         */
        ir->nstlist       = 1;
        ir->nstcalcenergy = 1;
        nstglobalcomm     = 1;
    }

    check_ir_old_tpx_versions(cr,fplog,ir,top_global);

    nstglobalcomm = check_nstglobalcomm(fplog,cr,nstglobalcomm,ir);
    bGStatEveryStep = (nstglobalcomm == 1);

    if (!bGStatEveryStep && ir->nstlist == -1 && fplog != NULL)
    {
        fprintf(fplog,
                "To reduce the energy communication with nstlist = -1\n"
                "the neighbor list validity should not be checked at every step,\n"
                "this means that exact integration is not guaranteed.\n"
                "The neighbor list validity is checked after:\n"
                "  <n.list life time> - 2*std.dev.(n.list life time)  steps.\n"
                "In most cases this will result in exact integration.\n"
                "This reduces the energy communication by a factor of 2 to 3.\n"
                "If you want less energy communication, set nstlist > 3.\n\n");
    }

    if (bRerunMD || bFFscan)
    {
        ir->nstxtcout = 0;
    }
    groups = &top_global->groups;

    /* Initial values */
    init_md(fplog,cr,ir,oenv,&t,&t0,&state_global->lambda,&lam0,
            nrnb,top_global,&upd,
            nfile,fnm,&outf,&mdebin,
            force_vir,shake_vir,mu_tot,&bSimAnn,&vcm,state_global,Flags);

    clear_mat(total_vir);
    clear_mat(pres);
    /* Energy terms and groups */
    snew(enerd,1);
    init_enerdata(top_global->groups.grps[egcENER].nr,ir->n_flambda,enerd);
    if (DOMAINDECOMP(cr))
    {
        f = NULL;
    }
    else
    {
        snew(f,top_global->natoms);
    }

    /* Kinetic energy data */
    snew(ekind,1);
    init_ekindata(fplog,top_global,&(ir->opts),ekind);
    /* needed for iteration of constraints */
    snew(ekind_save,1);
    init_ekindata(fplog,top_global,&(ir->opts),ekind_save);
    /* Copy the cos acceleration to the groups struct */    
    ekind->cosacc.cos_accel = ir->cos_accel;

    gstat = global_stat_init(ir);
    debug_gmx();

    /* Check for polarizable models and flexible constraints */
    shellfc = init_shell_flexcon(fplog,
                                 top_global,n_flexible_constraints(constr),
                                 (ir->bContinuation || 
                                  (DOMAINDECOMP(cr) && !MASTER(cr))) ?
                                 NULL : state_global->x);

    if (DEFORM(*ir))
    {
#ifdef GMX_THREADS
        tMPI_Thread_mutex_lock(&deform_init_box_mutex);
#endif
        set_deform_reference_box(upd,
                                 deform_init_init_step_tpx,
                                 deform_init_box_tpx);
#ifdef GMX_THREADS
        tMPI_Thread_mutex_unlock(&deform_init_box_mutex);
#endif
    }

    {
        double io = compute_io(ir,top_global->natoms,groups,mdebin->ebin->nener,1);
        if ((io > 2000) && MASTER(cr))
            fprintf(stderr,
                    "\nWARNING: This run will generate roughly %.0f Mb of data\n\n",
                    io);
    }

    if (DOMAINDECOMP(cr)) {
        top = dd_init_local_top(top_global);

        snew(state,1);
        dd_init_local_state(cr->dd,state_global,state);

    //    if (DDMASTER(cr->dd) && ir->nstfout) {
            snew(f_global,state_global->natoms);
    //     }
    } else {
        if (PAR(cr)) {
            /* Initialize the particle decomposition and split the topology */
            top = split_system(fplog,top_global,ir,cr);

            pd_cg_range(cr,&fr->cg0,&fr->hcg);
            pd_at_range(cr,&a0,&a1);
        } else {
            top = gmx_mtop_generate_local_top(top_global,ir);

            a0 = 0;
            a1 = top_global->natoms;
        }

        state = partdec_init_local_state(cr,state_global);
        f_global = f;

        atoms2md(top_global,ir,0,NULL,a0,a1-a0,mdatoms);

        if (vsite) {
            set_vsite_top(vsite,top,mdatoms,cr);
        }

        if (ir->ePBC != epbcNONE && !ir->bPeriodicMols) {
            graph = mk_graph(fplog,&(top->idef),0,top_global->natoms,FALSE,FALSE);
        }

        if (shellfc) {
            make_local_shells(cr,mdatoms,shellfc);
        }

        if (ir->pull && PAR(cr)) {
            dd_make_local_pull_groups(NULL,ir->pull,mdatoms);
        }
    }

    if (DOMAINDECOMP(cr))
    {
        /* Distribute the charge groups over the nodes from the master node */
        dd_partition_system(fplog,ir->init_step,cr,TRUE,1,
                            state_global,top_global,ir,
                            state,&f,mdatoms,top,fr,
                            vsite,shellfc,constr,
                            nrnb,wcycle,FALSE);
    }

    update_mdatoms(mdatoms,state->lambda);

    if (MASTER(cr))
    {
        if (opt2bSet("-cpi",nfile,fnm))
        {
            /* Update mdebin with energy history if appending to output files */
            if ( Flags & MD_APPENDFILES )
            {
                restore_energyhistory_from_state(mdebin,&state_global->enerhist);
            }
            else
            {
                /* We might have read an energy history from checkpoint,
                 * free the allocated memory and reset the counts.
                 */
                done_energyhistory(&state_global->enerhist);
                init_energyhistory(&state_global->enerhist);
            }
        }
        /* Set the initial energy history in state by updating once */
        update_energyhistory(&state_global->enerhist,mdebin);
    }	

    if ((state->flags & (1<<estLD_RNG)) && (Flags & MD_READ_RNG)) {
        /* Set the random state if we read a checkpoint file */
        set_stochd_state(upd,state);
    }

    /* Initialize constraints */
    if (constr) {
        if (!DOMAINDECOMP(cr))
            set_constraints(constr,top,ir,mdatoms,cr);
    }

    /* Check whether we have to GCT stuff */
    bTCR = ftp2bSet(efGCT,nfile,fnm);
    if (bTCR) {
        if (MASTER(cr)) {
            fprintf(stderr,"Will do General Coupling Theory!\n");
        }
        gnx = top_global->mols.nr;
        snew(grpindex,gnx);
        for(i=0; (i<gnx); i++) {
            grpindex[i] = i;
        }
    }

    if (repl_ex_nst > 0)
    {
        /* We need to be sure replica exchange can only occur
         * when the energies are current */
        check_nst_param(fplog,cr,"nstcalcenergy",ir->nstcalcenergy,
                        "repl_ex_nst",&repl_ex_nst);
        /* This check needs to happen before inter-simulation
         * signals are initialized, too */
    }
    if (repl_ex_nst > 0 && MASTER(cr))
        repl_ex = init_replica_exchange(fplog,cr->ms,state_global,ir,
                                        repl_ex_nst,repl_ex_seed);

    if (!ir->bContinuation && !bRerunMD)
    {
        if (mdatoms->cFREEZE && (state->flags & (1<<estV)))
        {
            /* Set the velocities of frozen particles to zero */
            for(i=mdatoms->start; i<mdatoms->start+mdatoms->homenr; i++)
            {
                for(m=0; m<DIM; m++)
                {
                    if (ir->opts.nFreeze[mdatoms->cFREEZE[i]][m])
                    {
                        state->v[i][m] = 0;
                    }
                }
            }
        }

        if (constr)
        {
            /* Constrain the initial coordinates and velocities */
            do_constrain_first(fplog,constr,ir,mdatoms,state,f,
                               graph,cr,nrnb,fr,top,shake_vir);
        }
        if (vsite)
        {
            /* Construct the virtual sites for the initial configuration */
            construct_vsites(fplog,vsite,state->x,nrnb,ir->delta_t,NULL,
                             top->idef.iparams,top->idef.il,
                             fr->ePBC,fr->bMolPBC,graph,cr,state->box);
        }
    }

    debug_gmx();
  
    /* I'm assuming we need global communication the first time! MRS */
    cglo_flags = (CGLO_TEMPERATURE | CGLO_GSTAT
                  | (bVV ? CGLO_PRESSURE:0)
                  | (bVV ? CGLO_CONSTRAINT:0)
                  | (bRerunMD ? CGLO_RERUNMD:0)
                  | ((Flags & MD_READ_EKIN) ? CGLO_READEKIN:0));
    
    bSumEkinhOld = FALSE;
    compute_globals(fplog,gstat,cr,ir,fr,ekind,state,state_global,mdatoms,nrnb,vcm,
                    wcycle,enerd,force_vir,shake_vir,total_vir,pres,mu_tot,
                    constr,NULL,FALSE,state->box,
                    top_global,&pcurr,top_global->natoms,&bSumEkinhOld,cglo_flags);
    if (ir->eI == eiVVAK) {
        /* a second call to get the half step temperature initialized as well */ 
        /* we do the same call as above, but turn the pressure off -- internally to 
           compute_globals, this is recognized as a velocity verlet half-step 
           kinetic energy calculation.  This minimized excess variables, but 
           perhaps loses some logic?*/
        
        compute_globals(fplog,gstat,cr,ir,fr,ekind,state,state_global,mdatoms,nrnb,vcm,
                        wcycle,enerd,force_vir,shake_vir,total_vir,pres,mu_tot,
                        constr,NULL,FALSE,state->box,
                        top_global,&pcurr,top_global->natoms,&bSumEkinhOld,
                        cglo_flags &~ CGLO_PRESSURE);
    }
    
    /* Calculate the initial half step temperature, and save the ekinh_old */
    if (!(Flags & MD_STARTFROMCPT)) 
    {
        for(i=0; (i<ir->opts.ngtc); i++) 
        {
            copy_mat(ekind->tcstat[i].ekinh,ekind->tcstat[i].ekinh_old);
        } 
    }
    if (ir->eI != eiVV) 
    {
        enerd->term[F_TEMP] *= 2; /* result of averages being done over previous and current step,
                                     and there is no previous step */
    }
    temp0 = enerd->term[F_TEMP];
    
    /* if using an iterative algorithm, we need to create a working directory for the state. */
    if (bIterations) 
    {
            bufstate = init_bufstate(state);
    }
    if (bFFscan) 
    {
        snew(xcopy,state->natoms);
        snew(vcopy,state->natoms);
        copy_rvecn(state->x,xcopy,0,state->natoms);
        copy_rvecn(state->v,vcopy,0,state->natoms);
        copy_mat(state->box,boxcopy);
    } 
    
    /* need to make an initiation call to get the Trotter variables set, as well as other constants for non-trotter
       temperature control */
    trotter_seq = init_npt_vars(ir,state,&MassQ,bTrotter);
    
    if (MASTER(cr))
    {
        if (constr && !ir->bContinuation && ir->eConstrAlg == econtLINCS)
        {
            fprintf(fplog,
                    "RMS relative constraint deviation after constraining: %.2e\n",
                    constr_rmsd(constr,FALSE));
        }
        fprintf(fplog,"Initial temperature: %g K\n",enerd->term[F_TEMP]);
        if (bRerunMD)
        {
            fprintf(stderr,"starting md rerun '%s', reading coordinates from"
                    " input trajectory '%s'\n\n",
                    *(top_global->name),opt2fn("-rerun",nfile,fnm));
            if (bVerbose)
            {
                fprintf(stderr,"Calculated time to finish depends on nsteps from "
                        "run input file,\nwhich may not correspond to the time "
                        "needed to process input trajectory.\n\n");
            }
        }
        else
        {
            char tbuf[20];
            fprintf(stderr,"starting mdrun '%s'\n",
                    *(top_global->name));
            if (ir->nsteps >= 0)
            {
                sprintf(tbuf,"%8.1f",(ir->init_step+ir->nsteps)*ir->delta_t);
            }
            else
            {
                sprintf(tbuf,"%s","infinite");
            }
            if (ir->init_step > 0)
            {
                fprintf(stderr,"%s steps, %s ps (continuing from step %s, %8.1f ps).\n",
                        gmx_step_str(ir->init_step+ir->nsteps,sbuf),tbuf,
                        gmx_step_str(ir->init_step,sbuf2),
                        ir->init_step*ir->delta_t);
            }
            else
            {
                fprintf(stderr,"%s steps, %s ps.\n",
                        gmx_step_str(ir->nsteps,sbuf),tbuf);
            }
        }
        fprintf(fplog,"\n");
    }

    /* Set and write start time */
    runtime_start(runtime);
    print_date_and_time(fplog,cr->nodeid,"Started mdrun",runtime);
    wallcycle_start(wcycle,ewcRUN);
    if (fplog)
        fprintf(fplog,"\n");

    /* safest point to do file checkpointing is here.  More general point would be immediately before integrator call */
#ifdef GMX_FAHCORE
    chkpt_ret=fcCheckPointParallel( cr->nodeid,
                                    NULL,0);
    if ( chkpt_ret == 0 ) 
        gmx_fatal( 3,__FILE__,__LINE__, "Checkpoint error on step %d\n", 0 );
#endif

    debug_gmx();
    /***********************************************************
     *
     *             Loop over MD steps 
     *
     ************************************************************/

    /* if rerunMD then read coordinates and velocities from input trajectory */
    if (bRerunMD)
    {
        if (getenv("GMX_FORCE_UPDATE"))
        {
            bForceUpdate = TRUE;
        }

        rerun_fr.natoms = 0;
        if (MASTER(cr))
        {
            bNotLastFrame = read_first_frame(oenv,&status,
                                             opt2fn("-rerun",nfile,fnm),
                                             &rerun_fr,TRX_NEED_X | TRX_READ_V);
            if (rerun_fr.natoms != top_global->natoms)
            {
                gmx_fatal(FARGS,
                          "Number of atoms in trajectory (%d) does not match the "
                          "run input file (%d)\n",
                          rerun_fr.natoms,top_global->natoms);
            }
            if (ir->ePBC != epbcNONE)
            {
                if (!rerun_fr.bBox)
                {
                    gmx_fatal(FARGS,"Rerun trajectory frame step %d time %f does not contain a box, while pbc is used",rerun_fr.step,rerun_fr.time);
                }
                if (max_cutoff2(ir->ePBC,rerun_fr.box) < sqr(fr->rlistlong))
                {
                    gmx_fatal(FARGS,"Rerun trajectory frame step %d time %f has too small box dimensions",rerun_fr.step,rerun_fr.time);
                }
            }
        }

        if (PAR(cr))
        {
            rerun_parallel_comm(cr,&rerun_fr,&bNotLastFrame);
        }

        if (ir->ePBC != epbcNONE)
        {
            /* Set the shift vectors.
             * Necessary here when have a static box different from the tpr box.
             */
            calc_shifts(rerun_fr.box,fr->shift_vec);
        }
    }

    /* loop over MD steps or if rerunMD to end of input trajectory */
    bFirstStep = TRUE;
    /* Skip the first Nose-Hoover integration when we get the state from tpx */
    bStateFromTPX = !opt2bSet("-cpi",nfile,fnm);
    bInitStep = bFirstStep && (bStateFromTPX || bVV);
    bStartingFromCpt = (Flags & MD_STARTFROMCPT) && bInitStep;
    bLastStep    = FALSE;
    bSumEkinhOld = FALSE;
    bExchanged   = FALSE;

    init_global_signals(&gs,cr,ir,repl_ex_nst);

    step = ir->init_step;
    step_rel = 0;

    if (ir->nstlist == -1)
    {
        init_nlistheuristics(&nlh,bGStatEveryStep,step);
    }

    if (MULTISIM(cr) && (repl_ex_nst <=0 ))
    {
        /* check how many steps are left in other sims */
        multisim_nsteps=get_multisim_nsteps(cr, ir->nsteps);
    }


    /* and stop now if we should */
    bLastStep = (bRerunMD || (ir->nsteps >= 0 && step_rel > ir->nsteps) ||
                 ((multisim_nsteps >= 0) && (step_rel >= multisim_nsteps )));
    while (!bLastStep || (bRerunMD && bNotLastFrame)) {

        wallcycle_start(wcycle,ewcSTEP);

        GMX_MPE_LOG(ev_timestep1);

        if (bRerunMD) {
            if (rerun_fr.bStep) {
                step = rerun_fr.step;
                step_rel = step - ir->init_step;
            }
            if (rerun_fr.bTime) {
                t = rerun_fr.time;
            }
            else
            {
                t = step;
            }
        } 
        else 
        {
            bLastStep = (step_rel == ir->nsteps);
            t = t0 + step*ir->delta_t;
        }

        if (ir->efep != efepNO)
        {
            if (bRerunMD && rerun_fr.bLambda && (ir->delta_lambda!=0))
            {
                state_global->lambda = rerun_fr.lambda;
            }
            else
            {
                state_global->lambda = lam0 + step*ir->delta_lambda;
            }
            state->lambda = state_global->lambda;
            bDoDHDL = do_per_step(step,ir->nstdhdl);
        }

        if (bSimAnn) 
        {
            update_annealing_target_temp(&(ir->opts),t);
        }

        if (bRerunMD)
        {
            if (!(DOMAINDECOMP(cr) && !MASTER(cr)))
            {
                for(i=0; i<state_global->natoms; i++)
                {
                    copy_rvec(rerun_fr.x[i],state_global->x[i]);
                }
                if (rerun_fr.bV)
                {
                    for(i=0; i<state_global->natoms; i++)
                    {
                        copy_rvec(rerun_fr.v[i],state_global->v[i]);
                    }
                }
                else
                {
                    for(i=0; i<state_global->natoms; i++)
                    {
                        clear_rvec(state_global->v[i]);
                    }
                    if (bRerunWarnNoV)
                    {
                        fprintf(stderr,"\nWARNING: Some frames do not contain velocities.\n"
                                "         Ekin, temperature and pressure are incorrect,\n"
                                "         the virial will be incorrect when constraints are present.\n"
                                "\n");
                        bRerunWarnNoV = FALSE;
                    }
                }
            }
            copy_mat(rerun_fr.box,state_global->box);
            copy_mat(state_global->box,state->box);

            if (vsite && (Flags & MD_RERUN_VSITE))
            {
                if (DOMAINDECOMP(cr))
                {
                    gmx_fatal(FARGS,"Vsite recalculation with -rerun is not implemented for domain decomposition, use particle decomposition");
                }
                if (graph)
                {
                    /* Following is necessary because the graph may get out of sync
                     * with the coordinates if we only have every N'th coordinate set
                     */
                    mk_mshift(fplog,graph,fr->ePBC,state->box,state->x);
                    shift_self(graph,state->box,state->x);
                }
                construct_vsites(fplog,vsite,state->x,nrnb,ir->delta_t,state->v,
                                 top->idef.iparams,top->idef.il,
                                 fr->ePBC,fr->bMolPBC,graph,cr,state->box);
                if (graph)
                {
                    unshift_self(graph,state->box,state->x);
                }
            }
        }

        /* Stop Center of Mass motion */
        bStopCM = (ir->comm_mode != ecmNO && do_per_step(step,ir->nstcomm));

        /* Copy back starting coordinates in case we're doing a forcefield scan */
        if (bFFscan)
        {
            for(ii=0; (ii<state->natoms); ii++)
            {
                copy_rvec(xcopy[ii],state->x[ii]);
                copy_rvec(vcopy[ii],state->v[ii]);
            }
            copy_mat(boxcopy,state->box);
        }

        if (bRerunMD)
        {
            /* for rerun MD always do Neighbour Searching */
            bNS = (bFirstStep || ir->nstlist != 0);
            bNStList = bNS;
        }
        else
        {
            /* Determine whether or not to do Neighbour Searching and LR */
            bNStList = (ir->nstlist > 0  && step % ir->nstlist == 0);
            
            bNS = (bFirstStep || bExchanged || bNStList ||
                   (ir->nstlist == -1 && nlh.nabnsb > 0));

            if (bNS && ir->nstlist == -1)
            {
                set_nlistheuristics(&nlh,bFirstStep || bExchanged,step);
            }
        } 

        /* check whether we should stop because another simulation has 
           stopped. */
        if (MULTISIM(cr))
        {
            if ( (multisim_nsteps >= 0) &&  (step_rel >= multisim_nsteps)  &&  
                 (multisim_nsteps != ir->nsteps) )  
            {
                if (bNS)
                {
                    if (MASTER(cr))
                    {
                        fprintf(stderr, 
                                "Stopping simulation %d because another one has finished\n",
                                cr->ms->sim);
                    }
                    bLastStep=TRUE;
                    gs.sig[eglsCHKPT] = 1;
                }
            }
        }

        /* < 0 means stop at next step, > 0 means stop at next NS step */
        if ( (gs.set[eglsSTOPCOND] < 0 ) ||
             ( (gs.set[eglsSTOPCOND] > 0 ) && ( bNS || ir->nstlist==0)) )
        {
            bLastStep = TRUE;
        }

        /* Determine whether or not to update the Born radii if doing GB */
        bBornRadii=bFirstStep;
        if (ir->implicit_solvent && (step % ir->nstgbradii==0))
        {
            bBornRadii=TRUE;
        }
        
        do_log = do_per_step(step,ir->nstlog) || bFirstStep || bLastStep;
        do_verbose = bVerbose &&
                  (step % stepout == 0 || bFirstStep || bLastStep);

        if (bNS && !(bFirstStep && ir->bContinuation && !bRerunMD))
        {
            if (bRerunMD)
            {
                bMasterState = TRUE;
            }
            else
            {
                bMasterState = FALSE;
                /* Correct the new box if it is too skewed */
                if (DYNAMIC_BOX(*ir))
                {
                    if (correct_box(fplog,step,state->box,graph))
                    {
                        bMasterState = TRUE;
                    }
                }
                if (DOMAINDECOMP(cr) && bMasterState)
                {
                    dd_collect_state(cr->dd,state,state_global);
                }
            }

            if (DOMAINDECOMP(cr))
            {
                /* Repartition the domain decomposition */
                wallcycle_start(wcycle,ewcDOMDEC);
                dd_partition_system(fplog,step,cr,
                                    bMasterState,nstglobalcomm,
                                    state_global,top_global,ir,
                                    state,&f,mdatoms,top,fr,
                                    vsite,shellfc,constr,
                                    nrnb,wcycle,do_verbose);
                wallcycle_stop(wcycle,ewcDOMDEC);
                /* If using an iterative integrator, reallocate space to match the decomposition */
            }
        }

        if (MASTER(cr) && do_log && !bFFscan)
        {
            print_ebin_header(fplog,step,t,state->lambda);
        }

        if (ir->efep != efepNO)
        {
            update_mdatoms(mdatoms,state->lambda); 
        }

        if (bRerunMD && rerun_fr.bV)
        {
            
            /* We need the kinetic energy at minus the half step for determining
             * the full step kinetic energy and possibly for T-coupling.*/
            /* This may not be quite working correctly yet . . . . */
            compute_globals(fplog,gstat,cr,ir,fr,ekind,state,state_global,mdatoms,nrnb,vcm,
                            wcycle,enerd,NULL,NULL,NULL,NULL,mu_tot,
                            constr,NULL,FALSE,state->box,
                            top_global,&pcurr,top_global->natoms,&bSumEkinhOld,
                            CGLO_RERUNMD | CGLO_GSTAT | CGLO_TEMPERATURE);
        }
        clear_mat(force_vir);
        
        /* Ionize the atoms if necessary */
        if (bIonize)
        {
            ionize(fplog,oenv,mdatoms,top_global,t,ir,state->x,state->v,
                   mdatoms->start,mdatoms->start+mdatoms->homenr,state->box,cr);
        }
        
        /* Update force field in ffscan program */
        if (bFFscan)
        {
            if (update_forcefield(fplog,
                                  nfile,fnm,fr,
                                  mdatoms->nr,state->x,state->box)) {
                if (gmx_parallel_env_initialized())
                {
                    gmx_finalize();
                }
                exit(0);
            }
        }

        GMX_MPE_LOG(ev_timestep2);

        /* We write a checkpoint at this MD step when:
         * either at an NS step when we signalled through gs,
         * or at the last step (but not when we do not want confout),
         * but never at the first step or with rerun.
         */
        bCPT = (((gs.set[eglsCHKPT] && (bNS || ir->nstlist == 0)) ||
                 (bLastStep && (Flags & MD_CONFOUT))) &&
                step > ir->init_step && !bRerunMD);
        if (bCPT)
        {
            gs.set[eglsCHKPT] = 0;
        }

        /* Determine the energy and pressure:
         * at nstcalcenergy steps and at energy output steps (set below).
         */
        bNstEner = do_per_step(step,ir->nstcalcenergy);
        bCalcEnerPres =
            (bNstEner ||
             (ir->epc != epcNO && do_per_step(step,ir->nstpcouple)));

        /* Do we need global communication ? */
        bGStat = (bCalcEnerPres || bStopCM ||
                  do_per_step(step,nstglobalcomm) ||
                  (ir->nstlist == -1 && !bRerunMD && step >= nlh.step_nscheck));

        do_ene = (do_per_step(step,ir->nstenergy) || bLastStep);

        if (do_ene || do_log)
        {
            bCalcEnerPres = TRUE;
            bGStat        = TRUE;
        }
        
        /* these CGLO_ options remain the same throughout the iteration */
        cglo_flags = ((bRerunMD ? CGLO_RERUNMD : 0) |
                      (bStopCM ? CGLO_STOPCM : 0) |
                      (bGStat ? CGLO_GSTAT : 0)
            );
        
        force_flags = (GMX_FORCE_STATECHANGED |
                       ((DYNAMIC_BOX(*ir) || bRerunMD) ? GMX_FORCE_DYNAMICBOX : 0) |
                       GMX_FORCE_ALLFORCES |
                       (bNStList ? GMX_FORCE_DOLR : 0) |
                       GMX_FORCE_SEPLRF |
                       (bCalcEnerPres ? GMX_FORCE_VIRIAL : 0) |
                       (bDoDHDL ? GMX_FORCE_DHDL : 0)
            );
        
        if (shellfc)
        {
            /* Now is the time to relax the shells */
            count=relax_shell_flexcon(fplog,cr,bVerbose,bFFscan ? step+1 : step,
                                      ir,bNS,force_flags,
                                      bStopCM,top,top_global,
                                      constr,enerd,fcd,
                                      state,f,force_vir,mdatoms,
                                      nrnb,wcycle,graph,groups,
                                      shellfc,fr,bBornRadii,t,mu_tot,
                                      state->natoms,&bConverged,vsite,
                                      outf->fp_field);
            tcount+=count;

            if (bConverged)
            {
                nconverged++;
            }
        }
        else
        {
            /* The coordinates (x) are shifted (to get whole molecules)
             * in do_force.
             * This is parallellized as well, and does communication too. 
             * Check comments in sim_util.c
             */
    /*    if (EEL_RF(fr->eeltype)){
            for(i=0;i<=dend-1;i++) {
                
               for(k=0;k<=3;k++) {
                  
                   velocity[i][k] = state->v[i][k];
                   
                };
            };
        }; */
            
            do_force(fplog,cr,ir,step,nrnb,wcycle,top,top_global,groups,
                     state->box,state->x,&state->hist,
                     f,force_vir,mdatoms,enerd,fcd,
                     state->lambda,graph,
                     fr,vsite,mu_tot,t,outf->fp_field,ed,bBornRadii,
                     (bNS ? GMX_FORCE_NS : 0) | force_flags,x_1,y_1,z_1,
                     charge_at_site,rf_factor,num_grid,friction_constant,velocity,timeframe_sol);
        }
    
        GMX_BARRIER(cr->mpi_comm_mygroup);
        
        if (bTCR)
        {
            mu_aver = calc_mu_aver(cr,state->x,mdatoms->chargeA,
                                   mu_tot,&top_global->mols,mdatoms,gnx,grpindex);
        }
        
        if (bTCR && bFirstStep)
        {
            tcr=init_coupling(fplog,nfile,fnm,cr,fr,mdatoms,&(top->idef));
            fprintf(fplog,"Done init_coupling\n"); 
            fflush(fplog);
        }
        
        if (bVV && !bStartingFromCpt && !bRerunMD)

	  	  
        /*  ############### START FIRST UPDATE HALF-STEP FOR VV METHODS############### */
        {
            if (ir->eI==eiVV && bInitStep) 
            {
                /* if using velocity verlet with full time step Ekin,
                 * take the first half step only to compute the 
                 * virial for the first step. From there,
                 * revert back to the initial coordinates
                 * so that the input is actually the initial step.
                 */
                copy_rvecn(state->v,cbuf,0,state->natoms); /* should make this better for parallelizing? */
            } else {
                /* this is for NHC in the Ekin(t+dt/2) version of vv */
                trotter_update(ir,step,ekind,enerd,state,total_vir,mdatoms,&MassQ,trotter_seq,ettTSEQ1);            
            }

            update_coords(fplog,step,ir,mdatoms,state,
                          f,fr->bTwinRange && bNStList,fr->f_twin,fcd,
                          ekind,M,wcycle,upd,bInitStep,etrtVELOCITY1,
                          cr,nrnb,constr,&top->idef);
            
            if (bIterations)
            {
                gmx_iterate_init(&iterate,bIterations && !bInitStep);
            }
            /* for iterations, we save these vectors, as we will be self-consistently iterating
               the calculations */

            /*#### UPDATE EXTENDED VARIABLES IN TROTTER FORMULATION */
            
            /* save the state */
            if (bIterations && iterate.bIterate) { 
                copy_coupling_state(state,bufstate,ekind,ekind_save,&(ir->opts));
            }
            
            bFirstIterate = TRUE;
            while (bFirstIterate || (bIterations && iterate.bIterate))
            {
                if (bIterations && iterate.bIterate) 
                {
                    copy_coupling_state(bufstate,state,ekind_save,ekind,&(ir->opts));
                    if (bFirstIterate && bTrotter) 
                    {
                        /* The first time through, we need a decent first estimate
                           of veta(t+dt) to compute the constraints.  Do
                           this by computing the box volume part of the
                           trotter integration at this time. Nothing else
                           should be changed by this routine here.  If
                           !(first time), we start with the previous value
                           of veta.  */
                        
                        veta_save = state->veta;
                        trotter_update(ir,step,ekind,enerd,state,total_vir,mdatoms,&MassQ,trotter_seq,ettTSEQ0);
                        vetanew = state->veta;
                        state->veta = veta_save;
                    } 
                } 
                
                bOK = TRUE;
                if ( !bRerunMD || rerun_fr.bV || bForceUpdate) {  /* Why is rerun_fr.bV here?  Unclear. */
                    dvdl = 0;
                    
                    update_constraints(fplog,step,&dvdl,ir,ekind,mdatoms,state,graph,f,
                                       &top->idef,shake_vir,NULL,
                                       cr,nrnb,wcycle,upd,constr,
                                       bInitStep,TRUE,bCalcEnerPres,vetanew);
                    
                    if (!bOK && !bFFscan)
                    {
                        gmx_fatal(FARGS,"Constraint error: Shake, Lincs or Settle could not solve the constrains");
                    }
                    
                } 
                else if (graph)
                { /* Need to unshift here if a do_force has been
                     called in the previous step */
                    unshift_self(graph,state->box,state->x);
                }
                
                
                /* if VV, compute the pressure and constraints */
                /* For VV2, we strictly only need this if using pressure
                 * control, but we really would like to have accurate pressures
                 * printed out.
                 * Think about ways around this in the future?
                 * For now, keep this choice in comments.
                 */
                /*bPres = (ir->eI==eiVV || IR_NPT_TROTTER(ir)); */
                    /*bTemp = ((ir->eI==eiVV &&(!bInitStep)) || (ir->eI==eiVVAK && IR_NPT_TROTTER(ir)));*/
                bPres = TRUE;
                bTemp = ((ir->eI==eiVV &&(!bInitStep)) || (ir->eI==eiVVAK));
                compute_globals(fplog,gstat,cr,ir,fr,ekind,state,state_global,mdatoms,nrnb,vcm,
                                wcycle,enerd,force_vir,shake_vir,total_vir,pres,mu_tot,
                                constr,NULL,FALSE,state->box,
                                top_global,&pcurr,top_global->natoms,&bSumEkinhOld,
                                cglo_flags 
                                | CGLO_ENERGY 
                                | (bTemp ? CGLO_TEMPERATURE:0) 
                                | (bPres ? CGLO_PRESSURE : 0) 
                                | (bPres ? CGLO_CONSTRAINT : 0)
                                | ((bIterations && iterate.bIterate) ? CGLO_ITERATE : 0)  
                                | (bFirstIterate ? CGLO_FIRSTITERATE : 0)
                                | CGLO_SCALEEKIN 
                    );
                /* explanation of above: 
                   a) We compute Ekin at the full time step
                   if 1) we are using the AveVel Ekin, and it's not the
                   initial step, or 2) if we are using AveEkin, but need the full
                   time step kinetic energy for the pressure (always true now, since we want accurate statistics).
                   b) If we are using EkinAveEkin for the kinetic energy for the temperture control, we still feed in 
                   EkinAveVel because it's needed for the pressure */
                
                /* temperature scaling and pressure scaling to produce the extended variables at t+dt */
                if (!bInitStep) 
                {
                    if (bTrotter)
                    {
                        trotter_update(ir,step,ekind,enerd,state,total_vir,mdatoms,&MassQ,trotter_seq,ettTSEQ2);
                    } 
                    else 
                    {
                        update_tcouple(fplog,step,ir,state,ekind,wcycle,upd,&MassQ,mdatoms);
                    }
                }
                
                if (bIterations &&
                    done_iterating(cr,fplog,step,&iterate,bFirstIterate,
                                   state->veta,&vetanew)) 
                {
                    break;
                }
                bFirstIterate = FALSE;
            }

            if (bTrotter && !bInitStep) {
                copy_mat(shake_vir,state->svir_prev);
                copy_mat(force_vir,state->fvir_prev);
                if (IR_NVT_TROTTER(ir) && ir->eI==eiVV) {
                    /* update temperature and kinetic energy now that step is over - this is the v(t+dt) point */
                    enerd->term[F_TEMP] = sum_ekin(&(ir->opts),ekind,NULL,(ir->eI==eiVV),FALSE,FALSE);
                    enerd->term[F_EKIN] = trace(ekind->ekin);
                }
            }
            /* if it's the initial step, we performed this first step just to get the constraint virial */
            if (bInitStep && ir->eI==eiVV) {
                copy_rvecn(cbuf,state->v,0,state->natoms);
            }
            
            if (fr->bSepDVDL && fplog && do_log) 
            {
                fprintf(fplog,sepdvdlformat,"Constraint",0.0,dvdl);
            }
            enerd->term[F_DHDL_CON] += dvdl;
            
            GMX_MPE_LOG(ev_timestep1);
        }
    
        /* MRS -- now done iterating -- compute the conserved quantity */
        if (bVV) {
            saved_conserved_quantity = compute_conserved_from_auxiliary(ir,state,&MassQ);
            if (ir->eI==eiVV) 
            {
                last_ekin = enerd->term[F_EKIN]; /* does this get preserved through checkpointing? */
            }
            if ((ir->eDispCorr != edispcEnerPres) && (ir->eDispCorr != edispcAllEnerPres)) 
            {
                saved_conserved_quantity -= enerd->term[F_DISPCORR];
            }
        }       
        
            if(DOMAINDECOMP(cr)){

                 dd_collect_vec(cr->dd,state,f,f_global);
//                 dd_collect_state(cr->dd,state,state_global);

            };

      stepper++;
      counter_exchange += 1.0;
     if(tau2_a > 0) { 
      if(stepper%(tau2_a/10) == 0) stepper22 ++;
     };
      
    if(MASTER(cr))
	      
     {
                
    if(stepper == 1 && tau2_a > 0) {    
       for(i=1;i<=natoms1;i++){
           for(nr=1;nr<=number_of_replicas;nr++){  
                
                 //|| stepper%((tau[i][nr]))==0) {                      
                      
                      delta_t2[i][nr] = 0.0;  
                      diff_t_last[i][nr] = 0.0;
                      count2[nr] = 0.0;
                    
                      d_x1[i][nr] = 0.0;
                      d_y1[i][nr] = 0.0;
                      d_z1[i][nr] = 0.0;
                  
                      delta_fx1[i] = 0.0;
                      delta_fy1[i] = 0.0;
                      delta_fz1[i] = 0.0;
                      
                      delta_fx2[i] = 0.0;
                      delta_fy2[i] = 0.0;
                      delta_fz2[i] = 0.0;                      
                    
                      delta_fx3[i] = 0.0;
                      delta_fy3[i] = 0.0;
                      delta_fz3[i] = 0.0;                       
                      
                      delta_fx4[i][nr] = 0.0;
                      delta_fy4[i][nr] = 0.0;
                      delta_fz4[i][nr] = 0.0;
                      
                      delta_fx5[i][nr] = 0.0;
                      delta_fy5[i][nr] = 0.0;
                      delta_fz5[i][nr] = 0.0;                      
                      
                  };
                };
              };
                   
            if(stepper == 1 && tau2_a > 0) {
                
              for(i=1;i<=natoms1;i++){
                
                  for(nr = 1; nr <= number_of_replicas ; nr ++) {   
                  
                  alpha_1_x[i][nr] = beta2;
                  alpha_1_y[i][nr] = beta2;
                  alpha_1_z[i][nr] = beta2;
                  
                  alpha_2_x[i][nr] = beta2;
                  alpha_2_y[i][nr] = beta2;
                  alpha_2_z[i][nr] = beta2;                  
                  
                  delta_alpha_1x[i][nr] = beta2*stepsize_alpha;
                  delta_alpha_1y[i][nr] = beta2*stepsize_alpha;
                  delta_alpha_1z[i][nr] = beta2*stepsize_alpha;  
                  
                  delta_alpha_2x[i][nr] = beta2*stepsize_alpha;
                  delta_alpha_2y[i][nr] = beta2*stepsize_alpha;
                  delta_alpha_2z[i][nr] = beta2*stepsize_alpha;                   
                  
                 };          
                  
              };
            };
        
if(tau_c > 0 && dih_rdf_sampling_on == 1) {
  
  double x_d1,y_d1,z_d1;
  double x_d2,y_d2,z_d2;
  double x_d3,y_d3,z_d3;
  double x_d4,y_d4,z_d4;
  double d_d1_x,d_d1_y,d_d1_z;
  double d_d2_x,d_d2_y,d_d2_z;
  double d_d3_x,d_d3_y,d_d3_z;
  double cross_1_x,cross_1_y,cross_1_z;
  double cross_2_x,cross_2_y,cross_2_z;
  double angle,pot_angle;
  double rgsq,fg,hg,fga,hgb,gaa,gbb;
  double dtfx,dtfy,dtfz;
  double dfgx,dfgy,dfgz;
  double dthx,dthy,dthz;
  double s_x2,s_y2,s_z2;
  double force_a1_x,force_a1_y,force_a1_z;
  double force_a2_x,force_a2_y,force_a2_z;
  double force_a3_x,force_a3_y,force_a3_z;
  double force_a4_x,force_a4_y,force_a4_z;
  double range;  
  double direction_x,direction_y,direction_z;
  double cross_3_x,cross_3_y,cross_3_z;
    
    for(n=1;n<=number_of_dihedrals;n++) {  
        
        x_d1 = state_global->x[at_id_d1[n]-1][0];
        y_d1 = state_global->x[at_id_d1[n]-1][1];
        z_d1 = state_global->x[at_id_d1[n]-1][2];        
        
        x_d2 = state_global->x[at_id_d2[n]-1][0];
        y_d2 = state_global->x[at_id_d2[n]-1][1];
        z_d2 = state_global->x[at_id_d2[n]-1][2];          
        
        x_d3 = state_global->x[at_id_d3[n]-1][0];
        y_d3 = state_global->x[at_id_d3[n]-1][1];
        z_d3 = state_global->x[at_id_d3[n]-1][2];        
        
        x_d4 = state_global->x[at_id_d4[n]-1][0];
        y_d4 = state_global->x[at_id_d4[n]-1][1];
        z_d4 = state_global->x[at_id_d4[n]-1][2];          
      
        d_d1_x = x_d1 - x_d2;
        d_d1_y = y_d1 - y_d2;
        d_d1_z = z_d1 - z_d2;
        
        d_d2_x = x_d2 - x_d3;
        d_d2_y = y_d2 - y_d3;
        d_d2_z = z_d2 - z_d3;

        d_d3_x = x_d3 - x_d4;
        d_d3_y = y_d3 - y_d4;
        d_d3_z = z_d3 - z_d4;

        cross_1_x = d_d1_y*d_d2_z - d_d1_z*d_d2_y;
        cross_1_y = d_d1_z*d_d2_x - d_d1_x*d_d2_z;
        cross_1_z = d_d1_x*d_d2_y - d_d1_y*d_d2_x;
        
        cross_2_x = d_d2_y*d_d3_z - d_d2_z*d_d3_y;
        cross_2_y = d_d2_z*d_d3_x - d_d2_x*d_d3_z;
        cross_2_z = d_d2_x*d_d3_y - d_d2_y*d_d3_x;        
        
        d_t  = sqrt(pow(cross_1_x,2) + pow(cross_1_y,2) + pow(cross_1_z,2));
        d_t2 = sqrt(pow(cross_2_x,2) + pow(cross_2_y,2) + pow(cross_2_z,2));
        
        cross_3_x = cross_1_y*cross_2_z - cross_1_z*cross_2_y;
        cross_3_y = cross_1_z*cross_2_x - cross_1_x*cross_2_z;
        cross_3_z = cross_1_x*cross_2_y - cross_1_y*cross_2_x;
        
        direction_x = cross_3_x*d_d2_x;
        direction_y = cross_3_y*d_d2_y;
        direction_z = cross_3_z*d_d2_z;         
        
        angle = 180.0/PI*acos((cross_1_x*cross_2_x+cross_1_y*cross_2_y+cross_1_z*cross_2_z)/(d_t*d_t2));
    
        if(direction_x > 0.0 || direction_y > 0.0 || direction_z > 0.0) {
            
          angle = -angle;
            
        };          
        
        if(angle < 0.0) angle = angle + 360.0;
        
        range = 0.0;
        pot_angle = 0.0;
        
        for(k=1;k<=360;k++) {
          
            if(range-0.5 <= angle && range+0.5 > angle) {
              
                pot_angle = dih_en[n][k] - dih_en[n][k+1];

            };  
            
            range += 1.0;
        };
        
        rgsq = sqrt(d_d2_x*d_d2_x+ d_d2_y*d_d2_y + d_d2_z*d_d2_z);
        fg = d_d1_x*d_d2_x + d_d1_y*d_d2_y + d_d1_z*d_d2_z;
        hg = d_d3_x*d_d2_x + d_d3_y*d_d2_y + d_d3_z*d_d2_z;
        
     if(rgsq > 0.0 && d_t > 0.0 && d_t2 > 0.0 && pot_angle != 0.0) {   
        
        fga = fg/(d_t*rgsq);
        hgb = hg/(d_t2*rgsq);        
        gaa = - 1.0/d_t*rgsq;
        gbb = 1.0/d_t2*rgsq;
        
        dtfx = gaa*cross_1_x;
        dtfy = gaa*cross_1_y;
        dtfz = gaa*cross_1_z;
        
        dfgx = fga*cross_1_x - hgb*cross_2_x;
        dfgy = fga*cross_1_y - hgb*cross_2_y;
        dfgz = fga*cross_1_z - hgb*cross_2_z;
        
        dthx = gbb*cross_2_x;
        dthy = gbb*cross_2_y;
        dthz = gbb*cross_2_z;

        s_x2 = dfgx*pot_angle;
        s_y2 = dfgy*pot_angle;
        s_z2 = dfgz*pot_angle;
        
        force_a1_x = pot_angle*dtfx;
        force_a1_y = pot_angle*dtfy;
        force_a1_z = pot_angle*dtfz;
        
        force_a2_x = s_x2 - force_a1_x;
        force_a2_y = s_y2 - force_a1_y;
        force_a2_z = s_z2 - force_a1_z;
        
        force_a4_x = pot_angle*dthx;
        force_a4_y = pot_angle*dthy;
        force_a4_z = pot_angle*dthz;

        force_a3_x = -s_x2 - force_a4_x;
        force_a3_y = -s_y2 - force_a4_y;
        force_a3_z = -s_z2 - force_a4_z;
        
        h5 = sqrt(pow(force_a1_x,2)+pow(force_a1_y,2)+pow(force_a1_z,2)+pow(force_a2_x,2)+pow(force_a2_y,2)+pow(force_a2_z,2)
                 +pow(force_a3_x,2)+pow(force_a3_y,2)+pow(force_a3_z,2)+pow(force_a4_x,2)+pow(force_a4_y,2)+pow(force_a4_z,2)); 
        h4 = sqrt(pow(f_global[at_id_d1[n]-1][0],2)+pow(f_global[at_id_d1[n]-1][1],2)+pow(f_global[at_id_d1[n]-1][2],2)
                 +pow(f_global[at_id_d2[n]-1][0],2)+pow(f_global[at_id_d2[n]-1][1],2)+pow(f_global[at_id_d2[n]-1][2],2)
                 +pow(f_global[at_id_d3[n]-1][0],2)+pow(f_global[at_id_d3[n]-1][1],2)+pow(f_global[at_id_d3[n]-1][2],2)
                 +pow(f_global[at_id_d4[n]-1][0],2)+pow(f_global[at_id_d4[n]-1][1],2)+pow(f_global[at_id_d4[n]-1][2],2));

        ran = (rand()%100/100.0);
                    
        if(h5 > 0.0) {  
                     
          f_global[at_id_d1[n]-1][0] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d1[n]-1][0] + h4/h5*beta3*epsilon*ran*force_a1_x;
          f_global[at_id_d1[n]-1][1] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d1[n]-1][1] + h4/h5*beta3*epsilon*ran*force_a1_y;
          f_global[at_id_d1[n]-1][2] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d1[n]-1][2] + h4/h5*beta3*epsilon*ran*force_a1_z;         
     
            
        };
          
    //    h5 = 4.0*sqrt(pow(force_a2_x,2)+pow(force_a2_y,2)+pow(force_a2_z,2));      
    //    h4 = sqrt(pow(f_global[at_id_d2[n]-1][0],2)+pow(f_global[at_id_d2[n]-1][1],2)+pow(f_global[at_id_d2[n]-1][2],2));

        ran = (rand()%100/100.0);
                    
        if(h5 > 0.0) {  
                     
          f_global[at_id_d2[n]-1][0] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d2[n]-1][0] + h4/h5*beta3*epsilon*ran*force_a2_x;
          f_global[at_id_d2[n]-1][1] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d2[n]-1][1] + h4/h5*beta3*epsilon*ran*force_a2_y;
          f_global[at_id_d2[n]-1][2] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d2[n]-1][2] + h4/h5*beta3*epsilon*ran*force_a2_z;         
     
        };

     //   h5 = 4.0*sqrt(pow(force_a3_x,2)+pow(force_a3_y,2)+pow(force_a3_z,2));      
     //   h4 = sqrt(pow(f_global[at_id_d1[n]-1][0],2)+pow(f_global[at_id_d1[n]-1][1],2)+pow(f_global[at_id_d1[n]-1][2],2));

        ran = (rand()%100/100.0);
                    
        if(h5 > 0.0) {  
                     
          f_global[at_id_d3[n]-1][0] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d3[n]-1][0] + h4/h5*beta3*epsilon*ran*force_a3_x;
          f_global[at_id_d3[n]-1][1] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d3[n]-1][1] + h4/h5*beta3*epsilon*ran*force_a3_y;
          f_global[at_id_d3[n]-1][2] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d3[n]-1][2] + h4/h5*beta3*epsilon*ran*force_a3_z;         
     
        };

     //   h5 = 4.0*sqrt(pow(force_a4_x,2)+pow(force_a4_y,2)+pow(force_a4_z,2));        
     //   h4 = sqrt(pow(f_global[at_id_d1[n]-1][0],2)+pow(f_global[at_id_d1[n]-1][1],2)+pow(f_global[at_id_d1[n]-1][2],2));

        ran = (rand()%100/100.0);
                    
        if(h5 > 0.0) {  
                     
          f_global[at_id_d4[n]-1][0] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d4[n]-1][0] + h4/h5*beta3*epsilon*ran*force_a4_x;
          f_global[at_id_d4[n]-1][1] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d4[n]-1][1] + h4/h5*beta3*epsilon*ran*force_a4_y;
          f_global[at_id_d4[n]-1][2] = 1.0/(1.0 + 1.0/h5*beta3*epsilon_md*ran)*f_global[at_id_d4[n]-1][2] + h4/h5*beta3*epsilon*ran*force_a4_z;         
     
        };
        
     /*   printf("%lg\t%lg\t%lg\n",force_a1_x,force_a1_y,force_a1_z);
        printf("%lg\t%lg\t%lg\n",force_a2_x,force_a2_y,force_a2_z);
        printf("%lg\t%lg\t%lg\n",force_a3_x,force_a3_y,force_a3_z);  
        printf("%lg\t%lg\t%lg\n",force_a4_x,force_a4_y,force_a4_z);  */      
        
     };
    };
};
            
if(tau_c > 0) {  
            
       if(stepper%tau_c == 0 && stepper > tau2_a) {
           
           for(i=1;i<=natoms2;i++) {

             a_x_b[i] = 0.0;
             a_y_b[i] = 0.0;
             a_z_b[i] = 0.0;

             sum_exp = 0.0;
             exp_of_funct[i] = 0.0;
             sum_of_all_derivatives_x = 0.0;
             sum_of_all_derivatives_y = 0.0;
             sum_of_all_derivatives_z = 0.0;                          
             
             for(k=1;k<=natoms2;k++) {

                 dx = (state_global->x[at_id_b[i]][0]-state_global->x[at_id_b[k]][0]);
                 dy = (state_global->x[at_id_b[i]][1]-state_global->x[at_id_b[k]][1]);
                 dz = (state_global->x[at_id_b[i]][2]-state_global->x[at_id_b[k]][2]);
                 
                  if(dx > state_global->box[0][0]/2.0) dx = dx - state_global->box[0][0]/2.0;
                  if(dy > state_global->box[1][1]/2.0) dy = dy - state_global->box[1][1]/2.0;
                  if(dz > state_global->box[2][2]/2.0) dz = dz - state_global->box[2][2]/2.0;

                  if(dx < -state_global->box[0][0]/2.0) dx = dx + state_global->box[0][0]/2.0;
                  if(dy < -state_global->box[1][1]/2.0) dy = dy + state_global->box[1][1]/2.0;
                  if(dz < -state_global->box[2][2]/2.0) dz = dz + state_global->box[2][2]/2.0;

                 d_t_a[i] = sqrt(pow(dx,2)+pow(dy,2)+pow(dz,2));
                 
                  radius = 0.0;
                  
               if(d_t_a[i] > 0.0) {    

                 for(n=1;n<=number_of_lines-1;n++){
                     
                    radius = radius + range_dist/(double)number_of_lines; 
                    
                      if(radius-range_dist/(double)number_of_lines < d_t_a[i] && radius+range_dist/(double)number_of_lines > d_t_a[i]) {
                        
                          a_x_b[i] += dx/d_t_a[i]*(fee_en[i][k][n] - fee_en[i][k][n+1])/range_dist/(double)number_of_lines;
                          a_y_b[i] += dy/d_t_a[i]*(fee_en[i][k][n] - fee_en[i][k][n+1])/range_dist/(double)number_of_lines;                        
                          a_z_b[i] += dz/d_t_a[i]*(fee_en[i][k][n] - fee_en[i][k][n+1])/range_dist/(double)number_of_lines;                        
                          
                       };
                      };
                     }; 
                    }; 
                   };
      
          for(i=1;i<=natoms2;i++) {
           
             grad_x[i] =  a_x_b[i];
             grad_y[i] =  a_y_b[i];
             grad_z[i] =  a_z_b[i];           
           
           };            
          };    
    
    
      if(stepper > tau2_a) {    
       for(i=1;i<=natoms2;i++){
               
                     h5 = sqrt(pow(grad_x[i],2)+pow(grad_y[i],2)+pow(grad_z[i],2));
                     h4 = sqrt(pow(f_global[i][0],2)+pow(f_global[i][1],2)+pow(f_global[i][2],2));

                     ran = (rand()%100/100.0);
                    
                   if(h5 > 0.0) {  
                     
                     f_global[at_id_b[i]][0] = 1.0/(1.0 + beta3*epsilon_md*ran)*f_global[at_id_b[i]][0] + beta3*h4/h5*epsilon*ran*grad_x[i];//beta3*(h4/h5)*grad_x[i];
                     f_global[at_id_b[i]][1] = 1.0/(1.0 + beta3*epsilon_md*ran)*f_global[at_id_b[i]][1] + beta3*h4/h5*epsilon*ran*grad_y[i];//beta3*(h4/h5)*grad_y[i];
                     f_global[at_id_b[i]][2] = 1.0/(1.0 + beta3*epsilon_md*ran)*f_global[at_id_b[i]][2] + beta3*h4/h5*epsilon*ran*grad_z[i];//beta3*(h4/h5)*grad_z[i];                 
                 
                   };
                 
               };
              };           
};  
      
if(tau2_a > 0) {

            if(stepper22%2 == 0) {

             for(i=1;i<=natoms1;i++){
                  
                    f_x1[i] = f_global[at_id[i]][0]*ir->delta_t;
                    f_y1[i] = f_global[at_id[i]][1]*ir->delta_t;
                    f_z1[i] = f_global[at_id[i]][2]*ir->delta_t;                    
                 
                    x0[i] = state_global->x[at_id[i]][0];
		            y0[i] = state_global->x[at_id[i]][1];
                    z0[i] = state_global->x[at_id[i]][2];
                    
                  };
                  
              } else {
              
                  for(i=1;i<=natoms1;i++){
                  
                    f_x2[i] = f_global[at_id[i]][0]*ir->delta_t;
                    f_y2[i] = f_global[at_id[i]][1]*ir->delta_t;
                    f_z2[i] = f_global[at_id[i]][2]*ir->delta_t;                    
                 
                    x1[i] = state_global->x[at_id[i]][0];
		            y1[i] = state_global->x[at_id[i]][1];
                    z1[i] = state_global->x[at_id[i]][2];
							    

                  };                    
                };        
                
            if(stepper22%2 == 0) {
            
              for(i=1;i<=natoms1;i++){  
                
               df_x1[i] = f_x1[i] + (f_x1[i]-f_x2[i])/ir->delta_t;
               df_y1[i] = f_y1[i] + (f_y1[i]-f_y2[i])/ir->delta_t;              
               df_z1[i] = f_z1[i] + (f_z1[i]-f_z2[i])/ir->delta_t; 
           
               for(nr=1;nr<=number_of_replicas;nr++){   
               
               if(x0[i] - x1[i] < state_global->box[0][0]/2.0) d_x1[i][nr] = (x0[i] - x1[i]);
               if(y0[i] - y1[i] < state_global->box[1][1]/2.0) d_y1[i][nr] = (y0[i] - y1[i]);
               if(z0[i] - z1[i] < state_global->box[2][2]/2.0) d_z1[i][nr] = (z0[i] - z1[i]);                
           
               if(x0[i] - x1[i] > state_global->box[0][0]/2.0) d_x1[i][nr] = ((x0[i] - x1[i])-state_global->box[0][0]/2.0);
               if(y0[i] - y1[i] > state_global->box[1][1]/2.0) d_y1[i][nr] = ((y0[i] - y1[i])-state_global->box[1][1]/2.0);
               if(z0[i] - z1[i] > state_global->box[2][2]/2.0) d_z1[i][nr] = ((z0[i] - z1[i])-state_global->box[2][2]/2.0);               
               
               if(x0[i] - x1[i] < -state_global->box[0][0]/2.0) d_x1[i][nr] = ((x0[i] - x1[i])+state_global->box[0][0]/2.0);
               if(y0[i] - y1[i] < -state_global->box[1][1]/2.0) d_y1[i][nr] = ((y0[i] - y1[i])+state_global->box[1][1]/2.0);
               if(z0[i] - z1[i] < -state_global->box[2][2]/2.0) d_z1[i][nr] = ((z0[i] - z1[i])+state_global->box[2][2]/2.0);                
               
       //       printf("%lg\t%lg\t%lg\n",d_x1[i],d_y1[i],d_z1[i]);              
               
                };
               };
              };
            
            if(stepper22%2 != 0) {
            
              for(i=1;i<=natoms1;i++){  
                
              df_x1[i] = f_x2[i] + (f_x2[i]-f_x1[i])/ir->delta_t;
              df_y1[i] = f_y2[i] + (f_y2[i]-f_y1[i])/ir->delta_t;              
              df_z1[i] = f_z2[i] + (f_z2[i]-f_z1[i])/ir->delta_t; 
             
               for(nr=1;nr<=number_of_replicas;nr++){

               if(x1[i] - x0[i] < state_global->box[0][0]/2.0) d_x1[i][nr] = (x1[i] - x0[i]);
               if(y1[i] - y0[i] < state_global->box[1][1]/2.0) d_y1[i][nr] = (y1[i] - y0[i]);
               if(z1[i] - z0[i] < state_global->box[2][2]/2.0) d_z1[i][nr] = (z1[i] - z0[i]);                
           
               if(x1[i] - x0[i] > state_global->box[0][0]/2.0) d_x1[i][nr] = ((x1[i] - x0[i])-state_global->box[0][0]/2.0);
               if(y1[i] - y0[i] > state_global->box[1][1]/2.0) d_y1[i][nr] = ((y1[i] - y0[i])-state_global->box[1][1]/2.0);
               if(z1[i] - z0[i] > state_global->box[2][2]/2.0) d_z1[i][nr] = ((z1[i] - z0[i])-state_global->box[2][2]/2.0);               
               
               if(x1[i] - x0[i] < -state_global->box[0][0]/2.0) d_x1[i][nr] = ((x1[i] - x0[i])+state_global->box[0][0]/2.0);
               if(y1[i] - y0[i] < -state_global->box[1][1]/2.0) d_y1[i][nr] = ((y1[i] - y0[i])+state_global->box[1][1]/2.0);
               if(z1[i] - z0[i] < -state_global->box[2][2]/2.0) d_z1[i][nr] = ((z1[i] - z0[i])+state_global->box[2][2]/2.0);                
               
                 };
	           };
            };
               
    if(stepper22%2 == 0){
        
           for(i=1;i<=natoms1;i++){               
	  //     d_t = sqrt(pow(d_x1[i],2)+pow(d_y1[i],2)+pow(d_z1[i],2));   
           
              delta_fx1[i] = 0.0;
              delta_fy1[i] = 0.0;
              delta_fz1[i] = 0.0;               
               
           for(nr=1;nr<=number_of_replicas;nr++){  
                 
            if( (rand()%100)/100.0 <= probability_input2) {                  
               
                delta_fx1[i] +=  (df_x1[i] *d_x1[i][nr]);
                delta_fy1[i] +=  (df_y1[i] *d_y1[i][nr]);  
                delta_fz1[i] +=  (df_z1[i] *d_z1[i][nr]);
             
                };
               };
                
                delta_fx3[i]  = (delta_fx1[i] - delta_fx2[i])/ir->delta_t;
                delta_fy3[i]  = (delta_fy1[i] - delta_fy2[i])/ir->delta_t;
                delta_fz3[i]  = (delta_fz1[i] - delta_fz2[i])/ir->delta_t;               
            
           };                 
           
           } else {
               
           for(i=1;i<=natoms1;i++){               
           
              delta_fx2[i] = 0.0;
              delta_fy2[i] = 0.0;
              delta_fz2[i] = 0.0;
               
             for(nr=1;nr<=number_of_replicas;nr++){   
              if( (rand()%100)/100.0 <= probability_input2) {   
                  
                delta_fx2[i] +=  (df_x1[i] *d_x1[i][nr]);
                delta_fy2[i] +=  (df_y1[i] *d_y1[i][nr]);  
                delta_fz2[i] +=  (df_z1[i] *d_z1[i][nr]);                               
             
                 };
               };
             };               
           }; 
           
            if(stepper22%2 == 0) {
            
              for(i=1;i<=natoms1;i++){  
                
               df_x1[i] = f_x1[i] + (f_x1[i]-f_x2[i])/ir->delta_t;
               df_y1[i] = f_y1[i] + (f_y1[i]-f_y2[i])/ir->delta_t;              
               df_z1[i] = f_z1[i] + (f_z1[i]-f_z2[i])/ir->delta_t; 
           
               for(nr=1;nr<=number_of_replicas;nr++){   
               
               if(x0[i] - x1[i] < state_global->box[0][0]/2.0) d_x1[i][nr] = (x0[i] - x1[i]);
               if(y0[i] - y1[i] < state_global->box[1][1]/2.0) d_y1[i][nr] = (y0[i] - y1[i]);
               if(z0[i] - z1[i] < state_global->box[2][2]/2.0) d_z1[i][nr] = (z0[i] - z1[i]);                
           
               if(x0[i] - x1[i] > state_global->box[0][0]/2.0) d_x1[i][nr] = ((x0[i] - x1[i])-state_global->box[0][0]/2.0);
               if(y0[i] - y1[i] > state_global->box[1][1]/2.0) d_y1[i][nr] = ((y0[i] - y1[i])-state_global->box[1][1]/2.0);
               if(z0[i] - z1[i] > state_global->box[2][2]/2.0) d_z1[i][nr] = ((z0[i] - z1[i])-state_global->box[2][2]/2.0);               
               
               if(x0[i] - x1[i] < -state_global->box[0][0]/2.0) d_x1[i][nr] = ((x0[i] - x1[i])+state_global->box[0][0]/2.0);
               if(y0[i] - y1[i] < -state_global->box[1][1]/2.0) d_y1[i][nr] = ((y0[i] - y1[i])+state_global->box[1][1]/2.0);
               if(z0[i] - z1[i] < -state_global->box[2][2]/2.0) d_z1[i][nr] = ((z0[i] - z1[i])+state_global->box[2][2]/2.0);                
               
       //       printf("%lg\t%lg\t%lg\n",d_x1[i],d_y1[i],d_z1[i]);              
               
                };
               };
              };
            
            if(stepper22%2 != 0) {
            
              for(i=1;i<=natoms1;i++){  
                
              df_x1[i] = f_x2[i] + (f_x2[i]-f_x1[i])/ir->delta_t;
              df_y1[i] = f_y2[i] + (f_y2[i]-f_y1[i])/ir->delta_t;              
              df_z1[i] = f_z2[i] + (f_z2[i]-f_z1[i])/ir->delta_t; 
             
               for(nr=1;nr<=number_of_replicas;nr++){

               if(x1[i] - x0[i] < state_global->box[0][0]/2.0) d_x1[i][nr] = (x1[i] - x0[i]);
               if(y1[i] - y0[i] < state_global->box[1][1]/2.0) d_y1[i][nr] = (y1[i] - y0[i]);
               if(z1[i] - z0[i] < state_global->box[2][2]/2.0) d_z1[i][nr] = (z1[i] - z0[i]);                
           
               if(x1[i] - x0[i] > state_global->box[0][0]/2.0) d_x1[i][nr] = ((x1[i] - x0[i])-state_global->box[0][0]/2.0);
               if(y1[i] - y0[i] > state_global->box[1][1]/2.0) d_y1[i][nr] = ((y1[i] - y0[i])-state_global->box[1][1]/2.0);
               if(z1[i] - z0[i] > state_global->box[2][2]/2.0) d_z1[i][nr] = ((z1[i] - z0[i])-state_global->box[2][2]/2.0);               
               
               if(x1[i] - x0[i] < -state_global->box[0][0]/2.0) d_x1[i][nr] = ((x1[i] - x0[i])+state_global->box[0][0]/2.0);
               if(y1[i] - y0[i] < -state_global->box[1][1]/2.0) d_y1[i][nr] = ((y1[i] - y0[i])+state_global->box[1][1]/2.0);
               if(z1[i] - z0[i] < -state_global->box[2][2]/2.0) d_z1[i][nr] = ((z1[i] - z0[i])+state_global->box[2][2]/2.0);                
               
                 };
	           };
            };
               
    if(stepper22%2 == 0){
        
           for(i=1;i<=natoms1;i++){               
	  //     d_t = sqrt(pow(d_x1[i],2)+pow(d_y1[i],2)+pow(d_z1[i],2));   
           
              delta_fx1[i] = 0.0;
              delta_fy1[i] = 0.0;
              delta_fz1[i] = 0.0;               
               
           for(nr=1;nr<=number_of_replicas;nr++){  
                 
            if( (rand()%100)/100.0 <= probability_input2) {                  
               
                delta_fx1[i] +=  (df_x1[i] *d_x1[i][nr]);
                delta_fy1[i] +=  (df_y1[i] *d_y1[i][nr]);  
                delta_fz1[i] +=  (df_z1[i] *d_z1[i][nr]);
             
                };
               };
                
                delta_fx3[i]  = (delta_fx1[i] - delta_fx2[i])/ir->delta_t;
                delta_fy3[i]  = (delta_fy1[i] - delta_fy2[i])/ir->delta_t;
                delta_fz3[i]  = (delta_fz1[i] - delta_fz2[i])/ir->delta_t;               
            
           };                 
           
           } else {
               
           for(i=1;i<=natoms1;i++){               
           
              delta_fx2[i] = 0.0;
              delta_fy2[i] = 0.0;
              delta_fz2[i] = 0.0;
               
             for(nr=1;nr<=number_of_replicas;nr++){   
              if( (rand()%100)/100.0 <= probability_input2) {   
                  
                delta_fx2[i] +=  (df_x1[i] *d_x1[i][nr]);
                delta_fy2[i] +=  (df_y1[i] *d_y1[i][nr]);  
                delta_fz2[i] +=  (df_z1[i] *d_z1[i][nr]);                               
             
                 };
               };
             };               
           };      
   
        for(nr=1;nr<=number_of_replicas;nr++){   
           
         count2[nr] ++;    
         
       //  replica_fraction = (double)(nr); //*(double)natoms1/(double)number_of_replicas;
               
         replica_fraction_integer = 1; //+ (int)roundf(replica_fraction); 
      
          if(count2[nr]%2 == 0) {
             
              for(i=1;i<=natoms1;i++){
           
               if(stepper%tau[i][nr] == 0){                  
                  
                if(i%replica_fraction_integer == 0){  

                  delta_fx4[i][nr] = delta_fx3[i]/(double)number_of_replicas;
                  delta_fy4[i][nr] = delta_fy3[i]/(double)number_of_replicas;                  
                  delta_fz4[i][nr] = delta_fz3[i]/(double)number_of_replicas;                  
                  
                   }; 
                  };
                };  
               
           } else {
           
              for(i=1;i<=natoms1;i++){
            
               if(stepper%tau[i][nr] == 0){                  
                  
                 if(i%replica_fraction_integer == 0) { 
                  
                  delta_fx5[i][nr] = (delta_fx3[i]-delta_fx4[i][nr])/(double)number_of_replicas;
                  delta_fy5[i][nr] = (delta_fy3[i]-delta_fy4[i][nr])/(double)number_of_replicas;                  
                  delta_fz5[i][nr] = (delta_fz3[i]-delta_fz4[i][nr])/(double)number_of_replicas;                  
              
                 };
                };                 
               };
              };
             };   
       
         sum_exp = 0.0;
         sum_of_all_derivatives_x5 = 0.0;
         sum_of_all_derivatives_y5 = 0.0;
         sum_of_all_derivatives_z5 = 0.0;         
         sum_of_all_derivatives_x3 = 0.0;
         sum_of_all_derivatives_y3 = 0.0;
         sum_of_all_derivatives_z3 = 0.0;           
         
              
         for(i=1;i<=natoms1;i++) 
           {
               
            sum_exp += exp(-sqrt(pow(f_global[at_id[i]][0],2) + pow(f_global[at_id[i]][1],2)
                             + pow(f_global[at_id[i]][2],2))/(8.314*enerd->term[F_TEMP]));
            exp_of_funct2 = exp(-sqrt(pow(f_global[at_id[i]][0],2) + pow(f_global[at_id[i]][1],2)
                             + pow(f_global[at_id[i]][2],2))/(8.314*enerd->term[F_TEMP]));               
               
              sum_of_all_derivatives_x3 += (delta_fx3[i])*exp_of_funct2; 
              sum_of_all_derivatives_y3 += (delta_fy3[i])*exp_of_funct2;
              sum_of_all_derivatives_z3 += (delta_fz3[i])*exp_of_funct2;  
              
               
              sum_of_all_derivatives_x5 += (delta_fx5[i][nr])*exp_of_funct2; 
              sum_of_all_derivatives_y5 += (delta_fy5[i][nr])*exp_of_funct2;
              sum_of_all_derivatives_z5 += (delta_fz5[i][nr])*exp_of_funct2;
              
           }; 
      
          for(i=1;i<=natoms1;i++) {
              
                delta_fx3[i] = delta_fx3[i] - 1.0/sum_exp*sum_of_all_derivatives_x3;
                delta_fy3[i] = delta_fy3[i] - 1.0/sum_exp*sum_of_all_derivatives_y3;
                delta_fz3[i] = delta_fz3[i] - 1.0/sum_exp*sum_of_all_derivatives_z3;                
              
            for(nr=1;nr<=number_of_replicas;nr++){           
                
                delta_fx5[i][nr] = delta_fx5[i][nr] - 1.0/sum_exp*sum_of_all_derivatives_x5;
                delta_fy5[i][nr] = delta_fy5[i][nr] - 1.0/sum_exp*sum_of_all_derivatives_y5;
                delta_fz5[i][nr] = delta_fz5[i][nr] - 1.0/sum_exp*sum_of_all_derivatives_z5;                  
           
             };
            }; 
            
  flag_diagonalize = 0;         
           
  for(n=1;n<=natoms1;n++) {
   for(nr=1;nr<=number_of_replicas;nr++) {  
       if(stepper%tau[n][nr] == 0 && density_boolean == 1) {
           
           flag_diagonalize = nr;
           
              };
            }; 
          }; 
     
     if(flag_diagonalize > 0) {
     
            k = 1;
            l = 1;
            
              for(i=1;i<=(natoms1)*(natoms1);i++) {                            
                      
                         tmp_x[l]   = delta_fx4[k][flag_diagonalize];  
                         tmp_x[l+1] = delta_fy4[k][flag_diagonalize];
                         tmp_x[l+2] = delta_fz4[k][flag_diagonalize];
                         
                         if(k == natoms1) k = 0;
                         l = l + 3;
                         
                         k ++; 
                     };
                     
               info = eigensolver(tmp_x,natoms1*3,1,natoms1*3,eigval,mat);
                    
            if(info == 0) {   
               
               var = 1E+10;
               l = 1;               
               
               for(i=1;i<=natoms1;i++) {
   
                  d_t = sqrt(pow(eigval[l],2) + pow(eigval[l+1],2) + pow(eigval[l+2],2)); 
                   
                  if(d_t < var) {
                     
                      selector = i;
                      var      = d_t;
                      
                   };    
                   
                   l = l + 3;
                   
               }; 
                
               l = 1; 
               
               for(i=1;i<=natoms1;i++) {
                     
                        delta_fx4[i][flag_diagonalize] = mat[l+((selector-1)*3*natoms1*3)]; 
                        delta_fy4[i][flag_diagonalize] = mat[l+1+((selector-1)*3*natoms1*3)];
                        delta_fz4[i][flag_diagonalize] = mat[l+2+((selector-1)*3*natoms1*3)];                        
                        
                        l = l + 3;
                        
                    };
                };              
          
            k = 1;
            l = 1;
            
              for(i=1;i<=(natoms1)*(natoms1);i++) {                            
                      
                         tmp_x[l]   = delta_fx5[k][flag_diagonalize];  
                         tmp_x[l+1] = delta_fy5[k][flag_diagonalize];
                         tmp_x[l+2] = delta_fz5[k][flag_diagonalize];
                         
                         if(k == natoms1) k = 0;
                         l = l + 3;
                         
                         k ++; 
                     };
                     
               info = eigensolver(tmp_x,natoms1*3,1,natoms1*3,eigval,mat);
                    
            if(info == 0) {   
               
               var = 1E+10;
               l = 1;               
               
               for(i=1;i<=natoms1;i++) {
   
                  d_t = sqrt(pow(eigval[l],2) + pow(eigval[l+1],2) + pow(eigval[l+2],2)); 
                   
                  if(d_t < var) {
                     
                      selector = i;
                      var      = d_t;
                      
                   };    
                   
                   l = l + 3;
                   
               };                
                
               l = 1; 
                
               for(i=1;i<=natoms1;i++) {
                     
                        delta_fx5[i][flag_diagonalize] = mat[l+((selector-1)*3*natoms1*3)]; 
                        delta_fy5[i][flag_diagonalize] = mat[l+1+((selector-1)*3*natoms1*3)];
                        delta_fz5[i][flag_diagonalize] = mat[l+2+((selector-1)*3*natoms1*3)];                        
                        
                        l = l + 3;
                        
                    };
                };     
                        
             };                         
            
      if(stepper > tau2_a){  
          
        for(i=1;i<=natoms1;i++) {
          
            h1[i] = 0.0;
            h2[i] = 0.0;
            h3[i] = 0.0;
        };  
        
            
        for(i=1;i<=natoms1;i++) {            
          
            beta_sum_x[i] = 0.0;
            beta_sum_y[i] = 0.0;
            beta_sum_z[i] = 0.0;
        };         

        
        for(nr=1;nr<=number_of_replicas;nr++){        
        
           for(i=1;i<=natoms1;i++){
              
             if( (rand()%100)/100.0 <= probability_input) {  
               
               diff_t2[i] = 0.0;
 
               if(delta_fx5[i][nr] != 0.0 || delta_fy5[i][nr] != 0.0 || delta_fz5[i][nr] != 0.0) {
               
                  r = random_number(&ran);
                  
                  diff_t2[i] = (diff_t - 2.0*diff_t*(double)ran);

                  if(diff_t2[i] < 0.0) diff_t2[i] = -diff_t2[i];              
               
                  delta_t2[i][nr] += (diff_t2[i] - diff_t_last[i][nr])/(double)(tau[i][nr]); 

                  if(stepper%tau[i][nr] == 0) delta_t2[i][nr] = 0.0;

                          h1[i]  += sqrt(pow(delta_fx5[i][nr]*diff_t2[i],2)+
                              pow(delta_fy5[i][nr]*diff_t2[i],2)+
                              pow(delta_fz5[i][nr]*diff_t2[i],2));

                          h2[i] += sqrt(pow(f_global[at_id[i]][0],2) +
                              pow(f_global[at_id[i]][1],2) +
                              pow(f_global[at_id[i]][2],2));       

                          h3[i] += sqrt(pow(delta_t2[i][nr]*delta_fx3[i],2)+
                              pow(delta_t2[i][nr]*delta_fy3[i],2)+
                              pow(delta_t2[i][nr]*delta_fz3[i],2));
                   
                  if(stepper%tau[i][nr] == 0 || stepper == 1) {
 
                   if(density_boolean == 0) {   
                      
                       random_number(&ran);         

                       epsilon2[i][nr] = epsilon*(1.0-ran);
                       epsilon7[i][nr] = epsilon_md*(1.0-ran);
 
                       };
                                            
                    };
                    
                 if(h1[i] > 0.0 && h3[i] > 0.0) {         
                     
                 if(restraint_sampling[i] == 0) {    
                     
                   if(density_boolean == 0) {    
                       
                    f_global[at_id[i]][0] += epsilon2[i][nr]*alpha_2_x[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fx5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fx3[i]);
                    f_global[at_id[i]][1] += epsilon2[i][nr]*alpha_2_y[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fy5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fy3[i]);
                    f_global[at_id[i]][2] += epsilon2[i][nr]*alpha_2_z[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fz5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fz3[i]);        

                    diff_t_last[i][nr] = diff_t2[i];
 //                   printf("%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",delta_fx[i]*diff_t2[i] + delta_t2[i]*delta_fx3[i],delta_fy[i]*diff_t2[i] + delta_t2[i]*delta_fy3[i],delta_fz[i]*diff_t2[i] + delta_t2[i]*delta_fz3[i],delta_t2,diff_t,diff_t2);
                    
                    beta_sum_x[i] += epsilon7[i][nr]*alpha_2_x[i][nr];
                    beta_sum_y[i] += epsilon7[i][nr]*alpha_2_y[i][nr];
                    beta_sum_z[i] += epsilon7[i][nr]*alpha_2_z[i][nr];

                   };
                   
                   if(density_boolean == 1 ) { 
                       
                    random_number(&ran);                       
                     
                    if(d_x1[i][nr] != 0.0) f_global[at_id[i]][0] += ran*epsilon/(1.0+sqrt(pow(d_x1[i][nr],2)))*alpha_2_x[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fx5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fx3[i]);
                    if(d_y1[i][nr] != 0.0) f_global[at_id[i]][1] += ran*epsilon/(1.0+sqrt(pow(d_y1[i][nr],2)))*alpha_2_y[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fy5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fy3[i]);
                    if(d_z1[i][nr] != 0.0) f_global[at_id[i]][2] += ran*epsilon/(1.0+sqrt(pow(d_z1[i][nr],2)))*alpha_2_z[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fz5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fz3[i]);        

                    diff_t_last[i][nr] = diff_t2[i];
 //                   printf("%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",delta_fx[i]*diff_t2[i] + delta_t2[i]*delta_fx3[i],delta_fy[i]*diff_t2[i] + delta_t2[i]*delta_fy3[i],delta_fz[i]*diff_t2[i] + delta_t2[i]*delta_fz3[i],delta_t2,diff_t,diff_t2);
                    
                    beta_sum_x[i] += ran*epsilon_md/(1.0+sqrt(pow(d_x1[i][nr],2)))*alpha_2_x[i][nr];
                    beta_sum_y[i] += ran*epsilon_md/(1.0+sqrt(pow(d_y1[i][nr],2)))*alpha_2_y[i][nr];
                    beta_sum_z[i] += ran*epsilon_md/(1.0+sqrt(pow(d_z1[i][nr],2)))*alpha_2_z[i][nr];

                   };                   
                    
                    if(nr==number_of_replicas && i==natoms1){
                    
                    for(n=1;n<=natoms1;n++){
          
                        f_global[at_id[n]][0] = 1.0/(1.0+beta_sum_x[i])*f_global[at_id[n]][0];
                        f_global[at_id[n]][1] = 1.0/(1.0+beta_sum_y[i])*f_global[at_id[n]][1];           
                        f_global[at_id[n]][2] = 1.0/(1.0+beta_sum_z[i])*f_global[at_id[n]][2];
           
                        };                   
                     }; 
                     
                    if(stepper%tau[i][nr] == 0) delta_t2[i][nr] = 0.0;
    
                       } else {
                         
                        for(k=1;k<=number_of_restraints;k++) {   
                         
                              cos_val = 1.0;
                            
                              if(i == at_id_r[k]) {  
                                  
                               if(density_boolean == 0) {             
                                  
                                  d_x = state_global->x[at_id_r[k]][0] - state_global->x[at_id_r2[k]][0];
                                  d_y = state_global->x[at_id_r[k]][1] - state_global->x[at_id_r2[k]][1];
                                  d_z = state_global->x[at_id_r[k]][2] - state_global->x[at_id_r2[k]][2];

                                  d_t = sqrt(pow(d_x,2)+pow(d_y,2)+pow(d_z,2));
                                  
                                  d_t2 = sqrt(pow(delta_fx[i],2)+pow(delta_fy[i],2)+pow(delta_fz[i],2));
                                  
                                  if(d_t*d_t2 > 0.0)cos_val = (d_x*delta_fx[i] + d_y*delta_fy[i] + d_z*delta_fz[i])/(d_t*d_t2); 
                              
                              if(distance_restraint[k]-d_t != 0.0) {    
                                  
                                f_global[at_id_r[k]][0] += (epsilon2[i][nr]*alpha_2_x[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fx5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fx3[i]))*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                f_global[at_id_r[k]][1] += (epsilon2[i][nr]*alpha_2_y[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fy5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fy3[i]))*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                f_global[at_id_r[k]][2] += (epsilon2[i][nr]*alpha_2_z[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fz5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fz3[i]))*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;        

                                };

                               };
                               
                               if(density_boolean == 1 ) {             
                                   
                                  random_number(&ran);
                                  
                                  d_x = state_global->x[at_id_r[k]][0] - state_global->x[at_id_r2[k]][0];
                                  d_y = state_global->x[at_id_r[k]][1] - state_global->x[at_id_r2[k]][1];
                                  d_z = state_global->x[at_id_r[k]][2] - state_global->x[at_id_r2[k]][2];

                                  d_t = sqrt(pow(d_x,2)+pow(d_y,2)+pow(d_z,2));
                                  
                                  
                                  d_t2 = sqrt(pow(delta_fx[i],2)+pow(delta_fy[i],2)+pow(delta_fz[i],2));
                                  
                                  if(d_t*d_t2 > 0.0)cos_val = (d_x*delta_fx[i] + d_y*delta_fy[i] + d_z*delta_fz[i])/(d_t*d_t2); 
                                  
                              if(distance_restraint[k]-d_t != 0.0) {                                  
                                  
                                ran = rand()%100/100.0;  
                                  
                                if(d_x1[i][nr] != 0.0) f_global[at_id_r[k]][0] += (ran*epsilon/(1.0+sqrt(pow(d_x1[i][nr],2)))*alpha_2_x[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fx5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fx3[i]))*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                if(d_y1[i][nr] != 0.0) f_global[at_id_r[k]][1] += (ran*epsilon/(1.0+sqrt(pow(d_y1[i][nr],2)))*alpha_2_y[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fy5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fy3[i]))*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                if(d_z1[i][nr] != 0.0) f_global[at_id_r[k]][2] += (ran*epsilon/(1.0+sqrt(pow(d_z1[i][nr],2)))*alpha_2_z[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fz5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fz3[i]))*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;        

                                 };
                                
                               };                               
                                
                              };
                              
                              if(i == at_id_r2[k]) {
                                  
                                if(density_boolean == 0) {  
                                 
                                  d_x = state_global->x[at_id_r[k]][0] - state_global->x[at_id_r2[k]][0];
                                  d_y = state_global->x[at_id_r[k]][1] - state_global->x[at_id_r2[k]][1];
                                  d_z = state_global->x[at_id_r[k]][2] - state_global->x[at_id_r2[k]][2];

                                  d_t = sqrt(pow(d_x,2)+pow(d_y,2)+pow(d_z,2));
                                  
                                  
                                  d_t2 = sqrt(pow(delta_fx[i],2)+pow(delta_fy[i],2)+pow(delta_fz[i],2));
                                  
                                  if(d_t*d_t2 > 0.0)cos_val = (d_x*delta_fx[i] + d_y*delta_fy[i] + d_z*delta_fz[i])/(d_t*d_t2);                                   

                               if(distance_restraint[k]-d_t != 0.0) {  
                                  
                                f_global[at_id_r2[k]][0] += (epsilon2[i][nr]*alpha_2_x[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fx5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fx3[i]))*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                f_global[at_id_r2[k]][1] += (epsilon2[i][nr]*alpha_2_y[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fy5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fy3[i]))*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                f_global[at_id_r2[k]][2] += (epsilon2[i][nr]*alpha_2_z[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fz5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fz3[i]))*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;                                    
                               
                                  };
                                
                                };
                                
                                 if(density_boolean == 1 ) {  
                                 
                                  random_number(&ran);                                     
                                     
                                  d_x = state_global->x[at_id_r[k]][0] - state_global->x[at_id_r2[k]][0];
                                  d_y = state_global->x[at_id_r[k]][1] - state_global->x[at_id_r2[k]][1];
                                  d_z = state_global->x[at_id_r[k]][2] - state_global->x[at_id_r2[k]][2];

                                  d_t = sqrt(pow(d_x,2)+pow(d_y,2)+pow(d_z,2));
                                  
                                  
                                  d_t2 = sqrt(pow(delta_fx[i],2)+pow(delta_fy[i],2)+pow(delta_fz[i],2));
                                  
                                  if(d_t*d_t2 > 0.0)cos_val = (d_x*delta_fx[i] + d_y*delta_fy[i] + d_z*delta_fz[i])/(d_t*d_t2);
                                  
                               if(distance_restraint[k]-d_t != 0.0) { 
                                   
                                ran = rand()%100/100.0;    
                                   
                                if(d_x1[i][nr] != 0.0) f_global[at_id_r2[k]][0] += (ran*epsilon/(sqrt(pow(d_x1[i][nr],2))+1.0)*alpha_2_x[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fx5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fx3[i]))*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                if(d_y1[i][nr] != 0.0) f_global[at_id_r2[k]][1] += (ran*epsilon/(sqrt(pow(d_y1[i][nr],2))+1.0)*alpha_2_y[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fy5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fy3[i]))*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                if(d_z1[i][nr] != 0.0) f_global[at_id_r2[k]][2] += (ran*epsilon/(sqrt(pow(d_z1[i][nr],2))+1.0)*alpha_2_z[i][nr]*h2[i]/(h1[i]+h3[i])*(delta_fz5[i][nr]*diff_t2[i] + delta_t2[i][nr]*delta_fz3[i]))*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;                                    
                                  
                                 };
                                
                                };                               
                                
                               };
                                
                               diff_t_last[i][nr] = diff_t2[i];
 //                         printf("%lg\t%lg\t%lg\t%lg\t%lg\t%lg\n",delta_fx[i]*diff_t2[i] + delta_t2[i]*delta_fx3[i],delta_fy[i]*diff_t2[i] + delta_t2[i]*delta_fy3[i],delta_fz[i]*diff_t2[i] + delta_t2[i]*delta_fz3[i],delta_t2,diff_t,diff_t2);
                    
                            if(density_boolean == 0) {   
                               
                               beta_sum_x[i] += epsilon7[i][nr]*alpha_2_x[i][nr];
                               beta_sum_y[i] += epsilon7[i][nr]*alpha_2_y[i][nr];
                               beta_sum_z[i] += epsilon7[i][nr]*alpha_2_z[i][nr];
                    
                            };
                            
                            if(density_boolean == 1 ) { 
                                
                               ran = rand()%100/100.0;  
                                
                               if(d_x1[i][nr] != 0.0)  beta_sum_x[i] += ran*epsilon_md/(1.0+sqrt(pow(d_x1[i][nr],2)))*alpha_2_x[i][nr];
                               if(d_y1[i][nr] != 0.0)  beta_sum_y[i] += ran*epsilon_md/(1.0+sqrt(pow(d_y1[i][nr],2)))*alpha_2_y[i][nr];
                               if(d_z1[i][nr] != 0.0)  beta_sum_z[i] += ran*epsilon_md/(1.0+sqrt(pow(d_z1[i][nr],2)))*alpha_2_z[i][nr];
                    
                            };                            
                               
                              if(nr==number_of_replicas && i==natoms1){
                    
                               for(n=1;n<=natoms1;n++){
          
                                 f_global[at_id[n]][0] = 1.0/(1.0+beta_sum_x[i])*f_global[at_id[n]][0];
                                 f_global[at_id[n]][1] = 1.0/(1.0+beta_sum_y[i])*f_global[at_id[n]][1];           
                                 f_global[at_id[n]][2] = 1.0/(1.0+beta_sum_z[i])*f_global[at_id[n]][2];
           
                                                      
                                };                                                                    
                               };
                              }; 
                             };      
                      };
                     };
                    };  	  
                   };
                  };
                 };           
     
        
     if(stepper == 1) d_max = state_global->box[0][0]*5.0;
     if(stepper == 1) {    
         
		for(i=1;i<=natoms1;i++){
		  
		  r_g[i] = r_g1;	  
		  
		}; 
       };
       
   if(stepper == 1) {     
     for(i=1;i<=natoms1;i++){    
       for(nr=1;nr<=number_of_replicas;nr++){   
         //|| stepper%((tau2[i][nr]))==0) {	
            
                   for(k=1;k<=100;k++) {

                     grad_meta_x[i][k] = 0.0;
                     grad_meta_y[i][k] = 0.0;
                     grad_meta_z[i][k] = 0.0;                    

                     }; 
                     
                     df_x_s[i][nr] = 0.0;
                     df_y_s[i][nr] = 0.0;
                     df_z_s[i][nr] = 0.0;
                     
                     d_x1_s[i][nr] = 0.0;
                     d_y1_s[i][nr] = 0.0;
                     d_z1_s[i][nr] = 0.0;
                     
                  };  		
                };
              };     
                
		
        if(stepper%10 == 0) counter_test++;
        
              if(counter_test%2 == 0) {

                  for(i=1;i<=natoms1;i++){
                  
                    f_x1_s[i] = f_global[at_id[i]][0]*ir->delta_t;
                    f_y1_s[i] = f_global[at_id[i]][1]*ir->delta_t;
                    f_z1_s[i] = f_global[at_id[i]][2]*ir->delta_t;                    
                 
                    x0_s[i] = state_global->x[at_id[i]][0];
	                y0_s[i] = state_global->x[at_id[i]][1];
                    z0_s[i] = state_global->x[at_id[i]][2];
                    
                  };
                  
              } else {
             
                  for(i=1;i<=natoms1;i++){
                  
                    f_x2_s[i] = f_global[at_id[i]][0]*ir->delta_t;
                    f_y2_s[i] = f_global[at_id[i]][1]*ir->delta_t;
                    f_z2_s[i] = f_global[at_id[i]][2]*ir->delta_t;                    
                 
                    x1_s[i] = state_global->x[at_id[i]][0];
	                y1_s[i] = state_global->x[at_id[i]][1];
                    z1_s[i] = state_global->x[at_id[i]][2];
							    

		        };
               }; 
            
            
        if(counter_test%2 == 0) {
            
            for(i=1;i<=natoms1;i++){  
                  
               for(nr=1;nr<=number_of_replicas;nr++){         
           
               if( (rand()%100)/100.0 <= probability_input2) {         
                   
               if(x0_s[i] - x1_s[i] < state_global->box[0][0]/2.0) d_x1_s[i][nr] = (x0_s[i] - x1_s[i]);
               if(y0_s[i] - y1_s[i] < state_global->box[1][1]/2.0) d_y1_s[i][nr] = (y0_s[i] - y1_s[i]);
               if(z0_s[i] - z1_s[i] < state_global->box[2][2]/2.0) d_z1_s[i][nr] = (z0_s[i] - z1_s[i]);                
           
               if(x0_s[i] - x1_s[i] > state_global->box[0][0]/2.0) d_x1_s[i][nr] = ((x0_s[i] - x1_s[i])-state_global->box[0][0]/2.0);
               if(y0_s[i] - y1_s[i] > state_global->box[1][1]/2.0) d_y1_s[i][nr] = ((y0_s[i] - y1_s[i])-state_global->box[1][1]/2.0);
               if(z0_s[i] - z1_s[i] > state_global->box[2][2]/2.0) d_z1_s[i][nr] = ((z0_s[i] - z1_s[i])-state_global->box[2][2]/2.0);               
               
               if(x0_s[i] - x1_s[i] < -state_global->box[0][0]/2.0) d_x1_s[i][nr] = ((x0_s[i] - x1_s[i])+state_global->box[0][0]/2.0);
               if(y0_s[i] - y1_s[i] < -state_global->box[1][1]/2.0) d_y1_s[i][nr] = ((y0_s[i] - y1_s[i])+state_global->box[1][1]/2.0);
               if(z0_s[i] - z1_s[i] < -state_global->box[2][2]/2.0) d_z1_s[i][nr] = ((z0_s[i] - z1_s[i])+state_global->box[2][2]/2.0);                
               
               
      //         df_x_s[i][nr] += (f_x1_s[i] + (f_x1_s[i]-f_x2_s[i]))/ir->delta_t*d_x1_s[i][nr];
      //         df_y_s[i][nr] += (f_y1_s[i] + (f_y1_s[i]-f_y2_s[i]))/ir->delta_t*d_y1_s[i][nr];              
      //         df_z_s[i][nr] += (f_z1_s[i] + (f_z1_s[i]-f_z2_s[i]))/ir->delta_t*d_z1_s[i][nr];
               
      //         grad_x[i] = ((f_x1_s[i] + (f_x1_s[i]-f_x2_s[i]))/ir->delta_t);
      //         grad_y[i] = ((f_y1_s[i] + (f_y1_s[i]-f_y2_s[i]))/ir->delta_t);
      //         grad_z[i] = ((f_z1_s[i] + (f_z1_s[i]-f_z2_s[i]))/ir->delta_t);  
               
               df_x_s[i][nr] += ((f_y1_s[i] + (f_y1_s[i]-f_y2_s[i]))/ir->delta_t)*d_z1_s[i][nr] - ((f_z1_s[i] + (f_z1_s[i]-f_z2_s[i]))/ir->delta_t)*d_y1_s[i][nr];
               df_y_s[i][nr] += ((f_z1_s[i] + (f_z1_s[i]-f_z2_s[i]))/ir->delta_t)*d_x1_s[i][nr] - ((f_x1_s[i] + (f_x1_s[i]-f_x2_s[i]))/ir->delta_t)*d_z1_s[i][nr];               
               df_z_s[i][nr] += ((f_x1_s[i] + (f_x1_s[i]-f_x2_s[i]))/ir->delta_t)*d_y1_s[i][nr] - ((f_y1_s[i] + (f_y1_s[i]-f_y2_s[i]))/ir->delta_t)*d_x1_s[i][nr];
               
               grad_x[i] = ((f_y1_s[i] + (f_y1_s[i]-f_y2_s[i]))/ir->delta_t)*d_z1_s[i][nr] - ((f_z1_s[i] + (f_z1_s[i]-f_z2_s[i]))/ir->delta_t)*d_y1_s[i][nr];
               grad_y[i] = ((f_z1_s[i] + (f_z1_s[i]-f_z2_s[i]))/ir->delta_t)*d_x1_s[i][nr] - ((f_x1_s[i] + (f_x1_s[i]-f_x2_s[i]))/ir->delta_t)*d_z1_s[i][nr];
               grad_z[i] = ((f_x1_s[i] + (f_x1_s[i]-f_x2_s[i]))/ir->delta_t)*d_y1_s[i][nr] - ((f_y1_s[i] + (f_y1_s[i]-f_y2_s[i]))/ir->delta_t)*d_x1_s[i][nr];               

               d_t = 1.0/3.0*sqrt(pow(df_x_s[i][nr],2)+pow(df_y_s[i][nr],2)+pow(df_z_s[i][nr],2));

               grad_x1[i] = grad_x[i];
               grad_y1[i] = grad_y[i];
               grad_z1[i] = grad_z[i];
               
               if(d_t > d_max) d_max = d_t;               
               
                };
              };
            };
           };  
           
           
        if(counter_test%2 != 0) {
            
            count_num_resolve++;
                                        
                  alpha_bar_x = 0.0;
                  alpha_bar_y = 0.0;                  
                  alpha_bar_z = 0.0;
            
                  tau_bar = 0;
                
              for(i=1;i<=natoms1;i++){
                  
                for(nr=1;nr<=number_of_replicas;nr++){                                    
                  
                     tau_bar += tau[i][nr];
                     alpha_bar_x += alpha_1_x[i][nr];
                     alpha_bar_y += alpha_1_y[i][nr];                  
                     alpha_bar_z += alpha_1_z[i][nr];
                     
                  };  
                };                
                  
                  alpha_bar_x /= (double)(natoms1+number_of_replicas);
                  alpha_bar_y /= (double)(natoms1+number_of_replicas);                  
                  alpha_bar_z /= (double)(natoms1+number_of_replicas);
                
                  tau_bar = (int)(((double)tau_bar/(double)(natoms1+number_of_replicas)));
                  
                  
            d_t = 0.0;
            sum_dL = 0.0;
              
        for(i=1;i<=natoms1;i++){  
             
              for(nr=1;nr<=number_of_replicas;nr++){  
                  
               if( (rand()%100)/100.0 <= probability_input2) {
                   
               if(x1_s[i] - x0_s[i] < state_global->box[0][0]/2.0) d_x1_s[i][nr] = (x1_s[i] - x0_s[i]);
               if(y1_s[i] - y0_s[i] < state_global->box[1][1]/2.0) d_y1_s[i][nr] = (y1_s[i] - y0_s[i]);
               if(z1_s[i] - z0_s[i] < state_global->box[2][2]/2.0) d_z1_s[i][nr] = (z1_s[i] - z0_s[i]);                
           
               if(x1_s[i] - x0_s[i] > state_global->box[0][0]/2.0) d_x1_s[i][nr] = ((x1_s[i] - x0_s[i])-state_global->box[0][0]/2.0);
               if(y1_s[i] - y0_s[i] > state_global->box[1][1]/2.0) d_y1_s[i][nr] = ((y1_s[i] - y0_s[i])-state_global->box[1][1]/2.0);
               if(z1_s[i] - z0_s[i] > state_global->box[2][2]/2.0) d_z1_s[i][nr] = ((z1_s[i] - z0_s[i])-state_global->box[2][2]/2.0);               
               
               if(x1_s[i] - x0_s[i] < -state_global->box[0][0]/2.0) d_x1_s[i][nr] = ((x1_s[i] - x0_s[i])+state_global->box[0][0]/2.0);
               if(y1_s[i] - y0_s[i] < -state_global->box[1][1]/2.0) d_y1_s[i][nr] = ((y1_s[i] - y0_s[i])+state_global->box[1][1]/2.0);
               if(z1_s[i] - z0_s[i] < -state_global->box[2][2]/2.0) d_z1_s[i][nr] = ((z1_s[i] - z0_s[i])+state_global->box[2][2]/2.0);                
               
          //     df_x_s[i][nr] += (f_x2_s[i] + (f_x2_s[i]-f_x1_s[i]))/ir->delta_t*d_x1_s[i][nr];
          //     df_y_s[i][nr] += (f_y2_s[i] + (f_y2_s[i]-f_y1_s[i]))/ir->delta_t*d_y1_s[i][nr];              
          //     df_z_s[i][nr] += (f_z2_s[i] + (f_z2_s[i]-f_z1_s[i]))/ir->delta_t*d_z1_s[i][nr];  
               
               df_x_s[i][nr] += ((f_y2_s[i] + (f_y2_s[i]-f_y1_s[i]))/ir->delta_t)*d_z1_s[i][nr] - ((f_z2_s[i] + (f_z2_s[i]-f_z1_s[i]))/ir->delta_t)*d_y1_s[i][nr];
               df_y_s[i][nr] += ((f_z2_s[i] + (f_z2_s[i]-f_z1_s[i]))/ir->delta_t)*d_x1_s[i][nr] - ((f_x2_s[i] + (f_x2_s[i]-f_x1_s[i]))/ir->delta_t)*d_z1_s[i][nr];               
               df_z_s[i][nr] += ((f_x2_s[i] + (f_x2_s[i]-f_x1_s[i]))/ir->delta_t)*d_y1_s[i][nr] - ((f_y2_s[i] + (f_y2_s[i]-f_y1_s[i]))/ir->delta_t)*d_x1_s[i][nr];                  
               
           };
          };
         };
        };
        
        for(i=1;i<=natoms1;i++){  
             
              for(nr=1;nr<=number_of_replicas;nr++){
                  
             // density calculation for replica exchange
                        
                dL_abs[i][nr] = (f_x2_s[i] + (f_x2_s[i]-f_x1_s[i]))/ir->delta_t*d_x1_s[i][nr]+
                                (f_y2_s[i] + (f_y2_s[i]-f_y1_s[i]))/ir->delta_t*d_y1_s[i][nr]+
                                (f_z2_s[i] + (f_z2_s[i]-f_z1_s[i]))/ir->delta_t*d_z1_s[i][nr]; 
                sum_dL += dL_abs[i][nr];
    
         //    if(count_num_resolve%tau[i][nr] == 0) {
                
             if(stepper22%2 == 0){   
                
               delta_L_x[i][nr] = sqrt(pow((f_x2_s[i] + (f_x2_s[i]-f_x1_s[i]))/ir->delta_t*d_x1_s[i][nr],2));
               delta_L_y[i][nr] = sqrt(pow((f_y2_s[i] + (f_y2_s[i]-f_y1_s[i]))/ir->delta_t*d_y1_s[i][nr],2));
               delta_L_z[i][nr] = sqrt(pow((f_z2_s[i] + (f_z2_s[i]-f_z1_s[i]))/ir->delta_t*d_z1_s[i][nr],2));                 
               
             } else {
               
               delta_L_x2[i][nr] = delta_L_x[i][nr] - sqrt(pow((f_x2_s[i] + (f_x2_s[i]-f_x1_s[i]))/ir->delta_t*d_x1_s[i][nr],2));
               delta_L_y2[i][nr] = delta_L_y[i][nr] - sqrt(pow((f_y2_s[i] + (f_y2_s[i]-f_y1_s[i]))/ir->delta_t*d_y1_s[i][nr],2));
               delta_L_z2[i][nr] = delta_L_z[i][nr] - sqrt(pow((f_z2_s[i] + (f_z2_s[i]-f_z1_s[i]))/ir->delta_t*d_z1_s[i][nr],2)); 
               
             };
               
               if(delta_alpha_1x[i][nr] != 0.0 && delta_alpha_2x[i][nr] != 0.0) d_x = (delta_L_x[i][nr]/(delta_alpha_1x[i][nr])+delta_L_x2[i][nr]/(delta_alpha_1x[i][nr]*delta_alpha_2x[i][nr])*(alpha_1_x[i][nr]-alpha_bar_x));
               if(delta_alpha_1y[i][nr] != 0.0 && delta_alpha_2y[i][nr] != 0.0) d_y = (delta_L_y[i][nr]/(delta_alpha_1y[i][nr])+delta_L_y2[i][nr]/(delta_alpha_1y[i][nr]*delta_alpha_2y[i][nr])*(alpha_1_y[i][nr]-alpha_bar_y));
               if(delta_alpha_1z[i][nr] != 0.0 && delta_alpha_2z[i][nr] != 0.0) d_z = (delta_L_z[i][nr]/(delta_alpha_1z[i][nr])+delta_L_z2[i][nr]/(delta_alpha_1z[i][nr]*delta_alpha_2z[i][nr])*(alpha_1_z[i][nr]-alpha_bar_z));
               
               d_t = sqrt(pow(d_x,2)+pow(d_y,2)+pow(d_z,2));
               
               if(d_t > 0.0) alpha_2_x[i][nr] = alpha_1_x[i][nr] + alpha_1_x[i][nr]*stepsize_alpha*d_x/d_t;
               if(d_t > 0.0) alpha_2_y[i][nr] = alpha_1_y[i][nr] + alpha_1_y[i][nr]*stepsize_alpha*d_y/d_t;
               if(d_t > 0.0) alpha_2_z[i][nr] = alpha_1_z[i][nr] + alpha_1_z[i][nr]*stepsize_alpha*d_z/d_t;               
               
             if(stepper22%2 == 0) {  
               
               delta_alpha_2x[i][nr] = delta_alpha_1x[i][nr];
               delta_alpha_2y[i][nr] = delta_alpha_1y[i][nr];               
               delta_alpha_2z[i][nr] = delta_alpha_1z[i][nr];               
               
             };
             
             if(stepper22%2 != 0) {              
               
               delta_alpha_1x[i][nr] = alpha_2_x[i][nr] - alpha_1_x[i][nr];
               delta_alpha_1y[i][nr] = alpha_2_y[i][nr] - alpha_1_y[i][nr];
               delta_alpha_1z[i][nr] = alpha_2_z[i][nr] - alpha_1_z[i][nr];              
               
               delta_alpha_2x[i][nr] -= delta_alpha_1x[i][nr];
               delta_alpha_2y[i][nr] -= delta_alpha_1y[i][nr];
               delta_alpha_2z[i][nr] -= delta_alpha_1z[i][nr];               
               
             };
             
             if(stepper22%2 == 0) {             
               
               alpha_1_x[i][nr]    = alpha_2_x[i][nr];
               alpha_1_y[i][nr]    = alpha_2_y[i][nr];
               alpha_1_z[i][nr]    = alpha_2_z[i][nr];
               
             };
             
          /*   if(stepper22%20000 == 0) {
               fprintf(fplog,"%d\t%d\t%d\t%d\t%s\n",tau[i][nr],tau2[i][nr],nr,i," tau, tau2, replica "); 
               fprintf(fplog,"%e\t%e\t%e\t%s\n",d_x,d_y,d_z," delta_alpha 2x, delta_alpha 2y, delta_alpha 2z ");
               fprintf(fplog,"%e\t%e\t%e\t%s\n",alpha_2_x[i][nr],alpha_2_y[i][nr],alpha_2_z[i][nr]," alpha 2x, alpha 2y, alpha 2z ");              
             }; */
               
               if(alpha_1_x[i][nr] <= beta2/range_alpha) alpha_1_x[i][nr] = beta2;
               if(alpha_1_y[i][nr] <= beta2/range_alpha) alpha_1_y[i][nr] = beta2;
               if(alpha_1_z[i][nr] <= beta2/range_alpha) alpha_1_z[i][nr] = beta2; 
               if(alpha_2_x[i][nr] <= beta2/range_alpha) alpha_2_x[i][nr] = beta2;
               if(alpha_2_y[i][nr] <= beta2/range_alpha) alpha_2_y[i][nr] = beta2;
               if(alpha_2_z[i][nr] <= beta2/range_alpha) alpha_2_z[i][nr] = beta2; 
               
               if(alpha_1_x[i][nr] >= beta2*range_alpha) alpha_1_x[i][nr] = beta2;
               if(alpha_1_y[i][nr] >= beta2*range_alpha) alpha_1_y[i][nr] = beta2;
               if(alpha_1_z[i][nr] >= beta2*range_alpha) alpha_1_z[i][nr] = beta2; 
               if(alpha_2_x[i][nr] >= beta2*range_alpha) alpha_2_x[i][nr] = beta2;
               if(alpha_2_y[i][nr] >= beta2*range_alpha) alpha_2_y[i][nr] = beta2;
               if(alpha_2_z[i][nr] >= beta2*range_alpha) alpha_2_z[i][nr] = beta2;                
             
               d_t = 0.0; 
               
               if(delta_tau[i][nr] != 0.0 && delta_tau_2[i][nr] != 0.0) d_x = ((delta_L_x[i][nr])/((double)delta_tau[i][nr]) + 
                   (delta_L_x2[i][nr])/((double)delta_tau[i][nr]*(double)delta_tau_2[i][nr])*((double)tau[i][nr]-(double)tau_bar));              
               if(delta_tau[i][nr] != 0.0 && delta_tau_2[i][nr] != 0.0) d_y = ((delta_L_y[i][nr])/((double)delta_tau[i][nr]) + 
                   (delta_L_y2[i][nr])/((double)delta_tau[i][nr]*(double)delta_tau_2[i][nr])*((double)tau[i][nr]-(double)tau_bar));                 
               if(delta_tau[i][nr] != 0.0 && delta_tau_2[i][nr] != 0.0) d_z = ((delta_L_z[i][nr])/((double)delta_tau[i][nr]) + 
                   (delta_L_z2[i][nr])/((double)delta_tau[i][nr]*(double)delta_tau_2[i][nr])*((double)tau[i][nr]-(double)tau_bar));                 
               
               d_t = sqrt(pow(d_x,2)+pow(d_y,2)+pow(d_z,2));
               
               if(d_t > 0.0) d_x = (d_x+d_y+d_z)/(3.0*d_t);
               if(d_t > 0.0) tau[i][nr] = tau_1[i][nr] + (int)((stepsize_alpha*d_x/d_t*tau_1[i][nr]));
              
               if(stepper22%2 == 0) delta_tau_2[i][nr] = delta_tau[i][nr];
               if(stepper22%2 != 0) delta_tau[i][nr] = tau[i][nr] - tau_1[i][nr];
               if(stepper22%2 != 0) delta_tau_2[i][nr] -= delta_tau[i][nr];
               if(stepper22%2 == 0) tau_1[i][nr] = tau[i][nr];

               if(tau[i][nr] > tau_a*(int)range_alpha*nr) {
                   
                   tau[i][nr] = tau_a*nr;//*(int)range_alpha*nr;
                   tau2[i][nr]  = tau[i][nr]*tau2_a/tau_a; 
                   
               };
                   
               if(tau[i][nr] < 5) { 
                   
                   tau[i][nr] = tau_a*nr;               
                   tau2[i][nr]  = tau[i][nr]*tau2_a/tau_a; 
                   
               };
                                    
                d_t = 1.0/3.0*sqrt(pow(df_x_s[i][nr],2)+pow(df_y_s[i][nr],2)+pow(df_z_s[i][nr],2));
                if(d_t > d_max) d_max = d_t;
                
               }; 
              
               
          //     grad_x[i] = ((f_x2_s[i] + (f_x2_s[i]-f_x1_s[i]))/ir->delta_t);
          //     grad_y[i] = ((f_y2_s[i] + (f_y2_s[i]-f_y1_s[i]))/ir->delta_t);
          //     grad_z[i] = ((f_z2_s[i] + (f_z2_s[i]-f_z1_s[i]))/ir->delta_t); 
               
               grad_x[i] = ((f_y2_s[i] + (f_y2_s[i]-f_y1_s[i]))/ir->delta_t)*d_z1_s[i][nr] - ((f_z2_s[i] + (f_z2_s[i]-f_z1_s[i]))/ir->delta_t)*d_y1_s[i][nr];
               grad_y[i] = ((f_z2_s[i] + (f_z2_s[i]-f_z1_s[i]))/ir->delta_t)*d_x1_s[i][nr] - ((f_x2_s[i] + (f_x2_s[i]-f_x1_s[i]))/ir->delta_t)*d_z1_s[i][nr];
               grad_z[i] = ((f_x2_s[i] + (f_x2_s[i]-f_x1_s[i]))/ir->delta_t)*d_y1_s[i][nr] - ((f_y2_s[i] + (f_y2_s[i]-f_y1_s[i]))/ir->delta_t)*d_x1_s[i][nr];                 
               
               grad_x0[i] = grad_x[i];
               grad_y0[i] = grad_y[i];
               grad_z0[i] = grad_z[i];              
                            
               
              };
              
              
              
            for(i=1;i<=natoms1;i++){
              for(nr=1;nr<=number_of_replicas;nr++) {   
               
                  if(sum_dL != 0.0) {
                      
                      rho[i][nr] = dL_abs[i][nr]/sum_dL;
                      rho_sys += rho[i][nr];
                  
                      };
                 };
                };            
            
  flag_diagonalize = 0;
      
  for(nr = 1; nr <= number_of_replicas; nr ++) {
    for(n=1;n<=natoms1;n++) {  
       if(stepper%tau2[n][nr] == 0  && density_boolean == 1) {
           
           flag_diagonalize = nr;
           
       };
    };
  };
         
     if(flag_diagonalize > 0) {
  
              k = 1;
              l = 1;
              
              for(i=1;i<=natoms1*natoms1;i++) {                            
                      
                         tmp_x[l]   = df_x_s[k][flag_diagonalize];  
                         tmp_x[l+1] = df_y_s[k][flag_diagonalize];
                         tmp_x[l+2] = df_z_s[k][flag_diagonalize];
                         
                         if(k == natoms1) k = 0;
                         l = l + 3;
                         
                         k ++; 
                     };
                     
            info = eigensolver(tmp_x,natoms1*3,1,natoms1*3,eigval,mat);
            
            if(info == 0) {
                
               var = 1E+10;
               l = 1;               
               
               for(i=1;i<=natoms1;i++) {
   
                  d_t = sqrt(pow(eigval[l],2) + pow(eigval[l+1],2) + pow(eigval[l+2],2)); 
                   
                  if(d_t < var) {
                     
                      selector = i;
                      var      = d_t;
                      
                   };    
                   
                   l = l + 3;
                   
               };               
            
               l = 1; 
                
               for(i=1;i<=natoms1;i++) {
                     
                        df_x_s[i][flag_diagonalize] = mat[l+((selector-1)*3*natoms1*3)]; 
                        df_y_s[i][flag_diagonalize] = mat[l+1+((selector-1)*3*natoms1*3)];
                        df_z_s[i][flag_diagonalize] = mat[l+2+((selector-1)*3*natoms1*3)];
                        
                        l = l + 3;
                    };
                 };    
            
             };             
                        
    for(nr = 1; nr <= number_of_replicas; nr ++) {
 
    double width_WA;
            
     //    replica_fraction = (double)(nr);//*(double)natoms1/(double)number_of_replicas;
               
         replica_fraction_integer = 1;  // + (int)roundf(replica_fraction);        

	  for(i=1;i<=natoms1;i++){  

        if(stepper%tau2[i][nr] == 0){  
          
              if(i%replica_fraction_integer == 0) {

                         df_xx = df_x_s[i][nr];//state_global->x[dstart-1][0] + state_global->x[dend-1][0];
                         df_yx = df_y_s[i][nr];//state_global->x[dstart-1][1] + state_global->x[dend-1][1];
                         df_zx = df_z_s[i][nr];//state_global->x[dstart-1][2] + state_global->x[dend-1][2];
                         
                         d_t = sqrt(pow(df_xx,2));
			 			 
                         if(d_max > 0.0) d_t = d_t/d_max;
                         
                 if(d_t <= r_g[i] && d_t > 0.0) {  

                        sigma_xx[i] = d_t;
                    
                        delta_s[i] = sqrt(pow(sigma_xx[i]-sigma_strich_xx[i][nr],2));
                        if(sigma_xx[i] > 0.0) width_WA   = WA*sqrt(pow(sigma_xx[i]-sigma_strich_xx[i][nr],2))/sigma_xx[i]; 

//                    printf("%lg\t%lg\t%lg\t%lg\t%s\n",delta_s[i],width_WA,sigma_x[i],sigma_strich_x[i]," width WA ");
  
                        
                        d_range = 0.0; 

                  if(delta_s[i] > 0.0){      
                        
                   for(k=1;k<=100;k++){

                       d_range = d_range + r_g[i]/100.0;

                       phi[k] = - width_WA*exp(grad_meta_x[i][k]/wt_meta_delta_t)*exp(-(pow(d_range-d_t,2))/(2.0*(pow(delta_s[i],2))));

//		       printf("%lg\t%lg\t%lg\t%lg\t%s\n",phi[k],d_range,delta_s[i],angle_phi,"phi");
		       
                   };
                  };
                   
                          d_range = 0.0;

                   for(k=1;k<=100;k++){

                       grad_meta_x[i][k] += phi[k];

//                    printf("%lg\t%lg\t%s\n",r_g[i],grad_meta[i][k],"grad-meta");

                   }; 
                 };

                         d_t = sqrt(pow(df_yx,2));
			 			 
                         if(d_max > 0.0) d_t = d_t/d_max;
                         
                 if(d_t <= r_g[i] && d_t > 0.0) {  

                        sigma_yx[i] = d_t;

                        delta_s[i] = sqrt(pow(sigma_yx[i]-sigma_strich_yx[i][nr],2));
                        if(sigma_yx[i] > 0.0) width_WA   = WA*sqrt(pow(sigma_yx[i]-sigma_strich_yx[i][nr],2))/sigma_yx[i]; 

//                    printf("%lg\t%lg\t%lg\t%lg\t%s\n",delta_s[i],width_WA,sigma_x[i],sigma_strich_x[i]," width WA ");
  
                        d_range = 0.0; 

                 if(delta_s[i] > 0.0){       
                        
                   for(k=1;k<=100;k++){

                       d_range = d_range + r_g[i]/100.0;

                       phi[k] = - width_WA*exp(grad_meta_y[i][k]/wt_meta_delta_t)*exp(-(pow(d_range-d_t,2))/(2.0*(pow(delta_s[i],2))));

//		       printf("%lg\t%lg\t%lg\t%lg\t%s\n",phi[k],d_range,delta_s[i],angle_phi,"phi");
		       
                   };
                 };
                   
                          d_range = 0.0;

                   for(k=1;k<=100;k++){

                       grad_meta_y[i][k] += phi[k];

//                    printf("%lg\t%lg\t%s\n",r_g[i],grad_meta[i][k],"grad-meta");

                   };
                 };                 
                 
                         d_t = sqrt(pow(df_zx,2));
			 			 
                         if(d_max > 0.0) d_t = d_t/d_max;
                         
                 if(d_t <= r_g[i] && d_t > 0.0) {  

                        sigma_zx[i] = d_t;

                        delta_s[i] = sqrt(pow(sigma_zx[i]-sigma_strich_zx[i][nr],2));
                        if(sigma_zx[i] > 0.0) width_WA   = WA*sqrt(pow(sigma_zx[i]-sigma_strich_zx[i][nr],2))/sigma_zx[i]; 

//                    printf("%lg\t%lg\t%lg\t%lg\t%s\n",delta_s[i],width_WA,sigma_zx[i],sigma_strich_zx[i]," width WA ");
  
                        d_range = 0.0;
                        
                        
                 if(delta_s[i] > 0.0) {       

                   for(k=1;k<=100;k++){

                       d_range = d_range + r_g[i]/100.0;

                       phi[k] = - width_WA*exp(grad_meta_z[i][k]/wt_meta_delta_t)*exp(-(pow(d_range-d_t,2))/(2.0*(pow(delta_s[i],2))));

//		       printf("%lg\t%lg\t%lg\t%lg\t%s\n",phi[k],d_range,delta_s[i],angle_phi,"phi");
		       
                   };
                 };
                   
                          d_range = 0.0;

                   for(k=1;k<=100;k++){

                       grad_meta_z[i][k] += phi[k];

//                    printf("%lg\t%lg\t%s\n",r_g[i],grad_meta_z[i][k],"grad-meta");
                   };
                 };                    
                };
               };
                step_father[nr] = stepper;
             };
            };

      for(nr=1;nr<=number_of_replicas;nr++){
      
             if(stepper == step_father[nr] + 1) 

              {

                for(i=1;i<=natoms1;i++){

                    sigma_strich_xx[i][nr] = sigma_xx[i];
                    sigma_strich_yx[i][nr] = sigma_yx[i];
                    sigma_strich_zx[i][nr] = sigma_zx[i];                    
                    
                  };
                 }; 
                };   

    if(stepper > tau2_a && stepper22%2 == 0){     
              
      for(i=1;i<=natoms1;i++){ 

          
             df_xx = (grad_x[i]);//- state_global->x[dstart-1][0] + state_global->x[dend-1][0];
             df_yx = (grad_y[i]);//- state_global->x[dstart-1][1] + state_global->x[dend-1][1];
             df_zx = (grad_z[i]);//- state_global->x[dstart-1][2] + state_global->x[dend-1][2];
            
             
             df_xx = df_xx / d_max;
             df_yx = df_yx / d_max;
             df_zx = df_zx / d_max;
             
//             printf("%lg\t%lg\t%lg\t%lg\n",df_xx,df_yx,df_zx,angle_phi);             

             d_t = sqrt(pow(df_xx,2));  
             
             if(d_max > 0.0) d_t = d_t / d_max;            
             
          if(d_t <= r_g[i] && d_t > 0.0){  

             pot_x = 0.0;

             d_range = 0.0;

             for(k=1;k<=100;k++){

                 d_range = d_range + r_g[i]/100.0; 

                 if(d_range - r_g[i]/100.0 >= d_t && d_t < d_range + r_g[i]/100.0){

                     pot_x = (grad_meta_x[i][k+1]-grad_meta_x[i][k-1]);   
		     
//                     printf("%lg\t%lg\n",angle_phi,pot);
                 };  
             };
          };

             d_t = sqrt(pow(df_yx,2));  
             
             if(d_max > 0.0) d_t = d_t / d_max;            
             
          if(d_t <= r_g[i] && d_t > 0.0){  

             pot_y = 0.0;

             d_range = 0.0;

             for(k=1;k<=100;k++){

                 d_range = d_range + r_g[i]/100.0; 

                 if(d_range - r_g[i]/100.0 >= d_t && d_t < d_range + r_g[i]/100.0){

                     pot_y = (grad_meta_y[i][k+1]-grad_meta_y[i][k-1]);   
		     
//                     printf("%lg\t%lg\n",angle_phi,pot);
                 };  
             }; 
          };
             
             d_t = sqrt(pow(df_zx,2));  
             
             if(d_max > 0.0) d_t = d_t / d_max;            
             
          if(d_t <= r_g[i] && d_t > 0.0){  

             pot_z = 0.0;

             d_range = 0.0;

             for(k=1;k<=100;k++){

                 d_range = d_range + r_g[i]/100.0; 

                 if(d_range - r_g[i]/100.0 >= d_t && d_t < d_range + r_g[i]/100.0){

                     pot_z = (grad_meta_z[i][k+1]-grad_meta_z[i][k-1]);   
		     
//                     printf("%lg\t%lg\n",angle_phi,pot);
                 };  
              };                 
            };
             
             d_t = sqrt(pow(df_xx,2)+pow(df_yx,2)+pow(df_zx,2));
             
             
             if(d_t > 0.0 && r_g[i] > 0.0) {

                delta_fx[i] = df_xx/d_t*pot_x/r_g[i]/50.0; //*norm1/norm2*maxf0;
                delta_fy[i] = df_yx/d_t*pot_y/r_g[i]/50.0; //*norm1/norm2*maxf0;
                delta_fz[i] = df_zx/d_t*pot_z/r_g[i]/50.0; //*norm1/norm2*maxf0;
                
     //         if(stepper%1000 == 0) printf("%lg\t%lg\t%lg\n",delta_fx[i],delta_fy[i],delta_fz[i]);
	          } else {

                delta_fx[i] = 0.0; //*norm1/norm2*maxf0;
                delta_fy[i] = 0.0; //*norm1/norm2*maxf0;
                delta_fz[i] = 0.0; //*norm1/norm2*maxf0;

             };
            }; 
           };
           
         sum_exp = 0.0;
         sum_of_all_derivatives_x5 = 0.0;
         sum_of_all_derivatives_y5 = 0.0;
         sum_of_all_derivatives_z5 = 0.0;                    
                     
         for(i=1;i<=natoms1;i++) 
           {
               
              sum_exp += exp(-sqrt(pow(f_global[at_id[i]][0],2) + pow(f_global[at_id[i]][1],2)
                             + pow(f_global[at_id[i]][2],2))/(8.314*enerd->term[F_TEMP]));
              exp_of_funct2 = exp(-sqrt(pow(f_global[at_id[i]][0],2) + pow(f_global[at_id[i]][1],2)
                             + pow(f_global[at_id[i]][2],2))/(8.314*enerd->term[F_TEMP])); 
                 
              sum_of_all_derivatives_x5 += (delta_fx[i])*exp_of_funct2; 
              sum_of_all_derivatives_y5 += (delta_fy[i])*exp_of_funct2;
              sum_of_all_derivatives_z5 += (delta_fz[i])*exp_of_funct2;

               
        }; 
      
          for(i=1;i<=natoms1;i++) {
                
                delta_fx[i] = delta_fx[i] - 1.0/sum_exp*sum_of_all_derivatives_x5;
                delta_fy[i] = delta_fy[i] - 1.0/sum_exp*sum_of_all_derivatives_y5;
                delta_fz[i] = delta_fz[i] - 1.0/sum_exp*sum_of_all_derivatives_z5;                  

            };           
           
		
           for(i=1;i<=natoms1;i++){

            if( (rand()%100)/100.0 <= probability_input) {   
               
            if(stepper > tau2_a) {  
               
                h5 = sqrt(pow(delta_fx[i],2)+pow(delta_fy[i],2)+pow(delta_fz[i],2));                
                h4 = sqrt(pow(f_global[at_id[i]][0],2)+pow(f_global[at_id[i]][1],2)+pow(f_global[at_id[i]][2],2));
                
                 alpha_3_x[i] = 0.0;
                 alpha_3_y[i] = 0.0;                 
                 alpha_3_z[i] = 0.0;  
            
          if(h5 > 0.0) {        
                 
             if(restraint_sampling[i] == 0) {
                  
                for(nr = 1; nr <= number_of_replicas ; nr ++) {
 
                  if(stepper%tau2[i][nr] == 0 || stepper == 1) {
  
                      random_number(&ran);
 
                      epsilon3[i][nr] = epsilon*(1.0-ran);
                      epsilon6[i][nr] = epsilon_md*(1.0-ran);

                };

                if(density_boolean == 0) {
                
                    f_global[at_id[i]][0] += epsilon3[i][nr]*alpha_2_x[i][nr]*h4/h5*delta_fx[i];
                    f_global[at_id[i]][1] += epsilon3[i][nr]*alpha_2_y[i][nr]*h4/h5*delta_fy[i];
                    f_global[at_id[i]][2] += epsilon3[i][nr]*alpha_2_z[i][nr]*h4/h5*delta_fz[i];
                    
                    alpha_3_x[i] += epsilon6[i][nr]*alpha_2_x[i][nr];
                    alpha_3_y[i] += epsilon6[i][nr]*alpha_2_y[i][nr];                 
                    alpha_3_z[i] += epsilon6[i][nr]*alpha_2_z[i][nr];                     
                    
                    };
                    
                if(density_boolean == 1 ) {
                
                   if(pow(d_x1[i][nr],2) > 0.0) f_global[at_id[i]][0] += (ran*epsilon/(1.0+sqrt(pow(d_x1[i][nr],2))))*alpha_2_x[i][nr]*h4/h5*delta_fx[i];
                   if(pow(d_y1[i][nr],2) > 0.0) f_global[at_id[i]][1] += (ran*epsilon/(1.0+sqrt(pow(d_y1[i][nr],2))))*alpha_2_y[i][nr]*h4/h5*delta_fy[i];
                   if(pow(d_z1[i][nr],2) > 0.0) f_global[at_id[i]][2] += (ran*epsilon/(1.0+sqrt(pow(d_z1[i][nr],2))))*alpha_2_z[i][nr]*h4/h5*delta_fz[i];
                    
                   if(pow(d_x1[i][nr],2) > 0.0) alpha_3_x[i] += (ran*epsilon_md/(1.0+sqrt(pow(d_x1[i][nr],2))))*alpha_2_x[i][nr];
                   if(pow(d_y1[i][nr],2) > 0.0) alpha_3_y[i] += (ran*epsilon_md/(1.0+sqrt(pow(d_y1[i][nr],2))))*alpha_2_y[i][nr];                 
                   if(pow(d_z1[i][nr],2) > 0.0) alpha_3_z[i] += (ran*epsilon_md/(1.0+sqrt(pow(d_z1[i][nr],2))))*alpha_2_z[i][nr];                     
                    
                    };                  
                    
                };
        
             } else {
                    
                 for(k=1;k<=number_of_restraints;k++) {   
                     
                         if(density_boolean == 0) {
                              
                             cos_val = 1.0;
                             
                             if(i == at_id_r[k]) {
                                 
                                  d_x = state_global->x[at_id_r[k]][0] - state_global->x[at_id_r2[k]][0];
                                  d_y = state_global->x[at_id_r[k]][1] - state_global->x[at_id_r2[k]][1];
                                  d_z = state_global->x[at_id_r[k]][2] - state_global->x[at_id_r2[k]][2];

                                  d_t = sqrt(pow(d_x,2)+pow(d_y,2)+pow(d_z,2));
                                  
                                  d_t2 = sqrt(pow(delta_fx[i],2)+pow(delta_fy[i],2)+pow(delta_fz[i],2));
                                  
                                  if(d_t*d_t2 > 0.0) cos_val = (d_x*delta_fx[i] + d_y*delta_fy[i] + d_z*delta_fz[i])/(d_t*d_t2); 
                                  
                                  if(distance_restraint[k]-d_t != 0.0) {  
                                  
                                    f_global[at_id_r[k]][0] += epsilon3[i][nr]*alpha_2_x[i][nr]*h4/h5*delta_fx[i]*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                    f_global[at_id_r[k]][1] += epsilon3[i][nr]*alpha_2_y[i][nr]*h4/h5*delta_fy[i]*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                    f_global[at_id_r[k]][2] += epsilon3[i][nr]*alpha_2_z[i][nr]*h4/h5*delta_fz[i]*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;                                    
                               
                                    alpha_3_x[i] += epsilon3[i][nr]*alpha_2_x[i][nr];
                                    alpha_3_y[i] += epsilon3[i][nr]*alpha_2_y[i][nr];                 
                                    alpha_3_z[i] += epsilon3[i][nr]*alpha_2_z[i][nr];                                       
                                    
                                   };
                                  };
                                  
                             if(i == at_id_r2[k]) {
                                 
                                  d_x = state_global->x[at_id_r2[k]][0] - state_global->x[at_id_r[k]][0];
                                  d_y = state_global->x[at_id_r2[k]][1] - state_global->x[at_id_r[k]][1];
                                  d_z = state_global->x[at_id_r2[k]][2] - state_global->x[at_id_r[k]][2];

                                  d_t = sqrt(pow(d_x,2)+pow(d_y,2)+pow(d_z,2));
                                  
                                  d_t2 = sqrt(pow(delta_fx[i],2)+pow(delta_fy[i],2)+pow(delta_fz[i],2));
                                  
                                  if(d_t*d_t2 > 0.0)cos_val = (d_x*delta_fx[i] + d_y*delta_fy[i] + d_z*delta_fz[i])/(d_t*d_t2); 
                                  
                                  
                                  if(distance_restraint[k]-d_t != 0.0) {  
                                      
                                  
                                    if(d_x1[i][nr] != 0.0) f_global[at_id_r2[k]][0] += epsilon3[i][nr]*alpha_2_x[i][nr]*h4/h5*delta_fx[i]*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                    if(d_y1[i][nr] != 0.0) f_global[at_id_r2[k]][1] += epsilon3[i][nr]*alpha_2_y[i][nr]*h4/h5*delta_fy[i]*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                    if(d_z1[i][nr] != 0.0) f_global[at_id_r2[k]][2] += epsilon3[i][nr]*alpha_2_z[i][nr]*h4/h5*delta_fz[i]*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;                                    
                               
                                    alpha_3_x[i] += epsilon3[i][nr]*alpha_2_x[i][nr];
                                    alpha_3_y[i] += epsilon3[i][nr]*alpha_2_y[i][nr];                 
                                    alpha_3_z[i] += epsilon3[i][nr]*alpha_2_z[i][nr];                                     
                                    
                                    };
                                  };
                              };
                              
                         if(density_boolean == 1) {
                              
                             cos_val = 1.0;
                             
                             if(i == at_id_r[k]) {
                                 
                                  d_x = state_global->x[at_id_r[k]][0] - state_global->x[at_id_r2[k]][0];
                                  d_y = state_global->x[at_id_r[k]][1] - state_global->x[at_id_r2[k]][1];
                                  d_z = state_global->x[at_id_r[k]][2] - state_global->x[at_id_r2[k]][2];

                                  d_t = sqrt(pow(d_x,2)+pow(d_y,2)+pow(d_z,2));
                                  
                                  d_t2 = sqrt(pow(delta_fx[i],2)+pow(delta_fy[i],2)+pow(delta_fz[i],2));
                                  
                                  if(d_t*d_t2 > 0.0)cos_val = (d_x*delta_fx[i] + d_y*delta_fy[i] + d_z*delta_fz[i])/(d_t*d_t2); 
                                  
                                  if(distance_restraint[k]-d_t != 0.0) { 
                                      
                                    ran = rand()%100/100.0;                                       
                                  
                                    if(d_x1[i][nr] != 0.0) f_global[at_id_r[k]][0] += (ran*epsilon/(1.0+sqrt(pow(d_x1[i][nr],2))))*alpha_2_x[i][nr]*h4/h5*delta_fx[i]*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                    if(d_y1[i][nr] != 0.0) f_global[at_id_r[k]][1] += (ran*epsilon/(1.0+sqrt(pow(d_y1[i][nr],2))))*alpha_2_y[i][nr]*h4/h5*delta_fy[i]*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                    if(d_z1[i][nr] != 0.0) f_global[at_id_r[k]][2] += (ran*epsilon/(1.0+sqrt(pow(d_z1[i][nr],2))))*alpha_2_z[i][nr]*h4/h5*delta_fz[i]*(distance_restraint[k]-d_t)/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;                                    
                               
                                    if(d_x1[i][nr] != 0.0) alpha_3_x[i] += (ran*epsilon_md/(1.0+sqrt(pow(d_x1[i][nr],2))))*alpha_2_x[i][nr];
                                    if(d_y1[i][nr] != 0.0) alpha_3_y[i] += (ran*epsilon_md/(1.0+sqrt(pow(d_y1[i][nr],2))))*alpha_2_y[i][nr];                 
                                    if(d_z1[i][nr] != 0.0) alpha_3_z[i] += (ran*epsilon_md/(1.0+sqrt(pow(d_z1[i][nr],2))))*alpha_2_z[i][nr];                                      
                                    
                                    };
                                  };
                                  
                             if(i == at_id_r2[k]) {
                                 
                                  d_x = state_global->x[at_id_r2[k]][0] - state_global->x[at_id_r[k]][0];
                                  d_y = state_global->x[at_id_r2[k]][1] - state_global->x[at_id_r[k]][1];
                                  d_z = state_global->x[at_id_r2[k]][2] - state_global->x[at_id_r[k]][2];

                                  d_t = sqrt(pow(d_x,2)+pow(d_y,2)+pow(d_z,2));
                                  
                                  d_t2 = sqrt(pow(delta_fx[i],2)+pow(delta_fy[i],2)+pow(delta_fz[i],2));
                                  
                                  if(d_t*d_t2 > 0.0) cos_val = (d_x*delta_fx[i] + d_y*delta_fy[i] + d_z*delta_fz[i])/(d_t*d_t2); 
                                  
                                  if(distance_restraint[k]-d_t != 0.0) { 
                                  
                                    ran = rand()%100/100.0;  
                                      
                                    if(d_x1[i][nr] != 0.0) f_global[at_id_r2[k]][0] += (ran*epsilon/(1.0+sqrt(pow(d_x1[i][nr],2))))*alpha_2_x[i][nr]*h4/h5*delta_fx[i]*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                    if(d_y1[i][nr] != 0.0) f_global[at_id_r2[k]][1] += (ran*epsilon/(1.0+sqrt(pow(d_y1[i][nr],2))))*alpha_2_y[i][nr]*h4/h5*delta_fy[i]*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;
                                    if(d_z1[i][nr] != 0.0) f_global[at_id_r2[k]][2] += (ran*epsilon/(1.0+sqrt(pow(d_z1[i][nr],2))))*alpha_2_z[i][nr]*h4/h5*delta_fz[i]*(d_t-distance_restraint[k])/sqrt(pow(distance_restraint[k]-d_t,2))*cos_val;  
                                    
                                    if(d_x1[i][nr] != 0.0) alpha_3_x[i] += (ran*epsilon_md/(1.0+sqrt(pow(d_x1[i][nr],2))))*alpha_2_x[i][nr];
                                    if(d_y1[i][nr] != 0.0) alpha_3_y[i] += (ran*epsilon_md/(1.0+sqrt(pow(d_y1[i][nr],2))))*alpha_2_y[i][nr];                 
                                    if(d_z1[i][nr] != 0.0) alpha_3_z[i] += (ran*epsilon_md/(1.0+sqrt(pow(d_z1[i][nr],2))))*alpha_2_z[i][nr];                                     
                               
                                  };
                              };                              
                                  
                            };
                          };
                         };
                    };
                 
                 if(h5 > 0.0) {                 
               
                    f_global[at_id[i]][0] = 1.0/(1.0+alpha_3_x[i])*f_global[at_id[i]][0];
                    f_global[at_id[i]][1] = 1.0/(1.0+alpha_3_y[i])*f_global[at_id[i]][1];
                    f_global[at_id[i]][2] = 1.0/(1.0+alpha_3_z[i])*f_global[at_id[i]][2];

                    }; 
                   }; 
                  };
               };
               
           // implicit solvent vectors
        
      if(timeframe_sol > 0.0) {       
           
         double vec_av_dL_x,vec_av_dL_y,vec_av_dL_z;
         double vec_av_dL_x2,vec_av_dL_y2,vec_av_dL_z2;
         double var_x,var_y,var_z;
         double cos_dL_av;         
            
           vec_av_dL_x = 0.0;
           vec_av_dL_y = 0.0;
           vec_av_dL_z = 0.0;  
    
           
    if(stepper == 1) {
      
      for(i=1;i<=natoms1;i++) {  
        
        d_t0 += sqrt(pow(f_global[at_id[i]][0],2)+pow(f_global[at_id[i]][1],2)+pow(f_global[at_id[i]][2],2));       
        
        
      };
      
      d_t0 /= (double)natoms1;
      
    };
    
   double ran0,ran1,ran2; 
        
    if(stepper%timeframe_sol == 0) {
     
       for(i=1;i<=natoms1;i++){
           
           df_x_imp[i] = 0.0;
           df_y_imp[i] = 0.0;
           df_z_imp[i] = 0.0;
           
           vec_dL_x2[i] = 0.0;
           vec_dL_y2[i] = 0.0;
           vec_dL_z2[i] = 0.0;           
           
         for(nr=1;nr<=number_of_replicas;nr++) {  
             
           if(stepper%tau2[i][nr] == 0) {  
             
             df_x_imp[i] = df_x_s[i][nr];
             df_y_imp[i] = df_y_s[i][nr];
             df_z_imp[i] = df_z_s[i][nr];
             
             vec_dL_x2[i] = (f_x2_s[i] + (f_x2_s[i]-f_x1_s[i]))/ir->delta_t*d_x1[i][nr];
             vec_dL_y2[i] = (f_y2_s[i] + (f_y2_s[i]-f_y1_s[i]))/ir->delta_t*d_y1[i][nr];
             vec_dL_z2[i] = (f_z2_s[i] + (f_z2_s[i]-f_z1_s[i]))/ir->delta_t*d_z1[i][nr];
          
           };
             
             d_t = sqrt(pow(df_x_imp[i]+vec_dL_x2[i],2)+pow(df_y_imp[i]+vec_dL_y2[i],2)+pow(df_z_imp[i]+vec_dL_z2[i],2));
             d_t2 = sqrt(pow(f_global[at_id[i]][0],2)+pow(f_global[at_id[i]][1],2)+pow(f_global[at_id[i]][2],2));
             
             if(d_t > 0.0) {  
               
               ran = rand()%100/100.0;
               
               f_global[at_id[i]][0] += (d_t2/d_t)*(beta2*epsilon_sol*ran)*(df_x_imp[i]+vec_dL_x2[i]);              
               f_global[at_id[i]][1] += (d_t2/d_t)*(beta2*epsilon_sol*ran)*(df_y_imp[i]+vec_dL_y2[i]);          
               f_global[at_id[i]][2] += (d_t2/d_t)*(beta2*epsilon_sol*ran)*(df_z_imp[i]+vec_dL_z2[i]);
               
             };
               
               d_x = rand()%100/100.0;
               d_y = rand()%100/100.0;
               d_z = rand()%100/100.0;
               
               d_t = sqrt(pow(d_x,2)+pow(d_y,2)+pow(d_z,2));
               
               ran = rand()%100/100.0;    
               
             if(d_t > 0.0) {  
               
               f_global[at_id[i]][0] += ((d_t0/d_t))*(epsilon_sol*beta2*ran)*(d_x);
               f_global[at_id[i]][1] += ((d_t0/d_t))*(epsilon_sol*beta2*ran)*(d_y);                
               f_global[at_id[i]][2] += ((d_t0/d_t))*(epsilon_sol*beta2*ran)*(d_z);               

               };
          };
         };
        };
       };
      }; // if tau2_a > 0       
       
    };   // MASTER(cr)      
	     
            if(DOMAINDECOMP(cr)) dd_distribute_vec(cr->dd,dd_charge_groups_global(cr->dd),f_global,f);
	    
            MPI_Barrier(cr->mpi_comm_mygroup);	

        /* ########  END FIRST UPDATE STEP  ############## */
        /* ########  If doing VV, we now have v(dt) ###### */

        /* ################## START TRAJECTORY OUTPUT ################# */

        /* Now we have the energies and forces corresponding to the 
         * coordinates at time t. We must output all of this before
         * the update.
         * for RerunMD t is read from input trajectory
         */
        GMX_MPE_LOG(ev_output_start);
	
        mdof_flags = 0;
        if (do_per_step(step,ir->nstxout)) { mdof_flags |= MDOF_X; }
        if (do_per_step(step,ir->nstvout)) { mdof_flags |= MDOF_V; }
        if (do_per_step(step,ir->nstfout)) { mdof_flags |= MDOF_F; }
        if (do_per_step(step,ir->nstxtcout)) { mdof_flags |= MDOF_XTC; }
        if (bCPT) { mdof_flags |= MDOF_CPT; };

#if defined(GMX_FAHCORE) || defined(GMX_WRITELASTSTEP)
        if (bLastStep)
        {
            /* Enforce writing positions and velocities at end of run */
            mdof_flags |= (MDOF_X | MDOF_V);
        }
#endif
#ifdef GMX_FAHCORE
        if (MASTER(cr))
            fcReportProgress( ir->nsteps, step );

        /* sync bCPT and fc record-keeping */
        if (bCPT && MASTER(cr))
            fcRequestCheckPoint();
#endif
        
        if (mdof_flags != 0)
        {
            wallcycle_start(wcycle,ewcTRAJ);
            if (bCPT)
            {
                if (state->flags & (1<<estLD_RNG))
                {
                    get_stochd_state(upd,state);
                }
                if (MASTER(cr))
                {
                    if (bSumEkinhOld)
                    {
                        state_global->ekinstate.bUpToDate = FALSE;
                    }
                    else
                    {
                        update_ekinstate(&state_global->ekinstate,ekind);
                        state_global->ekinstate.bUpToDate = TRUE;
                    }
                    update_energyhistory(&state_global->enerhist,mdebin);
                }
            }
            write_traj(fplog,cr,outf,mdof_flags,top_global,
                       step,t,state,state_global,f,f_global,&n_xtc,&x_xtc);
            if (bCPT)
            {
                nchkpt++;
                bCPT = FALSE;
            }
            debug_gmx();
            if (bLastStep && step_rel == ir->nsteps &&
                (Flags & MD_CONFOUT) && MASTER(cr) &&
                !bRerunMD && !bFFscan)
            {
                /* x and v have been collected in write_traj,
                 * because a checkpoint file will always be written
                 * at the last step.
                 */
                fprintf(stderr,"\nWriting final coordinates.\n");
                if (ir->ePBC != epbcNONE && !ir->bPeriodicMols &&
                    DOMAINDECOMP(cr))
                {
                    /* Make molecules whole only for confout writing */
                    do_pbc_mtop(fplog,ir->ePBC,state->box,top_global,state_global->x);
                }
                write_sto_conf_mtop(ftp2fn(efSTO,nfile,fnm),
                                    *top_global->name,top_global,
                                    state_global->x,state_global->v,
                                    ir->ePBC,state->box);
                debug_gmx();
            }
            wallcycle_stop(wcycle,ewcTRAJ);
        }
        GMX_MPE_LOG(ev_output_finish);
        
        /* kludge -- virial is lost with restart for NPT control. Must restart */
        if (bStartingFromCpt && bVV) 
        {
            copy_mat(state->svir_prev,shake_vir);
            copy_mat(state->fvir_prev,force_vir);
        }
        /*  ################## END TRAJECTORY OUTPUT ################ */
        
        /* Determine the wallclock run time up till now */
        run_time = gmx_gettime() - (double)runtime->real;

        /* Check whether everything is still allright */    
        if (((int)gmx_get_stop_condition() > handled_stop_condition)
#ifdef GMX_THREADS
            && MASTER(cr)
#endif
            )
        {
            /* this is just make gs.sig compatible with the hack 
               of sending signals around by MPI_Reduce with together with
               other floats */
            if ( gmx_get_stop_condition() == gmx_stop_cond_next_ns )
                gs.sig[eglsSTOPCOND]=1;
            if ( gmx_get_stop_condition() == gmx_stop_cond_next )
                gs.sig[eglsSTOPCOND]=-1;
            /* < 0 means stop at next step, > 0 means stop at next NS step */
            if (fplog)
            {
                fprintf(fplog,
                        "\n\nReceived the %s signal, stopping at the next %sstep\n\n",
                        gmx_get_signal_name(),
                        gs.sig[eglsSTOPCOND]==1 ? "NS " : "");
                fflush(fplog);
            }
            fprintf(stderr,
                    "\n\nReceived the %s signal, stopping at the next %sstep\n\n",
                    gmx_get_signal_name(),
                    gs.sig[eglsSTOPCOND]==1 ? "NS " : "");
            fflush(stderr);
            handled_stop_condition=(int)gmx_get_stop_condition();
        }
        else if (MASTER(cr) && (bNS || ir->nstlist <= 0) &&
                 (max_hours > 0 && run_time > max_hours*60.0*60.0*0.99) &&
                 gs.sig[eglsSTOPCOND] == 0 && gs.set[eglsSTOPCOND] == 0)
        {
            /* Signal to terminate the run */
            gs.sig[eglsSTOPCOND] = 1;
            if (fplog)
            {
                fprintf(fplog,"\nStep %s: Run time exceeded %.3f hours, will terminate the run\n",gmx_step_str(step,sbuf),max_hours*0.99);
            }
            fprintf(stderr, "\nStep %s: Run time exceeded %.3f hours, will terminate the run\n",gmx_step_str(step,sbuf),max_hours*0.99);
        }

        if (bResetCountersHalfMaxH && MASTER(cr) &&
            run_time > max_hours*60.0*60.0*0.495)
        {
            gs.sig[eglsRESETCOUNTERS] = 1;
        }

        if (ir->nstlist == -1 && !bRerunMD)
        {
            /* When bGStatEveryStep=FALSE, global_stat is only called
             * when we check the atom displacements, not at NS steps.
             * This means that also the bonded interaction count check is not
             * performed immediately after NS. Therefore a few MD steps could
             * be performed with missing interactions.
             * But wrong energies are never written to file,
             * since energies are only written after global_stat
             * has been called.
             */
            if (step >= nlh.step_nscheck)
            {
                nlh.nabnsb = natoms_beyond_ns_buffer(ir,fr,&top->cgs,
                                                     nlh.scale_tot,state->x);
            }
            else
            {
                /* This is not necessarily true,
                 * but step_nscheck is determined quite conservatively.
                 */
                nlh.nabnsb = 0;
            }
        }

        /* In parallel we only have to check for checkpointing in steps
         * where we do global communication,
         *  otherwise the other nodes don't know.
         */
        if (MASTER(cr) && ((bGStat || !PAR(cr)) &&
                           cpt_period >= 0 &&
                           (cpt_period == 0 || 
                            run_time >= nchkpt*cpt_period*60.0)) &&
            gs.set[eglsCHKPT] == 0)
        {
            gs.sig[eglsCHKPT] = 1;
        }
  
        if (bIterations)
        {
            gmx_iterate_init(&iterate,bIterations);
        }
    
        /* for iterations, we save these vectors, as we will be redoing the calculations */
        if (bIterations && iterate.bIterate) 
        {
            copy_coupling_state(state,bufstate,ekind,ekind_save,&(ir->opts));
        }
        bFirstIterate = TRUE;
        while (bFirstIterate || (bIterations && iterate.bIterate))
        {
            /* We now restore these vectors to redo the calculation with improved extended variables */    
            if (bIterations) 
            { 
                copy_coupling_state(bufstate,state,ekind_save,ekind,&(ir->opts));
            }

            /* We make the decision to break or not -after- the calculation of Ekin and Pressure,
               so scroll down for that logic */
            
            /* #########   START SECOND UPDATE STEP ################# */
            GMX_MPE_LOG(ev_update_start);
            /* Box is changed in update() when we do pressure coupling,
             * but we should still use the old box for energy corrections and when
             * writing it to the energy file, so it matches the trajectory files for
             * the same timestep above. Make a copy in a separate array.
             */
            copy_mat(state->box,lastbox);

            bOK = TRUE;
            if (!(bRerunMD && !rerun_fr.bV && !bForceUpdate))
            {
                wallcycle_start(wcycle,ewcUPDATE);
                dvdl = 0;
                /* UPDATE PRESSURE VARIABLES IN TROTTER FORMULATION WITH CONSTRAINTS */
                if (bTrotter) 
                {
                    if (bIterations && iterate.bIterate) 
                    {
                        if (bFirstIterate) 
                        {
                            scalevir = 1;
                        }
                        else 
                        {
                            /* we use a new value of scalevir to converge the iterations faster */
                            scalevir = tracevir/trace(shake_vir);
                        }
                        msmul(shake_vir,scalevir,shake_vir); 
                        m_add(force_vir,shake_vir,total_vir);
                        clear_mat(shake_vir);
                    }
                    trotter_update(ir,step,ekind,enerd,state,total_vir,mdatoms,&MassQ,trotter_seq,ettTSEQ3);
                /* We can only do Berendsen coupling after we have summed
                 * the kinetic energy or virial. Since the happens
                 * in global_state after update, we should only do it at
                 * step % nstlist = 1 with bGStatEveryStep=FALSE.
                 */
                }
                else 
                {
                    update_tcouple(fplog,step,ir,state,ekind,wcycle,upd,&MassQ,mdatoms);
                    update_pcouple(fplog,step,ir,state,pcoupl_mu,M,wcycle,
                                   upd,bInitStep);
                }

                if (bVV)
                {
                    /* velocity half-step update */
                    update_coords(fplog,step,ir,mdatoms,state,f,
                                  fr->bTwinRange && bNStList,fr->f_twin,fcd,
                                  ekind,M,wcycle,upd,FALSE,etrtVELOCITY2,
                                  cr,nrnb,constr,&top->idef);
                }

                /* Above, initialize just copies ekinh into ekin,
                 * it doesn't copy position (for VV),
                 * and entire integrator for MD.
                 */
                
                if (ir->eI==eiVVAK) 
                {
                    copy_rvecn(state->x,cbuf,0,state->natoms);
                }
                
                update_coords(fplog,step,ir,mdatoms,state,f,fr->bTwinRange && bNStList,fr->f_twin,fcd,
                              ekind,M,wcycle,upd,bInitStep,etrtPOSITION,cr,nrnb,constr,&top->idef);
                wallcycle_stop(wcycle,ewcUPDATE);

                update_constraints(fplog,step,&dvdl,ir,ekind,mdatoms,state,graph,f,
                                   &top->idef,shake_vir,force_vir,
                                   cr,nrnb,wcycle,upd,constr,
                                   bInitStep,FALSE,bCalcEnerPres,state->veta);  
                
                if (ir->eI==eiVVAK) 
                {
                    /* erase F_EKIN and F_TEMP here? */
                    /* just compute the kinetic energy at the half step to perform a trotter step */
                    compute_globals(fplog,gstat,cr,ir,fr,ekind,state,state_global,mdatoms,nrnb,vcm,
                                    wcycle,enerd,force_vir,shake_vir,total_vir,pres,mu_tot,
                                    constr,NULL,FALSE,lastbox,
                                    top_global,&pcurr,top_global->natoms,&bSumEkinhOld,
                                    cglo_flags | CGLO_TEMPERATURE    
                        );
                    wallcycle_start(wcycle,ewcUPDATE);
                    trotter_update(ir,step,ekind,enerd,state,total_vir,mdatoms,&MassQ,trotter_seq,ettTSEQ4);            
                    /* now we know the scaling, we can compute the positions again again */
                    copy_rvecn(cbuf,state->x,0,state->natoms);

                    update_coords(fplog,step,ir,mdatoms,state,f,fr->bTwinRange && bNStList,fr->f_twin,fcd,
                                  ekind,M,wcycle,upd,bInitStep,etrtPOSITION,cr,nrnb,constr,&top->idef);
                    wallcycle_stop(wcycle,ewcUPDATE);

                    /* do we need an extra constraint here? just need to copy out of state->v to upd->xp? */
                    /* are the small terms in the shake_vir here due
                     * to numerical errors, or are they important
                     * physically? I'm thinking they are just errors, but not completely sure. 
                     * For now, will call without actually constraining, constr=NULL*/
                    update_constraints(fplog,step,&dvdl,ir,ekind,mdatoms,state,graph,f,
                                       &top->idef,tmp_vir,force_vir,
                                       cr,nrnb,wcycle,upd,NULL,
                                       bInitStep,FALSE,bCalcEnerPres,
                                       state->veta);  
                }
                if (!bOK && !bFFscan) 
                {
                    gmx_fatal(FARGS,"Constraint error: Shake, Lincs or Settle could not solve the constrains");
                }
                
                if (fr->bSepDVDL && fplog && do_log) 
                {
                    fprintf(fplog,sepdvdlformat,"Constraint",0.0,dvdl);
                }
                enerd->term[F_DHDL_CON] += dvdl;
            } 
            else if (graph) 
            {
                /* Need to unshift here */
                unshift_self(graph,state->box,state->x);
            }
            
            GMX_BARRIER(cr->mpi_comm_mygroup);
            GMX_MPE_LOG(ev_update_finish);

            if (vsite != NULL) 
            {
                wallcycle_start(wcycle,ewcVSITECONSTR);
                if (graph != NULL) 
                {
                    shift_self(graph,state->box,state->x);
                }
                construct_vsites(fplog,vsite,state->x,nrnb,ir->delta_t,state->v,
                                 top->idef.iparams,top->idef.il,
                                 fr->ePBC,fr->bMolPBC,graph,cr,state->box);
                
                if (graph != NULL) 
                {
                    unshift_self(graph,state->box,state->x);
                }
                wallcycle_stop(wcycle,ewcVSITECONSTR);
            }
            
            /* ############## IF NOT VV, Calculate globals HERE, also iterate constraints ############ */
            if (ir->nstlist == -1 && bFirstIterate)
            {
                gs.sig[eglsNABNSB] = nlh.nabnsb;
            }
            compute_globals(fplog,gstat,cr,ir,fr,ekind,state,state_global,mdatoms,nrnb,vcm,
                            wcycle,enerd,force_vir,shake_vir,total_vir,pres,mu_tot,
                            constr,
                            bFirstIterate ? &gs : NULL, 
                            (step_rel % gs.nstms == 0) && 
                                (multisim_nsteps<0 || (step_rel<multisim_nsteps)),
                            lastbox,
                            top_global,&pcurr,top_global->natoms,&bSumEkinhOld,
                            cglo_flags 
                            | (!EI_VV(ir->eI) ? CGLO_ENERGY : 0) 
                            | (!EI_VV(ir->eI) ? CGLO_TEMPERATURE : 0) 
                            | (!EI_VV(ir->eI) || bRerunMD ? CGLO_PRESSURE : 0) 
                            | (bIterations && iterate.bIterate ? CGLO_ITERATE : 0) 
                            | (bFirstIterate ? CGLO_FIRSTITERATE : 0)
                            | CGLO_CONSTRAINT 
                );
            if (ir->nstlist == -1 && bFirstIterate)
            {
                nlh.nabnsb = gs.set[eglsNABNSB];
                gs.set[eglsNABNSB] = 0;
            }
            /* bIterate is set to keep it from eliminating the old ekin kinetic energy terms */
            /* #############  END CALC EKIN AND PRESSURE ################# */
        
            /* Note: this is OK, but there are some numerical precision issues with using the convergence of
               the virial that should probably be addressed eventually. state->veta has better properies,
               but what we actually need entering the new cycle is the new shake_vir value. Ideally, we could
               generate the new shake_vir, but test the veta value for convergence.  This will take some thought. */

            if (bIterations && 
                done_iterating(cr,fplog,step,&iterate,bFirstIterate,
                               trace(shake_vir),&tracevir)) 
            {
                break;
            }
            bFirstIterate = FALSE;
        }

        update_box(fplog,step,ir,mdatoms,state,graph,f,
                   ir->nstlist==-1 ? &nlh.scale_tot : NULL,pcoupl_mu,nrnb,wcycle,upd,bInitStep,FALSE);
        
        /* ################# END UPDATE STEP 2 ################# */
        /* #### We now have r(t+dt) and v(t+dt/2)  ############# */
    
        /* The coordinates (x) were unshifted in update */
        if (bFFscan && (shellfc==NULL || bConverged))
        {
            if (print_forcefield(fplog,enerd->term,mdatoms->homenr,
                                 f,NULL,xcopy,
                                 &(top_global->mols),mdatoms->massT,pres))
            {
                if (gmx_parallel_env_initialized())
                {
                    gmx_finalize();
                }
                fprintf(stderr,"\n");
                exit(0);
            }
        }
        if (!bGStat)
        {
            /* We will not sum ekinh_old,                                                            
             * so signal that we still have to do it.                                                
             */
            bSumEkinhOld = TRUE;
        }
        
        if (bTCR)
        {
            /* Only do GCT when the relaxation of shells (minimization) has converged,
             * otherwise we might be coupling to bogus energies. 
             * In parallel we must always do this, because the other sims might
             * update the FF.
             */

            /* Since this is called with the new coordinates state->x, I assume
             * we want the new box state->box too. / EL 20040121
             */
            do_coupling(fplog,oenv,nfile,fnm,tcr,t,step,enerd->term,fr,
                        ir,MASTER(cr),
                        mdatoms,&(top->idef),mu_aver,
                        top_global->mols.nr,cr,
                        state->box,total_vir,pres,
                        mu_tot,state->x,f,bConverged);
            debug_gmx();
        }

        /* #########  BEGIN PREPARING EDR OUTPUT  ###########  */
        
        /* sum up the foreign energy and dhdl terms */
        sum_dhdl(enerd,state->lambda,ir);

        /* use the directly determined last velocity, not actually the averaged half steps */
        if (bTrotter && ir->eI==eiVV) 
        {
            enerd->term[F_EKIN] = last_ekin;
        }
        enerd->term[F_ETOT] = enerd->term[F_EPOT] + enerd->term[F_EKIN];
        
        if (bVV)
        {
            enerd->term[F_ECONSERVED] = enerd->term[F_ETOT] + saved_conserved_quantity;
        }
        else 
        {
            enerd->term[F_ECONSERVED] = enerd->term[F_ETOT] + compute_conserved_from_auxiliary(ir,state,&MassQ);
        }
        /* Check for excessively large energies */
        if (bIonize) 
        {
#ifdef GMX_DOUBLE
            real etot_max = 1e200;
#else
            real etot_max = 1e30;
#endif
            if (fabs(enerd->term[F_ETOT]) > etot_max) 
            {
                fprintf(stderr,"Energy too large (%g), giving up\n",
                        enerd->term[F_ETOT]);
            }
        }
        /* #########  END PREPARING EDR OUTPUT  ###########  */
        
        /* Time for performance */
        if (((step % stepout) == 0) || bLastStep) 
        {
            runtime_upd_proc(runtime);
        }
        
        /* Output stuff */
        if (MASTER(cr))
        {
            gmx_bool do_dr,do_or;
            
            if (!(bStartingFromCpt && (EI_VV(ir->eI)))) 
            {
                if (bNstEner)
                {
                    upd_mdebin(mdebin,bDoDHDL, TRUE,
                               t,mdatoms->tmass,enerd,state,lastbox,
                               shake_vir,force_vir,total_vir,pres,
                               ekind,mu_tot,constr);
                }
                else
                {
                    upd_mdebin_step(mdebin);
                }
                
                do_dr  = do_per_step(step,ir->nstdisreout);
                do_or  = do_per_step(step,ir->nstorireout);
                
                print_ebin(outf->fp_ene,do_ene,do_dr,do_or,do_log?fplog:NULL,
                           step,t,
                           eprNORMAL,bCompact,mdebin,fcd,groups,&(ir->opts));
            }
            if (ir->ePull != epullNO)
            {
                pull_print_output(ir->pull,step,t);
            }
            
            if (do_per_step(step,ir->nstlog))
            {
                if(fflush(fplog) != 0)
                {
                    gmx_fatal(FARGS,"Cannot flush logfile - maybe you are out of quota?");
                }
            }
        }


        /* Remaining runtime */
        if (MULTIMASTER(cr) && (do_verbose || gmx_got_usr_signal() ))
        {
            if (shellfc) 
            {
                fprintf(stderr,"\n");
            }
            print_time(stderr,runtime,step,ir,cr);
        }

        /* Replica exchange */
        bExchanged = FALSE;
        if ((repl_ex_nst > 0) && (step > 0) && !bLastStep &&
            do_per_step(step,repl_ex_nst)) 
        {
            
            rho_sys /= counter_exchange;
            
            counter_exchange = 0.0;
            sum_dL           = 0.0;  
           
            
            bExchanged = replica_exchange(fplog,cr,repl_ex,
                                          state_global,enerd->term,
                                          state,step,t,rho_sys,natoms1,
                                          number_of_replicas,df_x_s,df_y_s,df_z_s,
                                          grad_meta_x,grad_meta_y,grad_meta_z);
            
            
            rho_sys = 0.0;            

            if (bExchanged && DOMAINDECOMP(cr)) 
            {
                dd_partition_system(fplog,step,cr,TRUE,1,
                                    state_global,top_global,ir,
                                    state,&f,mdatoms,top,fr,
                                    vsite,shellfc,constr,
                                    nrnb,wcycle,FALSE);
            }
        }
        
        bFirstStep = FALSE;
        bInitStep = FALSE;
        bStartingFromCpt = FALSE;

        /* #######  SET VARIABLES FOR NEXT ITERATION IF THEY STILL NEED IT ###### */
        /* With all integrators, except VV, we need to retain the pressure
         * at the current step for coupling at the next step.
         */
        if ((state->flags & (1<<estPRES_PREV)) &&
            (bGStatEveryStep ||
             (ir->nstpcouple > 0 && step % ir->nstpcouple == 0)))
        {
            /* Store the pressure in t_state for pressure coupling
             * at the next MD step.
             */
            copy_mat(pres,state->pres_prev);
        }
        
        /* #######  END SET VARIABLES FOR NEXT ITERATION ###### */
        
        if (bRerunMD) 
        {
            if (MASTER(cr))
            {
                /* read next frame from input trajectory */
                bNotLastFrame = read_next_frame(oenv,status,&rerun_fr);
            }

            if (PAR(cr))
            {
                rerun_parallel_comm(cr,&rerun_fr,&bNotLastFrame);
            }
        }
        
        if (!bRerunMD || !rerun_fr.bStep)
        {
            /* increase the MD step number */
            step++;
            step_rel++;
        }
        
        cycles = wallcycle_stop(wcycle,ewcSTEP);
        if (DOMAINDECOMP(cr) && wcycle)
        {
            dd_cycles_add(cr->dd,cycles,ddCyclStep);
        }
        
        if (step_rel == wcycle_get_reset_counters(wcycle) ||
            gs.set[eglsRESETCOUNTERS] != 0)
        {
            /* Reset all the counters related to performance over the run */
            reset_all_counters(fplog,cr,step,&step_rel,ir,wcycle,nrnb,runtime);
            wcycle_set_reset_counters(wcycle,-1);
            /* Correct max_hours for the elapsed time */
            max_hours -= run_time/(60.0*60.0);
            bResetCountersHalfMaxH = FALSE;
            gs.set[eglsRESETCOUNTERS] = 0;
        }

    }
    /* End of main MD loop */
    debug_gmx();
    
    /* Stop the time */
    runtime_end(runtime);
    
    if (bRerunMD && MASTER(cr))
    {
        close_trj(status);
    }
    
    if (!(cr->duty & DUTY_PME))
    {
        /* Tell the PME only node to finish */
        gmx_pme_finish(cr);
    }
    
    if (MASTER(cr))
    {
        if (ir->nstcalcenergy > 0 && !bRerunMD) 
        {
            print_ebin(outf->fp_ene,FALSE,FALSE,FALSE,fplog,step,t,
                       eprAVER,FALSE,mdebin,fcd,groups,&(ir->opts));
        }
    }

    done_mdoutf(outf);

    debug_gmx();

    if (ir->nstlist == -1 && nlh.nns > 0 && fplog)
    {
        fprintf(fplog,"Average neighborlist lifetime: %.1f steps, std.dev.: %.1f steps\n",nlh.s1/nlh.nns,sqrt(nlh.s2/nlh.nns - sqr(nlh.s1/nlh.nns)));
        fprintf(fplog,"Average number of atoms that crossed the half buffer length: %.1f\n\n",nlh.ab/nlh.nns);
    }
    
    if (shellfc && fplog)
    {
        fprintf(fplog,"Fraction of iterations that converged:           %.2f %%\n",
                (nconverged*100.0)/step_rel);
        fprintf(fplog,"Average number of force evaluations per MD step: %.2f\n\n",
                tcount/step_rel);
    }
    
    if (repl_ex_nst > 0 && MASTER(cr))
    {
        print_replica_exchange_statistics(fplog,repl_ex);
    }
    
    runtime->nsteps_done = step_rel;

    return 0;
}

int random_number(float *x)

{

/*   int seed;
  
   seed = time(NULL);
   srand(seed);  
   
      
   srand( (unsigned)time( NULL ) ); */
   *x = (float) rand()  / (float) RAND_MAX;

   if(*x <= 0.09) *x = *x * 10;

   if(*x <= 0.009) *x = *x * 100;

   if(*x <= 0.0009) *x = *x * 1000;

   if(*x <= 0.00009) *x = *x * 10000;

   if(*x <= 0.000009) *x = *x * 100000;

   if(*x <= 0.0000009) *x = *x * 1000000;

   return 1;

}

int
eigensolver(real *   a,
            int      n,
            int      index_lower,
            int      index_upper,
            real *   eigenvalues,
            real *   eigenvectors)
{
    int *   isuppz;
    int     lwork,liwork;
    int     il,iu,m,iw0,info;
    real    w0,abstol;
    int *   iwork;
    real *  work;
    real    vl,vu;
    const char *  jobz;
    
    if(index_lower<0)
        index_lower = 0;
    
    if(index_upper>=n)
        index_upper = n-1;
    
    /* Make jobz point to the character "V" if eigenvectors
     * should be calculated, otherwise "N" (only eigenvalues).
     */   
    jobz = (eigenvectors != NULL) ? "V" : "N";

    /* allocate lapack stuff */
    snew(isuppz,2*n);
    vl = vu = 0;
    
    /* First time we ask the routine how much workspace it needs */
    lwork  = -1;
    liwork = -1;
    abstol = 0;
    
    /* Convert indices to fortran standard */
    index_lower++;
    index_upper++;
    
    /* Call LAPACK routine using fortran interface. Note that we use upper storage,
     * but this corresponds to lower storage ("L") in Fortran.
     */    
#ifdef GMX_DOUBLE
    F77_FUNC(dsyevr,DSYEVR)(jobz,"I","L",&n,a,&n,&vl,&vu,&index_lower,&index_upper,
                            &abstol,&m,eigenvalues,eigenvectors,&n,
                            isuppz,&w0,&lwork,&iw0,&liwork,&info);
#else
    F77_FUNC(ssyevr,SSYEVR)(jobz,"I","L",&n,a,&n,&vl,&vu,&index_lower,&index_upper,
                            &abstol,&m,eigenvalues,eigenvectors,&n,
                            isuppz,&w0,&lwork,&iw0,&liwork,&info);
#endif

/*    if(info != 0)
    {
        sfree(isuppz);
        gmx_fatal(FARGS,"Internal errror in LAPACK diagonalization.");        
    } */
    
    lwork = w0;
    liwork = iw0;
    
    snew(work,lwork);
    snew(iwork,liwork);
    
    abstol = 0;
    
#ifdef GMX_DOUBLE
    F77_FUNC(dsyevr,DSYEVR)(jobz,"I","L",&n,a,&n,&vl,&vu,&index_lower,&index_upper,
                            &abstol,&m,eigenvalues,eigenvectors,&n,
                            isuppz,work,&lwork,iwork,&liwork,&info);
#else
    F77_FUNC(ssyevr,SSYEVR)(jobz,"I","L",&n,a,&n,&vl,&vu,&index_lower,&index_upper,
                            &abstol,&m,eigenvalues,eigenvectors,&n,
                            isuppz,work,&lwork,iwork,&liwork,&info);
#endif
    
    sfree(isuppz);
    sfree(work);
    sfree(iwork);
    
   /* if(info != 0)
    {
        gmx_fatal(FARGS,"Internal errror in LAPACK diagonalization.");
    } */
    
    return info;
   
}

