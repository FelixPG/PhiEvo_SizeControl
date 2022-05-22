#define SIZE 6
#define NINTER 6
#define NSTEP 50000
#define NCELLTOT 1
#define NNEIGHBOR 3
#define NTRIES 10
#define  CONCENTRATION_SCALE 0.000000
#define GENERATION 0
#define NFREE_PRMT 0
static double free_prmt[] = {}; 
#define	NOUTPUT 1
#define	NINPUT 2
#define	NLIGAND 0
#define	NDIFFUSIBLE 0
#define SEED 96071
#define PRINT_BUF 0
#define DT 0.050000
static int trackin[] = {0, 1};
static int trackout[] = {2};
static int tracklig[] = {};
static int trackdiff[] = {};
static double diff_constant[] = {};
static int externallig[] = {};



/********  header file, arrays and functions common to all following subroutines and
  the same for all problems.  NOTE this file preceeded by compiler directives defining
  parameters and the track*[] arrays
********/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <limits.h>

/* global arrays for history and geometry */

static double history[SIZE][NSTEP][NCELLTOT];
static int geometry[NCELLTOT][NNEIGHBOR];

double MAX(double a,double b){
	 if (a>b) return a;
 	 return b;
}
 
double FRAND()  {
	return (double) rand()/((double)RAND_MAX + 1);
}
 
double MIN( double a, double b ) {
	if (a<b) return a;
	return b;
}



double POW(double x,double n){
  return exp(n*log(x));
}





double HillR(double x,double thresh,double n)
{
	double r=exp(n*log(x/thresh));
	return 1.0/(1+r);
}
 
 
double HillA(double x,double thresh,double n)
{
	double r=exp(n*log(x/thresh));
	return r/(1+r);
 }


/* Function to compute diffusion of ligands.Geometry encodes the neighbouring cells, i.e. 
geometry[i][] contains the indexes of the neighbours of cell i.  The table of diffusion
csts, difflig[] is static global variable in header
*/

void diffusion(int ncell, int step, double ds[],double history[][NSTEP][NCELLTOT], int geometry[][NNEIGHBOR]){

  int g,neig,index,index_diff,n_localneig;
  double diff,diffusion;
 
 
  for (g=0;g<NDIFFUSIBLE;g++){
 
      diffusion=0;
      index_diff=trackdiff[g];//tracks the diffusible species
      diff=diff_constant[g];//takes the corresponding diffusion constant
      n_localneig=0;//computes number of local neighbours
      for (neig=0;neig<NNEIGHBOR;neig++){
	index=geometry[ncell][neig];//takes the neighoubring cell
	if (index>=0){
	  diffusion+=history[index_diff][step][index];//concentration of the ligand in the neighbouring cell
	  n_localneig+=1;
	}
      }
      diffusion-=n_localneig*history[index_diff][step][ncell];//minus number of local neighbours times concentration of the ligand in the local cell
      diffusion*=diff;// times diffusion constant
      ds[index_diff]+=diffusion;
    }
  }


/* 
computation of the ligand concentration seen by each cells
*/




void sum_concentration(int ncell, int step, double concentrations[]) {

  int l,g,neig,index;
  // compute local ligands concentrations, returns the table "concentrations" 
  for (l=0;l<NLIGAND;l++){
    
  if (externallig[l]==1){
      concentrations[tracklig[l]]=history[tracklig[l]][step][ncell];   // For external ligands, we simply take the local external value as local ligand concentration
  }
  else
    {//in that case, we sum ligand concentration over all neighbouring cells
      g=tracklig[l];
      concentrations[g]=0;
      for (neig=0;neig<NNEIGHBOR;neig++){
	index=geometry[ncell][neig];
	if ((index>=0) && (index!=ncell) ) {//note that we exclude the local concentration of ligand from the sum
	  concentrations[g]+=history[g][step][index];  
	}
      }


    }

  }



}


/* print history array to file= BUFFER, where int trial is 0,1,..NTRIES-1 */

void print_history( int trial )  {

    int pas, i, j;
    char titre[50];
    FILE *fileptr;

    sprintf(titre, "Buffer%i", trial);
    fileptr=fopen(titre, "w");

    for(pas=0; pas<NSTEP; pas++)  {
	fprintf(fileptr,"%i", pas);
	for(j=0; j<NCELLTOT; j++)  {
	    for(i=0; i<SIZE; i++)  {
		fprintf(fileptr,"\t%f",history[i][pas][j]);
	    }
	}
	fprintf(fileptr,"\n");
    }
    fprintf(fileptr,"\n");
    fclose(fileptr);
}



/* statistical tools*/

/* Function to compute the mass dependent growth rate. 
*/ 
double mdg_rate(int pas, int ncell,double history[][NSTEP][NCELLTOT],int trackin[]){
  double growth_rate;
  double max_growth_rate = 0.25;
  if (history[trackin[0]][pas][ncell] <= 1){
    growth_rate = max_growth_rate*history[trackin[0]][pas][ncell];
  }
  else if (history[trackin[0]][pas][ncell] > 1 && history[trackin[0]][pas][ncell] <= 99){
    growth_rate = max_growth_rate;
  }
  else if (history[trackin[0]][pas][ncell] > 99 && history[trackin[0]][pas][ncell] <= 100){
    growth_rate = max_growth_rate*(100 - history[trackin[0]][pas][ncell]);
  }
  else {
    growth_rate = 0.0;
  }
  return growth_rate;

}

/* Function that averages the fitness scores*/
double average_score(double score[]){

  double average=0;
  int k;
  for (k=0;k<NTRIES;k++)
    average+=score[k];

  return average/NTRIES;


}


/* Function that computes the variance of fitness scores.*/
double variance_score(double score[]){

  double var=0;
  int k;
  for (k=0;k<NTRIES;k++)
    var+=score[k]*score[k];
  
  var/=NTRIES;
  double average=average_score(score);
  return var-average*average;


}



/* Function that computes the std_deviation*/
double std_dev(double score[]){

  double var=0;
  int k;
  for (k=0;k<NTRIES;k++)
    var+=score[k]*score[k];

  var/=NTRIES;
  double average=average_score(score);
  return sqrt(var-average*average);


}




double gaussdev()
{/*computes a normally distributed deviate with zero mean and unit variance
   using  Box-Muller method, Numerical Recipes C++ p293 */
	static int iset=0;
	static double gset;
	double fac,rsq,v1,v2;

	if (iset == 0) {
	  do {
	    v1=2.0*FRAND()-1.0;
	    v2=2.0*FRAND()-1.0;
	    rsq=v1*v1+v2*v2;
	  } while (rsq >= 1.0 || rsq == 0.0);
	  fac=sqrt(-2.0*log(rsq)/rsq);
	  gset=v1*fac;
	  iset=1;//set flag
	  return v2*fac;//returns one and keep other for nex time
	} 
	else {
	  iset=0;
	  return gset;
	}
}





double compute_noisy_increment(double rate,int step, int ncell)
{/*computes the increment to add to a ds*/

return rate+gaussdev()*sqrt(rate/(history[trackin[0]][step][ncell]*DT*CONCENTRATION_SCALE));


}
/***** end of header, begining of python computed functions ***/

void derivC(double s[],double history[][NSTEP][NCELLTOT],int step, double ds[],double memories[],int ncell){
 int index;	 for (index=0;index<SIZE;index++) ds[index]=0;//initialization
	 double increment=0;
	 double rate=0;

/**************degradation rates*****************/
	 	 rate=0.083827*s[2];
	 	 increment=rate;
	 	 ds[2]-=increment;
	 	 rate=0.4574764*s[3];
	 	 increment=rate;
	 	 ds[3]-=increment;
	 	 rate=4.244007*s[4];
	 	 increment=rate;
	 	 ds[4]-=increment;
	 	 rate=1.101876*s[5];
	 	 increment=rate;
	 	 ds[5]-=increment;

/**************Transcription rates*****************/
 	 int k,memory=-1;
	 memory=step-0;
	 if(memory>=0){
	 	 rate=MAX(0.369408*HillA(history[1][memory][ncell],1.104619,3.215727),0.000000);
	 	 increment=rate;
	 	 ds[2]+=increment;
	}
	 memory=step-0;
	 if(memory>=0){
	 	 rate=MAX(0.703658,0.000000)*HillR(history[2][memory][ncell],0.044938,3.266732);
	 	 increment=rate;
	 	 ds[3]+=increment;
	}

/**************Protein protein interactions*****************/
	 	 rate=6.783001 * s[2] * s[3];
	 	 increment=rate;
	 	 ds[2]-=increment;
	 	 ds[3]-=increment;
	 	 ds[4]+=increment;
	 	 rate=0.195168 * s[4];
	 	 increment=rate;
	 	 ds[4]-=increment;
	 	 ds[2]+=increment;
	 	 ds[3]+=increment;
	 	 rate=0.021174 * s[2] * s[2];
	 	 increment=rate;
	 	 ds[2]-=increment;
	 	 ds[2]-=increment;
	 	 ds[5]+=increment;
	 	 rate=1.075146 * s[5];
	 	 increment=rate;
	 	 ds[5]-=increment;
	 	 ds[2]+=increment;
	 	 ds[2]+=increment;

/**************Phosphorylation*****************/
 float total;
}

/***** end of python computed functions, beginning problem specific fns ***/

 /*

Defines fitness function
*/

#define NFUNCTIONS 2

static double result[NTRIES][NFUNCTIONS]; //result will contain the fitness plus the other results computed by the fitness function

void fitness( double history[][NSTEP][NCELLTOT],int trackin[],int trial)  {
  // MODIFY HERE to compute a fitness from the history array
  int pas,ncell;
  int n_div = 0;
  int dead = 0;
  int maxv = 0;
  
  double diff_cell_cycle = 0; // Cell cycle events

  // Compute statistics of the distributions

  double sumx_birth = 0;
  double sumx_start = 0;
  double sumx_div = 0;
  double sumx2_birth = 0;
  double sumx2_start = 0;
  double sumx2_div = 0;

  double sumy_g1 = 0;
  double sumy_g2 = 0;
  double sumy_cycle = 0;

  double sumxy_g1 = 0;
  double sumxy_g2 = 0;
  double sumxy_cycle = 0;

  double target_volume_birth = 5.0;
  double target_volume_START = 4.0;
  double target_volume_div = 6.0;
  double diff_volume_birth = 0;
  double diff_volume_START = 0;
  double diff_volume_div = 0;

  double v_birth = 0;
  double v_start = 0;
  double v_div = 0;

  double slope_g1 = 0;
  double slope_g2 = 0;
  double slope_cycle = 0;
  double denom_g1 = 0;
  double denom_g2 = 0;
  double denom_cycle = 0;
  double mean_volume_birth = 0;
  double mean_volume_START = 0;
  double mean_volume_div = 0;
  double std_volume_birth = 0;
  double std_volume_START = 0;
  double std_volume_div = 0;
  

// Loop over all cells, here NCELLTOT = 1 so no worries
  for (ncell=0;ncell<NCELLTOT;ncell++) {
    // Loop over all time steps
    for (pas=0;pas<NSTEP-1;pas++) {
      // Change in the cell cycle?
      if (pas == 0){
        v_birth = history[trackin[0]][pas][ncell];
        sumx_birth += v_birth;
        sumx2_birth += v_birth*v_birth;
      }
      diff_cell_cycle = history[trackin[1]][pas+1][ncell] - history[trackin[1]][pas][ncell];

      // Does the volume go below a threshold value? Cell is considered dead
      if (history[trackin[0]][pas][ncell] <= 0.1){
        dead = 1;
        break;
      }
      // Does the volume go above a threshold value? Cell is considered dead
      else if (history[trackin[0]][pas][ncell] >= 99.5){
        maxv = 1;
      }

      //START
      if (diff_cell_cycle == 1 && history[trackin[0]][pas][ncell] > 0.1){
        if (n_div > 5){
          diff_volume_START += pow(history[trackin[0]][pas+1][ncell]-target_volume_START,2);
          
          v_start = history[trackin[0]][pas+1][ncell];
          sumx_start += v_start;
          sumx2_start += v_start*v_start;

          sumxy_g1 += v_birth*(v_start - v_birth);
        }

      }

      //DIVISION and BIRTH
      else if (diff_cell_cycle == -1 && history[trackin[0]][pas][ncell] > 0.1){
        if (n_div > 5){
          diff_volume_div += pow(history[trackin[0]][pas][ncell] - target_volume_div,2);
          diff_volume_birth += pow(history[trackin[0]][pas][ncell] - target_volume_birth,2);
          
          v_div = history[trackin[0]][pas][ncell];
          sumx_div += v_div;
          sumx2_div += v_div*v_div;

          sumxy_g2 += v_start*(v_div - v_start);
          sumxy_cycle += v_birth*(v_div - v_birth);

          v_birth = history[trackin[0]][pas+1][ncell]; //reset value
          sumx_birth += v_birth;
          sumx2_birth += v_birth*v_birth;
        }
        n_div += 1*(1-dead);

      }
    }
  }
  // Compute the fitness for this run
  if (n_div >= 6){
    n_div = n_div - 5; // Ignore the first 5 divisions to get more the "Stead-state" added volume
    denom_g1 = (n_div*sumx2_birth - sumx_birth*sumx_birth);
    if (denom_g1 == 0.0){
      //Matrix is singular, we give bad fitness
      slope_g1 = 50.0;
    }
    else {
      slope_g1 = (n_div*sumxy_g1-sumx_birth*(sumx_start-sumx_birth))/denom_g1;
      if (fabs(slope_g1) > 10){
        slope_g1 = 10;
      }
    }
    
    denom_g2 = (n_div*sumx2_start - sumx_start*sumx_start);
    if (denom_g2 == 0.0){
      //Matrix is singular, we give bad fitness
      slope_g2 = 50.0;
    }
    else {
      slope_g2 = (n_div*sumxy_g2 - sumx_start*(sumx_div - sumx_start))/denom_g2;
      if (fabs(slope_g2) > 10){
        slope_g2 = 10;
      }
    }
    
    denom_cycle = (n_div*sumx2_birth - sumx_birth*sumx_birth);
    if (denom_cycle == 0.0){
      //Matrix is singular, we give bad fitness
      slope_cycle = 50.0;
    }
    else {
      slope_cycle = (n_div*sumxy_cycle - sumx_birth*(sumx_div - sumx_birth))/denom_cycle;
      if (fabs(slope_cycle) > 10){
        slope_cycle = 10;
      }
    }
    
    mean_volume_birth = sumx_birth/n_div;
    mean_volume_START = sumx_start/n_div;
    mean_volume_div = sumx_div/n_div;

    std_volume_birth = (sumx2_birth/n_div - mean_volume_birth*mean_volume_birth);
    if (std_volume_birth < 0){
      std_volume_birth = 1.0;
    }
    else {
      std_volume_birth = sqrt(std_volume_birth);
    }

    std_volume_START = (sumx2_start/n_div - mean_volume_START*mean_volume_START);
    if (std_volume_START < 0){
      std_volume_START = 1.0;
    }
    else {
      std_volume_START = sqrt(std_volume_START);
    }
    std_volume_div = (sumx2_div/n_div - mean_volume_div*mean_volume_div);
    if (std_volume_div < 0){
      std_volume_div = 1.0;
    }
    else {
      std_volume_div = sqrt(std_volume_div);
    }
    diff_volume_birth /= n_div*target_volume_birth;
    diff_volume_START /= n_div*target_volume_START;
    diff_volume_div /= n_div*target_volume_div;
    if (maxv == 1){
      result[trial][0] = -0.9*n_div;
    }
    else{
      result[trial][0] = -n_div; // Fitness result
    }
    
    if (mean_volume_birth == 0){
      result[trial][1] = 10000;
    }
    else {
      result[trial][1] = std_volume_birth/mean_volume_birth;
    }
  }
  else {
    result[trial][0] = 10000;
    result[trial][1] = 10000;
  }
}

// Combine the fitnesses obtained in the different trials.
void treatment_fitness( double history2[][NSTEP][NCELLTOT], int trackout[]){
  // MODIFY HERE to combine the fitnesses computed for the different trials.
  int trial = 0;
  int func = 0;
  double total_fitness1 = 0;
  double total_fitness2 = 0;
  for(trial=0;trial<NTRIES;trial++){
    total_fitness1 += result[trial][0]/NTRIES;
    total_fitness2 += result[trial][1]/NTRIES;    
  }

  // Print network's fitness caught by the python code
  printf("%f\n%f",total_fitness1,total_fitness2);
}
/* Define geometry[NCELLTOT][NNEIGHBOR] array defining the neighbors of each cell
   Convention geometry < 0 ie invalid index -> no neighbor, ie used for end cells
   NB even for NCELLTOT=1, should have NNEIGHBOR == 3 to avoid odd seg faults since
   3 neighbors accessed in various loops, eventhough does nothing.
*/
   
void init_geometry() {

  int index;

  for (index=0;index<NCELLTOT;index++){
    geometry[index][0]=index-1;//NB no left neighbour for cell 0 (-1)
    geometry[index][1]=index;
    geometry[index][2]=index+1;
  }

  geometry[NCELLTOT-1][2]=-1;//NB no right neighbour for cell ncell-1 (-1)

}


/* set history of all species at initial time to rand number in [0.1,1) eg avoid 0 and initializes Input.
For this problem, we created a function init_signal that defines random steps of inputs.
init_signal creates time dependent input from [0, NSTEP) by integrating a sum of random positive delta functions and then exponentiating.

tfitness is defined in the fitness C file
*/

void init_history(int trial)  {
  int ncell,n_gene;
  int t_time_step = 0;
  // Everything to 0
  for (ncell=0;ncell<NCELLTOT;ncell++)
  {
    for (n_gene=0;n_gene<SIZE;n_gene++)
    {
      if (n_gene==trackin[0]){
        history[n_gene][0][ncell] = 2.0+0.5*FRAND();
      }
      else if (n_gene == trackin[1]){
    
        history[n_gene][0][ncell] = 0;
      }
      else if (n_gene == trackout[0]){
        history[n_gene][0][ncell] = 1.0+FRAND();
      }
      else {
        history[n_gene][0][ncell] = 0.5 + FRAND();
      }
    }
  }
}

/* define input variables as a function of step, which cell, and n-attempts = loop [0,ntries)
set history[input# ][pas][ncell] to correct value.
*/


void inputs(int pas,int ncell,int n_attempts){
	//history[trackin[1]][pas][ncell] = 1;
}



/* compute the RHS of equations and run over NSTEP's with 1st order Euler method
   The arugment kk, is an index passed to inputs that records which iteration of
   initial or boundary conditions the same system of equs is being integrated for

   This version of code used with ligand at cell computed from sum neighbors
*/


void integrator(int kk){

    double s[SIZE];
    double ds[SIZE];
    double sumligands[SIZE];
    double memory[SIZE];
    int index,n,pas,ncell;
    int switch_pas;
    for (index=0;index<SIZE;index++){
	  s[index] = 0;
        ds[index]=0;
        memory[index]=0;
    }
    /* initialize geometry here, incase cells move  */
    init_geometry();
    init_history(kk);

    /* loop over time steps, then over each cell etc */
    for (pas=0;pas<NSTEP-1;pas++)  {
	for (ncell=0;ncell<NCELLTOT;ncell++)  {
            inputs(pas,ncell,kk);
            for (index=0;index<SIZE;index++) {
            s[index]=history[index][pas][ncell];
            }
            derivC(s,history,pas,ds,memory,ncell);  //local integration
            //sum_concentration(ncell,pas,sumligands);  //perform sum of ligands concentrations for non external ligands
	    //diffusion(ncell,pas,ds,history,geometry);//computes diffusion of external ligands
            /*LRinC(s,ds,sumligands);*/

            for (index=0;index<SIZE;index++) {
              double rate = mdg_rate(pas,ncell,history,trackin);
              
              if (index == trackin[0]){ 
                // Update the volume
                history[index][pas+1][ncell] = s[index] + DT*(rate*s[index]);
              }
              else if (index == trackin[1]){
                int timer_length = (30+2*(1-2*FRAND()));
                // Update the switch
                if ((2/(1+exp(pow(history[trackout[0]][pas][ncell]*history[trackin[0]][pas][ncell]/0.8,8))))>FRAND() && history[trackin[1]][pas][ncell] == 0){
                  // commit to start
                  for (switch_pas = pas; switch_pas<pas+timer_length; switch_pas++){
                    history[trackin[1]][switch_pas+1][ncell] = 1; // update switch for a TIMER of 10s in S/G2/M
                  }
                }
                if (history[trackin[1]][pas+1][ncell] != history[trackin[1]][pas][ncell] && history[trackin[1]][pas][ncell] == 1){
                  // commit to division after a TIMER of 10s in S/G2/M
                  history[trackin[0]][pas+1][ncell] = history[trackin[0]][pas][ncell]/2; // symmetric division
                }
              }
              else {                                                        // dilution term
                history[index][pas+1][ncell] = MIN(s[index] + DT*(ds[index]-rate*s[index]),200);
              }
	 	       
		 


         if (history[index][pas+1][ncell]<0)//might happen for langevin
		   history[index][pas+1][ncell]=0;
	    }
	}}

    /* fill in inputs for last time.  */
    for (ncell=0;ncell<NCELLTOT;ncell++)  {
      inputs(NSTEP-1,ncell,kk);
    }
}
/*
General main.c, should not need any others, see comments in fitness_template for
the functions that must be supplied in that file for the main to work,
*/

int main()  {
    srand( SEED );
    int i,k,l;
    double score = 0;
    int len_hist = SIZE*NCELLTOT*NSTEP;
    double *hptr = &history[0][0][0];
    /* following incase one wants to average history before doing fitness */
    static double history2[SIZE][NSTEP][NCELLTOT];
    double *h2ptr = &history2[0][0][0];//table for averaging output (used for multicell problems)

    /* dummy return when no outputs for fitness function */
    if(NOUTPUT <= 0) {
        printf("%s","no output variables? terminating without integration" );
    }

    for(i=0; i<len_hist; i++)  {
        *(h2ptr + i) = 0;
    }

    for (k=0; k<NTRIES; k++){
        integrator(k);
        fitness(history, trackin,k);
        if( PRINT_BUF )  {
            print_history(k);
        }

        for(i=0; i<len_hist; i++)  {
            *(h2ptr+i) += *(hptr +i);
        }
    }
    treatment_fitness(history2,trackout);
}
