 /*

Defines fitness function
*/


static double result[NTRIES]; //result will contain the fitness plus the other results computed by the fitness function
/*
void fitness( double history[][NSTEP][NCELLTOT],int trackin[],int trial)  {
  // MODIFY HERE to compute a fitness from the history array
  int pas,ncell;
  for (ncell=0;ncell<NCELLTOT;ncell++) {    
    for (pas=0;pas<NSTEP-1;pas++) {
      if (history[trackin[0]][pas][ncell] <= 0.5){
        result[trial] = 1000;
        break;
      }
      else if (history[trackin[1]][pas+1][ncell] != history[trackin[1]][pas][ncell] && history[trackin[1]][pas][ncell] == 0)
        result[trial] += -1;
    }
  }
}
*/
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
      else if (diff_cell_cycle == -1 && history[trackin[0]][pas][ncell] > 0.1){
        if (n_div > 5){
          diff_volume_START += pow(history[trackin[0]][pas+1][ncell]-target_volume_START,2);
          
          v_start = history[trackin[0]][pas+1][ncell];
          sumx_start += v_start;
          sumx2_start += v_start*v_start;

          sumxy_g1 += v_birth*(v_start - v_birth);
        }

      }

      //DIVISION and BIRTH
      else if (diff_cell_cycle == 1 && history[trackin[0]][pas][ncell] > 0.1){
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
      result[trial] = -0.9*n_div;
    }
    else{
      result[trial] = -n_div; // Fitness result
    }
    
    
  
  }
  else {
    result[trial] = 10000;
  }
}



// Combine the fitnesses obtained in the different trials.
void treatment_fitness( double history2[][NSTEP][NCELLTOT], int trackout[]){
  // MODIFY HERE to combine the fitnesses computed for the different trials.
  int trial = 0;
  int func = 0;
  double total_fitness = 0;
  for(trial=0;trial<NTRIES;trial++)
    {
      total_fitness += result[trial]/NTRIES;
    }
  
  // Print network's fitness caught by the python code
  printf("%f",total_fitness);
}
