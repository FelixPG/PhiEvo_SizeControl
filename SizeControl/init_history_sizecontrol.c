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

