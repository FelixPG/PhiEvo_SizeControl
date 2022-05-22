
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
