#!/home/sergio/miniconda2/envs/momi-py36/bin/ python

import momi	
import logging
import pickle			

logging.basicConfig(level=logging.INFO,
                    filename="log_model1_platy.log")

sfs = momi.Sfs.load("platy_sfs.gz")

#Model 1 - isolation without migration

platy_model1 = momi.DemographicModel(N_e=1e5, gen_time=2.3, muts_per_gen=2.5e-9)

platy_model1.set_data(sfs)

platy_model1.add_time_param("tdiv_AF_CEP", lower=5e3, upper=5e6)

platy_model1.add_leaf("AF", N=4.88e5)
platy_model1.add_leaf("CEP", N=1.28e5)

platy_model1.move_lineages("AF", "CEP", t="tdiv_AF_CEP")

platy_model1.add_time_param("tdiv_CEP_Andes", lower_constraints=["tdiv_AF_CEP"], upper=5e6)

platy_model1.add_leaf("Andes", N=3.13e5)
platy_model1.move_lineages("CEP", "Andes", t="tdiv_CEP_Andes")

platy_model1.optimize(method='L-BFGS-B')

lik = platy_model1.log_likelihood()

#### output
file = open("bestrun_platy_revisado.txt","a")
file.write("model1=run1" '\n')
file.write("Log_likelihood=")
file.write(str(lik))
file.write('\n')
file.close()

### repetitions ###

results = []
n_runs = 100
platy_model1_copy = platy_model1.copy()
for i in range(n_runs):
    print(f"Starting run {i+1} out of {n_runs}...")
    platy_model1.set_params(platy_model1.get_params(),randomize=True)
    results.append(platy_model1_copy.optimize(method='L-BFGS-B'))
    lik=platy_model1_copy.log_likelihood()
    print(lik)

# sort results according to log likelihood, platyk the best one
best_result = sorted(results, key=lambda r: r.log_likelihood, reverse=True)[0]

platy_model1_copy.set_params(best_result.parameters)
best_result
nparams= len(best_result.parameters)

#### output
file = open("bestrun_platy_revisado.txt","a")
file.write("Model=model1" '\n')
file.write("Log_likelihood=")
file.write(str(best_result.log_likelihood))
file.write('\n')
file.write("n_parameters=")
file.write(str(nparams))
file.write('\n')
file.write("Parameters_estimates:" '\n')
file.write(str(best_result.parameters))
file.write('\n')
file.write('\n')
file.close()

## exportar o melhor modelo

platy_model1 = best_result
f = open("platy_model1.pkl","wb")
platykle.dump(platy_model1,f)
f.close()

###############
quit()
