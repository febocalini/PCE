#!/home/sergio/miniconda2/envs/momi-py36/bin/ python

import momi	
import logging
import pickle			

logging.basicConfig(level=logging.INFO,
                    filename="log_model1_tan.log")

sfs = momi.Sfs.load("tan_sfs.gz")

## Modelo 1 - isolation without migration 
tan_model1 = momi.DemographicModel(N_e=1e5, gen_time=2.3, muts_per_gen=2.5e-9)

tan_model1.set_data(sfs)

tan_model1.add_time_param("tdiv", lower=5e3, upper=5e6)

tan_model1.add_leaf("AF", N=1.88e5)
tan_model1.add_leaf("CEP", N=1.03e5)

tan_model1.move_lineages("AF", "CEP", t="tdiv")

tan_model1.optimize()

lik = tan_model1.log_likelihood()

#### output
file = open("bestrun_tan_revisado.txt","a")
file.write("model1=run1" '\n')
file.write("Log_likelihood=")
file.write(str(lik))
file.write('\n')
file.close()

### repetitions ###

results = []
n_runs = 100
tan_model1_copy = tan_model1.copy()
for i in range(n_runs):
    print(f"Starting run {i+1} out of {n_runs}...")
    tan_model1.set_params(tan_model1.get_params(),randomize=True)
    results.append(tan_model1_copy.optimize(options={"maxiter":200}))
    lik=tan_model1_copy.log_likelihood()
    print(lik)

# sort results according to log likelihood, pick the best one
best_result = sorted(results, key=lambda r: r.log_likelihood, reverse=True)[0]

tan_model1_copy.set_params(best_result.parameters)
best_result
nparams= len(best_result.parameters)


#### output
file = open("bestrun_tan_revisado.txt","a")
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

## export the best result

tan_model1 = best_result
f = open("tan_model1.pkl","wb")
pickle.dump(tan_model1,f)
f.close()

###############
quit()
