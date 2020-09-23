#!/home/sergio/miniconda2/envs/momi-py36/bin/ python

import momi	
import logging
import pickle			

logging.basicConfig(level=logging.INFO,
                    filename="log_model6_tan.log")

sfs = momi.Sfs.load("tan_sfs.gz")

#Model 6 - isolation with migration with a bottleneck in the SCAF population

tan_model6 = momi.DemographicModel(N_e=1e5, gen_time=2.3, muts_per_gen=2.5e-9)

tan_model6.set_data(sfs)

tan_model6.add_time_param("tdiv_AF_CEP", lower=5e3, upper=5e6)
tan_model6.add_time_param("tmig_AF_CEP",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
tan_model6.add_pulse_param("mfrac_AF_CEP", upper=.2)
tan_model6.add_time_param("tmig_CEP_AF",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
tan_model6.add_pulse_param("mfrac_CEP_AF", upper=.2)

tan_model6.add_leaf("AF", g=1e-5, N=1.88e5)
tan_model6.add_leaf("CEP", N=1.03e5)

tan_model6.set_size("AF", t="tdiv_AF_CEP", N=5e4, g=0)

tan_model6.move_lineages("AF", "CEP", t="tmig_AF_CEP", p="mfrac_AF_CEP")
tan_model6.move_lineages("CEP","AF", t="tmig_CEP_AF", p="mfrac_CEP_AF")
tan_model6.move_lineages("AF", "CEP", t="tdiv_AF_CEP")

tan_model6.optimize()

lik = tan_model6.log_likelihood()

#### output
file = open("bestrun_tan_revisado.txt","a")
file.write("model6=run1" '\n')
file.write("Log_likelihood=")
file.write(str(lik))
file.write('\n')
file.close()

### repetitions ###

results = []
n_runs = 100
tan_model6_copy = tan_model6.copy()
for i in range(n_runs):
    print(f"Starting run {i+1} out of {n_runs}...")
    tan_model6.set_params(tan_model6.get_params(),randomize=True)
    results.append(tan_model6_copy.optimize(options={"maxiter":200}))
    lik=tan_model6_copy.log_likelihood()
    print(lik)

# sort results according to log likelihood, pick the best one
best_result = sorted(results, key=lambda r: r.log_likelihood, reverse=True)[0]

tan_model6_copy.set_params(best_result.parameters)
best_result
nparams= len(best_result.parameters)


#### output
file = open("bestrun_tan_revisado.txt","a")
file.write("Model=model6" '\n')
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

tan_model6 = best_result
f = open("tan_model6.pkl","wb")
pickle.dump(tan_model6,f)
f.close()

###############
quit()
