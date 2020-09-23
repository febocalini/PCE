#!/home/sergio/miniconda2/envs/momi-py36/bin/ python

import momi	
import logging
import pickle			

logging.basicConfig(level=logging.INFO,
                    filename="log_model1_pic.log")

sfs = momi.Sfs.load("pic_sfs.gz")

#Model 1 - isolation without migration

#Modelo 1 - isolamento sem migração
pic_model1 = momi.DemographicModel(N_e=1e5, gen_time=2.3, muts_per_gen=2.5e-9)

pic_model1.set_data(sfs)

pic_model1.add_time_param("tdiv_AF_CEP", lower=5e3, upper=5e6)

pic_model1.add_leaf("AF", N=2.5e4)
pic_model1.add_leaf("CEP", N=3.1e4)

pic_model1.move_lineages("AF", "CEP", t="tdiv_AF_CEP")

pic_model1.add_time_param("tdiv_CEP_AM", lower_constraints=["tdiv_AF_CEP"], upper=5e6)

pic_model1.add_leaf("AM", N=2.16e5)
pic_model1.move_lineages("CEP", "AM", t="tdiv_CEP_AM")

pic_model1.optimize(method='L-BFGS-B')

lik = pic_model1.log_likelihood()

#### output
file = open("bestrun_pic_revisado.txt","a")
file.write("model1=run1" '\n')
file.write("Log_likelihood=")
file.write(str(lik))
file.write('\n')
file.close()

### repetitions ###

results = []
n_runs = 100
pic_model1_copy = pic_model1.copy()
for i in range(n_runs):
    print(f"Starting run {i+1} out of {n_runs}...")
    pic_model1.set_params(pic_model1.get_params(),randomize=True)
    results.append(pic_model1_copy.optimize(method='L-BFGS-B'))
    lik=pic_model1_copy.log_likelihood()
    print(lik)

# sort results according to log likelihood, pick the best one
best_result = sorted(results, key=lambda r: r.log_likelihood, reverse=True)[0]

pic_model1_copy.set_params(best_result.parameters)
best_result
nparams= len(best_result.parameters)

#### output
file = open("bestrun_pic_revisado.txt","a")
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

pic_model1 = best_result
f = open("pic_model1.pkl","wb")
pickle.dump(pic_model1,f)
f.close()

###############
quit()
