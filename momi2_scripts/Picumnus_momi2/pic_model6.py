#!/home/sergio/miniconda2/envs/momi-py36/bin/ python

import momi	
import logging
import pickle			

logging.basicConfig(level=logging.INFO,
                    filename="log_model6_pic.log")

sfs = momi.Sfs.load("pic_sfs.gz")

#Model 6 - isolation with migration with population bottleneck in the SCAF population

pic_model6 = momi.DemographicModel(N_e=1e5, gen_time=2.3, muts_per_gen=2.5e-9)

pic_model6.set_data(sfs)

pic_model6.add_time_param("tdiv_AF_CEP", lower=5e3, upper=5e6)
pic_model6.add_time_param("tmig_AF_CEP",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
pic_model6.add_pulse_param("mfrac_AF_CEP", upper=.2)
pic_model6.add_time_param("tmig_CEP_AF",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
pic_model6.add_pulse_param("mfrac_CEP_AF", upper=.2)

pic_model6.add_leaf("AF", g=1e-5, N=2.5e4)
pic_model6.add_leaf("CEP", N=3.1e4)

pic_model6.set_size("AF", t="tdiv_AF_CEP", N=1e4, g=0)

pic_model6.move_lineages("AF", "CEP", t="tmig_AF_CEP", p="mfrac_AF_CEP")
pic_model6.move_lineages("CEP","AF", t="tmig_CEP_AF", p="mfrac_CEP_AF")
pic_model6.move_lineages("AF", "CEP", t="tdiv_AF_CEP")

pic_model6.add_time_param("tmig_AM_CEP",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
pic_model6.add_pulse_param("mfrac_AM_CEP", upper=.2)
pic_model6.add_time_param("tmig_CEP_AM",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
pic_model6.add_pulse_param("mfrac_CEP_AM", upper=.2)

pic_model6.add_time_param("tmig_AM_AF",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
pic_model6.add_pulse_param("mfrac_AM_AF", upper=.2)
pic_model6.add_time_param("tmig_AF_AM",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
pic_model6.add_pulse_param("mfrac_AF_AM", upper=.2)

pic_model6.add_time_param("tdiv_CEP_AM", lower_constraints=["tdiv_AF_CEP"], upper=5e6)

pic_model6.add_leaf("AM", N=2.16e5)
pic_model6.move_lineages("AM", "CEP", t="tmig_CEP_AM", p="mfrac_AM_CEP")
pic_model6.move_lineages("CEP","AM", t="tmig_CEP_AM", p="mfrac_CEP_AM")
pic_model6.move_lineages("AM", "AF", t="tmig_AM_AF", p="mfrac_AM_AF")
pic_model6.move_lineages("AF","AM", t="tmig_AF_AM", p="mfrac_AF_AM")

pic_model6.move_lineages("CEP", "AM", t="tdiv_CEP_AM")

pic_model6.optimize(method='L-BFGS-B')

lik = pic_model6.log_likelihood()

#### output
file = open("bestrun_pic_revisado.txt","a")
file.write("model6=run1" '\n')
file.write("Log_likelihood=")
file.write(str(lik))
file.write('\n')
file.close()

### repetitions ###

results = []
n_runs = 100
pic_model6_copy = pic_model6.copy()
for i in range(n_runs):
    print(f"Starting run {i+1} out of {n_runs}...")
    pic_model6.set_params(pic_model6.get_params(),randomize=True)
    results.append(pic_model6_copy.optimize(method='L-BFGS-B'))
    lik=pic_model6_copy.log_likelihood()
    print(lik)

# sort results according to log likelihood, pick the best one
best_result = sorted(results, key=lambda r: r.log_likelihood, reverse=True)[0]

pic_model6_copy.set_params(best_result.parameters)
best_result
nparams= len(best_result.parameters)

#### output
file = open("bestrun_pic_revisado.txt","a")
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

## exportar o melhor modelo

pic_model6 = best_result
f = open("pic_model6.pkl","wb")
pickle.dump(pic_model6,f)
f.close()

###############
quit()
