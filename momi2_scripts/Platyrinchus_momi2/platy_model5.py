#!/home/sergio/miniconda2/envs/momi-py36/bin/ python

import momi	
import logging
import pickle			

logging.basicConfig(level=logging.INFO,
                    filename="log_model5_platy.log")

sfs = momi.Sfs.load("platy_sfs.gz")

##Modelo 5 - isolation with migration and expansion of the SCAF population

play_model5 = momi.DemographicModel(N_e=1e5, gen_time=2.3, muts_per_gen=2.5e-9)

play_model5.set_data(sfs)

play_model5.add_time_param("tdiv_AF_CEP", lower=5e3, upper=5e6)
play_model5.add_time_param("tmig_AF_CEP",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
play_model5.add_pulse_param("mfrac_AF_CEP", upper=.2)
play_model5.add_time_param("tmig_CEP_AF",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
play_model5.add_pulse_param("mfrac_CEP_AF", upper=.2)

play_model5.add_leaf("AF", g=1e-5, N=4.88e5)
play_model5.add_leaf("CEP", N=1.28e5)

play_model5.set_size("AF", t="tdiv_AF_CEP", g=0)

play_model5.move_lineages("AF", "CEP", t="tmig_AF_CEP", p="mfrac_AF_CEP")
play_model5.move_lineages("CEP","AF", t="tmig_CEP_AF", p="mfrac_CEP_AF")
play_model5.move_lineages("AF", "CEP", t="tdiv_AF_CEP")

play_model5.add_time_param("tmig_Andes_CEP",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
play_model5.add_pulse_param("mfrac_Andes_CEP", upper=.2)
play_model5.add_time_param("tmig_CEP_Andes",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
play_model5.add_pulse_param("mfrac_CEP_Andes", upper=.2)

play_model5.add_time_param("tmig_Andes_AF",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
play_model5.add_pulse_param("mfrac_Andes_AF", upper=.2)
play_model5.add_time_param("tmig_AF_Andes",  upper_constraints=["tdiv_AF_CEP"], lower=5e3)
play_model5.add_pulse_param("mfrac_AF_Andes", upper=.2)

play_model5.add_time_param("tdiv_CEP_Andes", lower_constraints=["tdiv_AF_CEP"], upper=5e6)

play_model5.add_leaf("Andes", N=3.13e5)
play_model5.move_lineages("Andes", "CEP", t="tmig_CEP_Andes", p="mfrac_Andes_CEP")
play_model5.move_lineages("CEP","Andes", t="tmig_CEP_Andes", p="mfrac_CEP_Andes")
play_model5.move_lineages("Andes", "AF", t="tmig_Andes_AF", p="mfrac_Andes_AF")
play_model5.move_lineages("AF","Andes", t="tmig_AF_Andes", p="mfrac_AF_Andes")

play_model5.move_lineages("CEP", "Andes", t="tdiv_CEP_Andes")

play_model5.optimize(method='L-BFGS-B')

lik = platy_model5.log_likelihood()

#### output
file = open("bestrun_platy_revisado.txt","a")
file.write("model5=run1" '\n')
file.write("Log_likelihood=")
file.write(str(lik))
file.write('\n')
file.close()

### repetitions ###

results = []
n_runs = 100
platy_model5_copy = platy_model5.copy()
for i in range(n_runs):
    print(f"Starting run {i+1} out of {n_runs}...")
    platy_model5.set_params(platy_model5.get_params(),randomize=True)
    results.append(platy_model5_copy.optimize(method='L-BFGS-B'))
    lik=platy_model5_copy.log_likelihood()
    print(lik)

# sort results according to log likelihood, platyk the best one
best_result = sorted(results, key=lambda r: r.log_likelihood, reverse=True)[0]

platy_model5_copy.set_params(best_result.parameters)
best_result
nparams= len(best_result.parameters)

#### output
file = open("bestrun_platy_revisado.txt","a")
file.write("Model=model5" '\n')
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

platy_model5 = best_result
f = open("platy_model5.pkl","wb")
platykle.dump(platy_model5,f)
f.close()

###############
quit()
