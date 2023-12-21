import tensorflow
from tensorflow import keras 
from keras import layers 
import pygad.kerasga
import numpy as np
import pandas as pd 
import pygad 

def int_to_gray(n): 
  n ^= (n >> 1)
  return n 

def gray_to_int(n): 
    mask = n
    while mask != 0: 
        mask >>= 1
        n ^= mask 
    return n 

def int_to_float(g, xmin, xmax, m):
    return (g*(xmax-xmin))/((2**m)-1)+xmin 

def on_start(ga_instance): 
    for sol in ga_instance.population: 
        for index, gen in enumerate(sol): 
            sol[index] = int_to_gray(gen)  

def fitness_func(ga_instance, solution, sol_idx):
    global data_inputs, data_outputs, keras_ga, model 

    tmp_sol = np.empty(len(solution), dtype=np.float16) 
    for index, greyGen in enumerate(solution): 
        intGen = gray_to_int(greyGen) 
        floatGen = int_to_float(intGen,-1,1,8)
        tmp_sol[index] = floatGen 

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=tmp_sol) 
    model.set_weights(weights=model_weights_matrix) 
    predictions = model.predict(data_inputs) 
    mae = tensorflow.keras.losses.MeanAbsoluteError()
    mae_error = mae(data_outputs, predictions).numpy() + 0.00000001
    solution_fitness = 1.0/mae_error 

    return solution_fitness 

def callback_generation(ga_instance): 
    print(f"\nGen: {ga_instance.generations_completed} | Fitness: {ga_instance.best_solution()[1]}\n")  

#Get train data from csv
poker_train = pd.read_csv(
    "poker-hand-training-true.data",
    names=["suit_of_card1", "rank_of_card1", "suit_of_card2", "rank_of_card2", "suit_of_card3", "rank_of_card3", 
           "suit_of_card4", "rank_of_card4", "suit_of_card5", "rank_of_card5", "poker_hand"]) 

data_inputs = poker_train.copy()
data_outputs = data_inputs.pop("poker_hand") 

#Create model 
labels_inv_norm_layer = layers.Normalization(invert=True, axis=None)
labels_inv_norm_layer.adapt(data_outputs)

predictors_norm_layer = layers.Normalization(axis=1)
predictors_norm_layer.adapt(data_inputs) 

inputs = keras.Input(shape=(10,)) 
# normalized_input = predictors_norm_layer(inputs) 
hidden = layers.Dense(128, activation='relu')(inputs)
hidden = layers.Dense(128, activation='relu')(hidden) 
outputs = layers.Dense(1, activation='relu')(hidden) 
# denormalized_output = labels_inv_norm_layer(outputs)
model = keras.Model(inputs=inputs, outputs=outputs) 

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=20)

ga_instance = pygad.GA(
                        ###Main###
                        num_generations=10, 
                        sol_per_pop=20,
                        num_genes=len(data_inputs), 
                        num_parents_mating=10, 
                        parent_selection_type="rws",
                        crossover_type="single_point", 
                        crossover_probability=0.8,
                        mutation_type="random",
                        mutation_probability=0.05, 
                        ###Gene###
                        gene_type=int,
                        init_range_low=0,
                        init_range_high=255,
                        ###Functions###
                        fitness_func=fitness_func,
                        on_generation=callback_generation,
                        on_start=on_start, 
                       )

ga_instance.run() 

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx)) 

float_sol = np.empty(len(solution), dtype=np.float16) 
for index, greyGen in enumerate(solution): 
     intGen = gray_to_int(greyGen) 
     floatGen = int_to_float(intGen,-1,1,8) 
     float_sol[index] = floatGen 

# Fetch the parameters of the best solution.
best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=float_sol)
model.set_weights(best_solution_weights)
model.save('gpmodel.keras')
predictions = model.predict(data_inputs)
print("Predictions : \n", predictions)

mae = tensorflow.keras.losses.MeanAbsoluteError()
abs_error = mae(data_outputs, predictions).numpy()
print("Absolute Error : ", abs_error) 