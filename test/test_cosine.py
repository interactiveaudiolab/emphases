import emphases

from emphases.evaluate import core

def calc_cosine_similarity(prom_file, ground_truth_file):
    return core.eval_similarity(prom_file, ground_truth_file)

# Test 1
prom_file = "../eval/sanity_check_cosine/s02-1.prom"
ground_truth_file = "../eval/BuckEye-annotations.csv"

ground_truth_values = [0.     , 0.5625 , 0.15625, 0.46875, 0.     , 0.40625, 0.     ,
       0.78125, 0.53125, 0.     , 0.03125, 0.0625 , 0.     , 0.53125,
       0.09375]
pred_val = [0.959, 0.722, 0.482, 0.388, 1.004, 1.795, 0.948, 0.96 , 0.758,
       0.796, 0.275, 0.899, 0.106, 0.436, 2.032]

cosine_truth = 0.5796336259434719
calc_cosine = calc_cosine_similarity(prom_file, ground_truth_file)
print(calc_cosine, cosine_truth)
assert calc_cosine==cosine_truth
print('test case passed \n')

# Test 2
prom_file = "../eval/sanity_check_cosine/s17-1.prom"
ground_truth_file = "../eval/BuckEye-annotations.csv"

ground_truth_values = [0.59375, 0.     , 0.1875 , 0.0625 , 0.03125, 0.5625 , 0.     ,
       0.     , 0.03125, 0.0625 , 0.     , 0.     , 0.     , 0.53125,
       0.03125]
pred_val = [1.798, 0.94 , 1.746, 0.335, 1.011, 1.36 , 0.778, 0.792, 0.489,
       0.607, 0.   , 0.667, 0.609, 1.098, 0.887]

cosine_truth = 0.7489379035381454
calc_cosine = calc_cosine_similarity(prom_file, ground_truth_file)
print(calc_cosine, cosine_truth)
assert calc_cosine==cosine_truth
print('test case passed \n')

# Test 3
cosine_truth = None
prom_file = "../eval/sanity_check_cosine/s03-1.prom"
ground_truth_file = "../eval/BuckEye-annotations.csv"

ground_truth_values = [0.008]
pred_val = []

cosine_truth = None
calc_cosine = calc_cosine_similarity(prom_file, ground_truth_file)
print(calc_cosine, cosine_truth)
assert calc_cosine==cosine_truth
print('test case passed \n')
