"""
Performs independent t-test at p-value=0.05 to compare model performance across 5 random seeds.
"""

from lib.Eval import students_t_test
import numpy as np

rob_base = [42.38, 42.08, 42.14, 41.71, 42.49]

# Compare baseline to best domain context model
rob_basil_tapt = [44.16, 44.14, 42.96, 42.73, 41.61]
print("- Base RoBERTa VS Basil-adapted RoBERTa -")
students_t_test(rob_base, rob_basil_tapt)

# Compare baseline to ArtCIM
print("\n- Base RoBERTa VS ArtCIM -")
artcim = [42.24, 43.27, 42.70, 42.32, 43.47]
print(np.mean(artcim), np.std(artcim))
students_t_test(rob_base, artcim)

# Compare baseline to ArtCIM*
print("\n- Base RoBERTa VS ArtCIM* -")
artcimstar = [42.25, 42.8, 41.26]
students_t_test(rob_base, artcimstar)

# Compare baseline to EvCIM
print("\n- Base RoBERTa VS EvCIM -")
evcim = [44.28, 43.92, 44.17, 44.15, 43.97]
students_t_test(rob_base, evcim)

# Compare baseline to EvCIM*
print("\n- Base RoBERTa VS EvCIM* -")
evcimstar = [43.31, 43.22, 42.96, 42.84, 42.46]
students_t_test(rob_base, evcimstar)