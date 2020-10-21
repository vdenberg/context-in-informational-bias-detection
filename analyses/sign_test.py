from lib.Eval import students_t_test


rob_base = [42.38, 42.08, 42.14, 41.71, 42.49]

# Compare baseline to best domain context model
rob_basil_tapt = [44.16, 44.14, 42.96, 42.73, 41.61]
students_t_test(rob_base, rob_basil_tapt)

# Compare baseline to ArtCIM
# 34, 68, 102, 170 (43.54), 204
artcim = [44.16, 44.14, 42.96, 43.54, 41.61]
students_t_test(rob_base, artcim)

# Compare baseline to ArtCIM*
artcimstar = [44.16, 44.14, 42.96, 42.73, 41.61]
students_t_test(rob_base, artcimstar)

# Compare baseline to EvCIM
evcim = [44.28, 43.92, 44.17, 44.15, 43.97]
students_t_test(rob_base, evcim)

# Compare baseline to EvCIM*
evcimstar = [43.31, 43.22, 42.96, 42.84, 42.46]
students_t_test(rob_base, evcimstar)