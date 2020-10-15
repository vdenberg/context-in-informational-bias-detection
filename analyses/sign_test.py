from scipy.stats import ttest_ind


def students_t_test(results1, results2):
	stat, p = ttest_ind(results1, results2)
	print('stat=%.3f, p=%.3f' % (stat, p))
	if p > 0.05:
		print('Probably the same distribution')
	else:
		print('Probably different distributions')


rob_base = [42.93, 42.93, 42.93, 42.93, 42.93]

# Compare baseline to best domain context model
rob_basil_tapt = [44.16, 44.14, 42.96, 42.73, 41.61]
students_t_test(rob_base, rob_basil_tapt)

# Compare baseline to ArtCIM
artcim = [44.16, 44.14, 42.96, 42.73, 41.61]
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