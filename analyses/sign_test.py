from scipy.stats import ttest_ind


def students_t_test(results1, results2):
	stat, p = ttest_ind(results1, results2)
	print('stat=%.3f, p=%.3f' % (stat, p))
	if p > 0.05:
		print('Probably the same distribution')
	else:
		print('Probably different distributions')


# Compare baseline to best domain context model
rob_base = [42.93, 42.93, 42.93, 42.93, 42.93]
rob_basil_tapt = [44.16, 44.14, 42.96, 42.73, 41.61]
students_t_test(rob_base, rob_basil_tapt)

# Compare baseline to ArtCIM
rob_base = [42.93, 42.93, 42.93, 42.93, 42.93]
covcim = [44.16, 44.14, 42.96, 42.73, 41.61]
students_t_test(rob_base, rob_basil_tapt)

# Compare baseline to ArtCIM*
rob_base = [42.93, 42.93, 42.93, 42.93, 42.93]
covcim = [44.16, 44.14, 42.96, 42.73, 41.61]
students_t_test(rob_base, rob_basil_tapt)

# Compare baseline to EvCIM
rob_base = [42.93, 42.93, 42.93, 42.93, 42.93]
covcim = [44.16, 44.14, 42.96, 42.73, 41.61]
students_t_test(rob_base, rob_basil_tapt)

# Compare baseline to EvCIM*
rob_base = [42.93, 42.93, 42.93, 42.93, 42.93]
covcim = [44.16, 44.14, 42.96, 42.73, 41.61]
students_t_test(rob_base, rob_basil_tapt)