#!/usr/bin/env python3

import os
import sys

# PyQt4 still has a bug where it fails to yeild interactive control
# back to the starting terminal on some operating systems. This is
# at least observed on Linux and may affect OSX as well. Manually
# setting Qt4 return hooks didn't work. The only apparent fix
# is to move to Qt5 if able. This method check if the required
# PyQt5 modules and backend are present. If they are, we move the
# renderer forward to PyQt5. Otherwise, we fallback to the default.
#
# Backend *MUST* be set before any subpackages are imported. 
def select_pyplot_renderer():
	try:
		from PyQt5 import QtCore, QtGui, QtWidgets
		import matplotlib
		matplotlib.use('Qt5Agg')
	except ImportError:
		if sys.platform == "linux" or sys.platform == "linux2":
			print('*** could not set sim renderer to Qt5Agg')
			print('*** canvas may fail to yield interactive control on close')
			print('*** please install PyQt5 if you experience issues')
			print('\tDebian Package: python3-pyqt5')
			print('\tFedora Core: python3-qt5')
			print('')
			print('')

		# fallback renderer
		import matplotlib
		matplotlib.use('TkAgg')
	

select_pyplot_renderer()

print('launching sim program: ' + sys.argv[1])
print('')
os.system('python3 ' + sys.argv[1])
exit(0)

