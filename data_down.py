import os
import pandas as pd
import xlrd
fl_file = xlrd.open_workbook("Data Set Link sources.xlsx")
sheet = fl_file.sheet_by_index(0)
print (sheet.cell_value(0,0))
for i in range(sheet.nrows):
	print (sheet.cell_value(i,1))
