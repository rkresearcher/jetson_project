import os
import pandas as pd
import xlrd
fl_file  = pd.read_excel('Data Set Link sources.xlsx', index_col=0)

for i in range(len(fl_file['Link'])):
	os.system("youtube-dl -u rahul007ks@gmail.com -p Ra2hu4l5# "+ fl_file['Link'][i])
	print("Done")
