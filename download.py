import csv
import urllib.request
import zipfile
import os

os.chdir('C:/Users/ken_nakatsukasa/Desktop/download_checkTable')

with open('./download_list.csv', 'r') as ff:   #******only change file name here
	xx = csv.reader(ff)
	xlist = []
	for x in xx:
		xlist.extend(x)
		

address0 = "https://zzzzz.com//mnt/ai_create_job/agents/"   # dev

for n, x in enumerate(xlist):
    print("downoading = ", n+1, " of ", len(xlist) )
    zip_name = x + ".zip"
    url = address0 + zip_name 
    urllib.request.urlretrieve(url, zip_name)
    with zipfile.ZipFile(zip_name) as zip:
        zip.extractall(path=os.getcwd())
	
print("downloading finished!!" )

#run unix command
os.system('del *.zip') 
print("file deleted.")

