from os import listdir
from os.path import isfile, join

def GetFiles(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]
