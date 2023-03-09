import os, glob


class deltaSet:
    def __init__(self):
        self.imgs = dict()
        self.masks = dict()


def findNifTibyKeys(indir: str):
    initdir = os.getcwd()
    os.chdir(indir)
    filelist = list()
    for file in glob.glob("*.nii*"):
        filelist.append(file)
    return filelist


def evalFileName(deltadict, filename: str):
    temp = filename.split("_")


indir = "E:\home\Thomas\Sciebo\Projekte\Kidney\Kidney_Delta\data\pro4\imgs"
filelist = findNifTibyKeys(indir)
deltaDict = dict()
for file in filelist:
    evalFileName(deltaDict, file)
