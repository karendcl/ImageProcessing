import os
import repl, Canny, wt

#check for all the jpg in the folder

imgs = []

#load into videos all .mp4 files in the directory
for file_name in sorted(os.listdir('.')):
    if file_name[-3:]=='jpg': #checks for mp4 extension
        imgs.append(file_name)



for i in imgs:
    repl.main(i)
    Canny.main(i)
    wt.main(i)