import cv2
from glob import glob
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
import sys
from collections import Counter
from os import listdir
import csv
from difflib import SequenceMatcher
show_img = False
#takes an image and returns array with all chars in it
def chars_read(im):
    #copy becuase will be edited
    img = im.copy()

    #make an output, convert to grayscale and apply thresh-hold
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    #find conours
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #create empty return list
    ret_list =  []

    #for every contour if area large enoug to be digit add the box to list
    li = []
    for cnt in contours:
        if cv2.contourArea(cnt)>100:
            [x,y,w,h] = cv2.boundingRect(cnt)
            li.append([x,y,w,h])
    #sort list so it read from left to right
    li = sorted(li,key=lambda x: x[0], reverse=False)

    #loop over all chars found
    for i in li:
        #unpack data
        x,y,w,h = i[0], i[1], i[2], i[3]
        #check if large enough to be char but small enough to ignore rest
        if  h>20 and h<40 and w<60:
            ret = ""

            #draw rectangle with thresh-hold and shape to correct form
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            samples =  np.empty((0,100))
            sample = roismall.reshape((1,100))
            samples = np.append(samples,sample,0)

            #if user wants images shown
            if show_img:
                cv2.namedWindow('Window',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Window', 1600,600)
                cv2.imshow('Window', im)
                #show for 100 ms and check if exit called (esc key)
                key = cv2.waitKey(0)
                if key == 27:
                    sys.exit()

                #print what the NN would classify the char as
                print(chr(int(classify_name(samples))), sep=' ', end='', flush=True)
            
            #add char to return list
            ret = ret + chr(int(classify_name(samples)))
            ret_list.append(ret)

    #return all chars found
    if show_img:
        print('\n')
    return ret_list

def classify_name(data):
    return int(names_model.predict(data))

#list for wrongly classified 
img_list = []

#takes an image and returns array with all digits pixels in it
def digits_read(im, height, img_type, check=False):
    #copy becuase will be edited
    img = im.copy()

    #make an output, convert to grayscale and apply thresh-hold
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 0, 255,img_type)

    #find conours
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #create empty return list
    samples =  np.empty((0,100))

    #for every contour if area large enoug to be digit add the box to list
    li = []
    for cnt in contours:
        if cv2.contourArea(cnt)>20:
            [x,y,w,h] = cv2.boundingRect(cnt)
            li.append([x,y,w,h])
    #sort list so it read from right to left
    li = sorted(li,key=lambda x: x[0], reverse=True)

    #loop over all digits
    for i in li:
        #unpack data
        x,y,w,h = i[0], i[1], i[2], i[3]

        #cehck if large enough to be digit but small enough to ignore rest
        if  h>height and h<40 and w<40:

            #draw rectangle with thresh-hold and shape correct form
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            sample = roismall.reshape((1,100))
            samples = np.append(samples,sample,0)

            #if user wants images shown
            if show_img:
                cv2.namedWindow('1',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('1', 1600,600)
                cv2.imshow('1', im)
                #show for 100 ms and check if exit called (esc key)
                key = cv2.waitKey(0)
                if key == 27:
                    sys.exit()
                #print what the NN would classify the digits as
                print(int(classify(samples)))
    
    #if full number lower than 10m, add to wrongly classified list
    if check == True and int(classify(samples)) < 10000000:
        img_list.append([im, int(classify(samples))])
    #return all digits found
    return samples

#wrapper for white digits
def digits_read_white(im, check=False):
    return digits_read(im, 20, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV, check)

#wrapper for black digits
def digits_read_black(im, check=False):
    return digits_read(im, 17, cv2.THRESH_OTSU|cv2.THRESH_BINARY, check)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

#get list of found digits and runs it through NN
def classify(data):
    clas = []

    #run every found digit through NN
    for i in data:
        a = int(digits_model.predict([i])[0])
        if a == 11:
            a = 44
        clas.append(a)

    #reverse list and add all together in 1 integer to find final power
    clas.reverse()
    clas = map(str, clas)
    clas = ''.join(clas)
    return clas
#list for wrongly classified 
img_list = []

#load models
try:
	digits_model = pickle.load(open('digits_model.sav', 'rb'))
	names_model = pickle.load(open('names_model.sav', 'rb'))
except:
	print("No models found")
	sys.exit()

show_img = input("Want to show all images being classified? y/n ")
if show_img == 'y':
    show_img = True
else:
    show_img = False

dirs = listdir('TestingPictures/')

#get all images in kingdoms subdir
players = []
list_active = False
old_list_names = []
#loop over all images
for j in dirs:
    img_mask = f'TestingPictures/{j}/*.jpg'
    img_names = glob(img_mask)
    for fn in img_names:
        player = []

        #read image and zoom in on power
        img = cv2.imread(fn)
        img = img[0:1080, 0:2300]

        ####CHARS
        name = img[145:240, 650:1050]
        data = chars_read(name)
        data = ''.join(str(elem) for elem in data)
        player.append(data)

        ####DIGITS	
        power = img[150:220, 1186:1400]
        data = digits_read_white(power, True)
        data = int(classify(data))
        player.append(data)

        kills = img[150:220, 1588:1750]
        data = digits_read_white(kills)
        data = int(classify(data))
        player.append(data)

        kills_one = img[325:360, 1360:1490]
        data = digits_read_black(kills_one)
        data = int(classify(data))
        player.append(data)

        kills_two = img[325:360, 1540:1685]
        data = digits_read_black(kills_two)
        data = int(classify(data))
        player.append(data)

        kills_three = img[375:410, 1360:1490]
        data = digits_read_black(kills_three)
        data = int(classify(data))
        player.append(data)

        kills_four = img[375:410, 1540:1685]
        data = digits_read_black(kills_four)
        data = int(classify(data))
        player.append(data)

        kills_five = img[425:460, 1360:1490]
        data = digits_read_black(kills_five)
        data = int(classify(data))
        player.append(data)

        dead = img[520:590, 1610:1800]
        data = digits_read_white(dead)
        data = int(classify(data))
        player.append(data)

        rss_gath = img[730:790, 1570:1800]
        data = digits_read_white(rss_gath)
        data = int(classify(data))
        player.append(data)

        rss_ass = img[800:860, 1570:1800]
        data = digits_read_white(rss_ass)
        data = int(classify(data))
        player.append(data)

        alliance_help = img[870:930, 1650:1800]
        data = digits_read_white(alliance_help)
        data = int(classify(data))
        player.append(data)

        if j != 'new':
            players.append(player)
        else:
            no = True
            if list_active == False:
                old_list_names = [i[0] for i in players]
                list_active = True
            for i in players:
                sim = similar(i[0].lower(),player[0].lower())
                # if sim > 0.5:
                    # print(i[0], player[0], sim)
                if sim > 0.9:
                    a = player[1:]
                    for b in a:
                        i.append(b)
                    no = False
            if no: 
                if player[0] not in old_list_names:
                    for _ in range(0,6):
                        player.insert(1,0)
                    players.append(player)
                


players.insert(0,['Player name', 'Power (old)', 'Kills total (old)', 'Kills Tier 1 (old)', 'Kills Tier 2 (old)', 'Kills Tier 3 (old)', 'Kills Tier 4 (old)', 'Kills Tier 5 (old)', 'Dead (old)', 'Rss-gathered (old)','Rss-assistance (old)', 'Alliance help (old)', 'Power (new)', 'Kills total (new)', 'Kills Tier 1 (new)', 'Kills Tier 2 (new)', 'Kills Tier 3 (new)', 'Kills Tier 4 (new)', 'Kills Tier 5 (new)', 'Dead (new)', 'Rss-gathered (new)', 'Rss-assistance (new)', 'Alliance help (new)'])
#handle wrongly classified cases 
print(f"Could not read {len(img_list)} numbers. They will be shown to you, type them please!")
print("You can use enter to submit number and backspace to delete and escape to quit")

#loop over all wrongly classified images and let user enters manually
img_list_dict = {48:0, 49:1, 50:2, 51:3, 52:4, 53:5, 54:6, 55:7, 56:8, 57:9}
for it in img_list: 
    im = it[0]
    add = ""
    while True:
        cv2.imshow('View Digits', im)
        key = cv2.waitKey(0)
        back = False
        if key == 27:
            sys.exit()
        elif key == 8:
            back = True
        elif key == 13:
            print('\n')
            break
        elif key in img_list_dict.keys():
            number = img_list_dict[key]

        if not back:
            add = add + str(number)
        else:
            add = add[:-1]
        print(add)
    for i in players:
    	try:
    		a = i.index(it[1])
    		i[a] = int(add)
    	except:
    		pass

#save data as csv file
with open('total_stats.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in players:
    	wr.writerow(i)
