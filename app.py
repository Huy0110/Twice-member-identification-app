from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
import numpy as np
import cv2 as cv

root = Tk()
root.title('App')
#root.geometry("800x600")
#root.iconbitmap('c:/gui/codemy.ico')




def open1():
	path = filedialog.askopenfilename(title="Select A File", filetypes=(("jpg files", "*.jpg"),("all files", "*.*")))
	my_label = Label(root, text=path)
	global pic_path
	pic_path = path
	#my_label.grid(column = 1, row = 0)
	my_label.grid(column = 1, row = 0)

def detect(pic_path):
	# cac thao tac
	haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
	people = ['Nayeon', 'Jeongyeon', 'Momo', 'Sana', 'Jihyo', 'Mina', 'Dahyun' , 'Chaeyoung', 'Tzuyu']
	# features = np.load('features.npy', allow_pickle=True)
	# # labels = np.load('labels.npy')
	
	face_recognizer = cv.face.LBPHFaceRecognizer_create()
	face_recognizer.read('face_trained.yml')
	img = cv.imread(pic_path)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	#cv.imshow('Person', gray)
	# Detect the face in the image
	faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
	labels = []
	confidences = []
	for (x,y,w,h) in faces_rect:
		   faces_roi = gray[y:y+h,x:x+w]
		   label, confidence = face_recognizer.predict(faces_roi)
		   labels.append(people[label])
		   confidences.append(confidence)
		   print(f'Label = {people[label]} with a confidence of {confidence}')
		   cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
		   cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
	cv.imshow('Detected Face', img)
	cv.waitKey(0)
	open_Window(pic_path, labels, confidences)

def open_Window(pic_path, labels, confidences):
	global my_img
	top = Toplevel()
	top.title('Result')
	global a
	my_label1 = Label(top, text='Label:')
	e_labels = Entry(top, width = 80, borderwidth = 5)
	for i, label in enumerate(labels):
		current = e_labels.get()
		e_labels.delete(0, END)
		e_labels.insert(0, str(current) + ' , ' + str(label))
		
	my_label2 = Label(top, text='Confidence score:')
	e_confidences = Entry(top, width = 80, borderwidth = 5)
	for i, confidence in enumerate(confidences):
		current = e_confidences.get()
		e_confidences.delete(0, END)
		e_confidences.insert(0, str(current) + ' , ' + str(confidence)[0:5])
	my_label1.grid(column = 0, row = 0)
	e_labels.grid(column=1,row = 0, columnspan=3, padx=10, pady=10)
	my_label2.grid(column = 0, row = 1)
	e_confidences.grid(column=1,row = 1, columnspan=3, padx=10, pady=10)
	btn2 = Button(top, text="close window", command=top.destroy)
	btn2.grid(column = 0, row = 3, columnspan=4)

my_btn1 = Button(root, text="Select pic",  padx=40, pady=20, command=open1)
btn = Button(root, text="Detect",  padx=40, pady=20, command= lambda: detect(pic_path))
my_btn1.grid(column = 0, row = 0)
btn.grid(column = 0, row = 1, columnspan=2)



root.mainloop()