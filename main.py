# import sys
# import os
# import tkinter as tk
#
# top=tk.Tk()
# top.title("Recognizer")
# top.iconbitmap()
# def addface():
#     os.system('faceses.py')
#
# def trainer():
#     os.system('trainer.py')
#
# def video_recog():
#     os.system('video_face_recog.py')
#
# Label(text="Имя:").grid(row=0, column=0, sticky=W, pady=10, padx=10)
# add_face=tk.Button(top,text="Add face",command= addface)
# train=tk.Button(top,text="train",command= trainer)
# video_rec=tk.Button(top,text="video_face_recognition",command= video_recog)
#
#
#
# add_face.pack()
# train.pack()
# video_rec.pack()
# top.mainloop()
import tkinter as tk
import os
from tkinter import *

root = Tk()
root.title("Recognizer")
def addface():
    os.system('faceses.py')

def trainer():
    os.system('trainer.py')

def video_recog():
    os.system('video_face_recog.py')

def save():
    file_name = "names.csv"
    with open(file_name , 'a+' ) as file_object:
        file_object.write(table_name.get()+"\n")

    os.system('faceses.py')


entry_field_variable = tk.StringVar()
root.configure(bg = "white")

Label(text="Name:" , bg='white').grid(row=0, column=0, sticky=W, pady=10, padx=10)
table_name = Entry(textvariable=entry_field_variable)
table_name.grid(row=0, column=1, columnspan=3, sticky=W + E, padx=10)

Label(text="Id:" , bg='white').grid(row=1, column=0, sticky=W, padx=10, pady=10)
table_column = Spinbox(width=7, from_=1, to=10)
table_column.grid(row=1, column=1, padx=10)


Button(text="Add face",command= save, bg='white').grid(row=2, column=0, pady=10, padx=10)
Button(text="Train ",command= trainer , bg='white').grid(row=2, column=2)
Button(text="Video recog",command= video_recog , bg='white').grid(row=2, column=3, padx=10)

root.mainloop()