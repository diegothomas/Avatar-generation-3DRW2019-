# File created by Diego Thomas the 21-11-2016
# File overwrited by hayato Onizuka the 13-09-2017

# File to handle the menu main loop
import cv2
import Tkinter as tk
from PIL import ImageTk
from tkFileDialog import askopenfilename

from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) )

## Class to handle menu GUI
class Menu(tk.Frame):
    def key(self, event):
        if (event.keysym == 'Escape'):
            self.root.destroy()

    def callback(self, event):
        x = event.x
        y = event.y
        if (self.menu_label[y,x,2] == 255 and self.menu_label[y,x,1] == 0 and self.menu_label[y,x,0] == 0):
            self.filename = askopenfilename(filetypes=[('.ply','*.ply')]) # show an "Open" dialog box and return the path to the selected file
            if(len(self.filename) > 0):
                print(self.filename)
                self.root.destroy()
        elif (self.menu_label[y,x,2] == 0 and self.menu_label[y,x,1] == 255 and self.menu_label[y,x,0] == 0):
            self.root.destroy() # destroy the window
    
    def __init__(self, master=None):
        self.root = master
        self.filename = ""
        
        tk.Frame.__init__(self, master)
        self.pack()
        
        self.canvas = tk.Canvas(self, bg="white", height=768, width=1024)
        self.canvas.pack()
    
        #self.menu = ImageTk.PhotoImage(file = "./images/menu_image.jpeg")
        self.menu = ImageTk.PhotoImage(file = APP_ROOT+"/../images/menu_image.jpeg")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.menu)
        self.menu_label = cv2.imread(APP_ROOT+"/../images/menu_label.tiff")

        self.root.bind("<Key>", self.key)
        self.root.bind("<Button-1>", self.callback)
