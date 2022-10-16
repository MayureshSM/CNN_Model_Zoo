#pylint:disable=W0104
import tkinter as tk
import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from PIL import Image
from tkinter import filedialog

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('IMAGE    CLASSIFICATION')
        self.MainFrame()
        self.LeftFrame()
        self.RightFrame()
        self.TaskFrame()
        self.ModelFrame()
        self.FileFrame()
        tk.Button(self.leftframe,text='PREDICT',width=15,command=self.CanvasFrame).grid(row=3,column=0)
        self.M=models.Object_classification()
    
    def CanvasFrame(self):
        try:
            self.canvasframe.destroy()
            self.canvasframe.grid_forget()
            self.predframe.destroy()
            self.predframe.grid_forget()
        except:
            pass
            
        self.canvasframe=tk.Frame(self.rightframe,width=1420,height=1000,highlightbackground='black' , highlightthickness=2)
        self.canvasframe.grid(row=0,column=0)
        self.canvasframe.grid_propagate(False)
        FILENAME="/storage/emulated/0/Documents/Repos/Images/images4.jpg"
        self.fig=plt.figure(figsize=(15,10))
        plot1=self.fig.add_subplot(111)
        img=Image.open(self.filename)
        img = img.resize((1420, 1100), Image.ANTIALIAS)
        plot1.imshow(img)
        self.canvas=FigureCanvasTkAgg(self.fig,master=self.canvasframe)
        self.canvas.get_tk_widget().grid(row=0,column=0,sticky=tk.W)
        
        self.predframe=tk.Frame(self.rightframe,width=1420,height=70,highlightbackground='black' , highlightthickness=2)
        self.predframe.grid(row=1)
        self.predframe.grid_propagate(False)
        self.label=tk.Label(self.predframe,text=self.M.model(self.model.get(),img))
        self.label.grid(row=0)
        
    def MainFrame(self):    
            self.mainframe=tk.Frame(self,height=1100,width=1920,bg='#008080')
            self.mainframe.grid(row=0,column=0,sticky=tk.W)
            self.mainframe.grid_propagate(False)
    
    def LeftFrame(self):    
            self.leftframe=tk.Frame(self.mainframe,width=500,height=1100,highlightbackground='black' , highlightthickness=2,bg='#008080')
            self.leftframe.grid(row=0,column=0,sticky=tk.W)
            self.leftframe.grid_propagate(False)
            
    def RightFrame(self):
            self.rightframe=tk.Frame(self.mainframe,width=1420,height=1100,highlightbackground='black' , highlightthickness=2)
            self.rightframe.grid(row=0,column=1,sticky=tk.W)
            self.rightframe.grid_propagate(False)
   
   #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def TaskFrame(self):
        self.taskframe=tk.Frame(self.leftframe,width=490,height=100,highlightbackground='black' , highlightthickness=2) 
        self.taskframe.grid(row=0,column=0,sticky=tk.W,padx=3,pady=3)
        self.taskframe.grid_propagate(False)
        tk.Label(self.taskframe,text='CV OBJECTIVE').grid(row=0,column=0,sticky=tk.W)
        self.tasklist=['Classification','Detection','Semantic Seg.','Instance Seg.']
        self.task=tk.StringVar(self)
        self.task.set(self.tasklist[0])
        self.Task=tk.OptionMenu(self.taskframe, self.task, *self.tasklist)
        self.Task.config(width=15)
        self.Task.grid(row=1,column=0,sticky=tk.W)
        tk.Button(self.taskframe,text="Submit",command=self.select_model).grid(row=1,column=1)
    
   #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def ModelFrame(self):
        self.modelframe=tk.Frame(self.leftframe,width=490,height=100,highlightbackground='black' , highlightthickness=2)
        self.modelframe.grid(row=1,column=0,sticky=tk.W,padx=3,pady=3)
        self.modelframe.grid_propagate(False)
    
        tk.Label(self.modelframe,text='SELECT  MODEL').grid(row=0,column=0,sticky=tk.W)
        self.modellist=['None']
        self.model=tk.StringVar(self)
        self.model.set(self.tasklist[0])
        self.Model=tk.OptionMenu(self.modelframe, self.model, *self.modellist)
        self.Model.config(width=15)
        self.Model.grid(row=1,column=0,sticky=tk.W)
    
    def FileFrame(self):
        self.fileframe=tk.Frame(self.leftframe,width=490,height=100,highlightbackground='black' , highlightthickness=2)
        self.fileframe.grid(row=2,column=0,sticky=tk.W,padx=3,pady=3)
        self.fileframe.grid_propagate(False)
        tk.Label(self.fileframe,text='SELECT  IMAGE').grid(row=0,column=0,sticky=tk.W)
        tk.Button(self.fileframe,text='BROWSE IMAGE',width=18,command=self.browsefile).grid(row=1,column=0,sticky=tk.W)
        self.filelabel=tk.Label(self.fileframe,text='NONE')
        self.filelabel.grid(row=1,column=1,sticky=tk.W)
        
    def browsefile(self):
        self.filename=filedialog.askopenfilename() 
        self.filelabel.configure(text=self.filename.split('/')[-1])     
        
    def select_model(self):
         if self.task.get()=='Classification':
             self.modellist=dir(models.Object_classification)
             self.modellist=[i for i in self.modellist if i[0]!="_"]
             menu=self.Model["menu"]
             menu.delete(0,"end")
             for s in self.modellist:
                 menu.add_command(label=s, command=lambda value=s:self.model.set(value))
         else:
             self.modellist=['None']
             menu=self.Model["menu"]
             menu.delete(0,"end")
             for s in self.modellist:
                 menu.add_command(label=s, command=lambda value=s:self.model.set(value))
             
             
             
A=GUI()
A.mainloop()        