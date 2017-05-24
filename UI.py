#! /usr/bin/env python
# coding: utf-8
# GUI module generated by PAGE version 4.9
# In conjunction with Tcl version 8.6
#    May 22, 2017 11:30:34 PM
import sys

try:
    from Tkinter import *
except ImportError:
    from tkinter import *

try:
    import ttk
    py3 = 0
except ImportError:
    import tkinter.ttk as ttk
    py3 = 1

import button_support

def vp_start_gui():
    #'Starting point when module is the main routine.'
    global val, w, root
    root = Tk()
    button_support.set_Tk_var()
    top = NeuralNetwork_IA(root)
    button_support.init(root, top)
    root.mainloop()

w = None
def NeuralNetwork_IA(root, *args, **kwargs):
    #'Starting point when module is imported by another program.'
    global w, w_win, rt
    rt = root
    w = Toplevel (root)
    button_support.set_Tk_var()
    top = NeuralNetwork_IA (w)
    button_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_NeuralNetwork_IA():
    global w
    w.destroy()
    w = None


class NeuralNetwork_IA:
    def __init__(self, top=None):
        #'This class configures and populates the toplevel window.
           #top is the toplevel containing window.'
        _bgcolor = 'wheat'  # X11 color: #f5deb3
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85' 
        _ana2color = '#d9d9d9' # X11 color: 'gray85' 
        font11 = "-family {DejaVu Sans} -size 14 -weight normal -slant"  \
            " roman -underline 0 -overstrike 0"
        font12 = "-family {DejaVu Sans} -size 20 -weight normal -slant"  \
            " roman -underline 0 -overstrike 0"
        font13 = "-family {DejaVu Sans} -size 12 -weight normal -slant"  \
            " roman -underline 0 -overstrike 0"
        font9 = "-family {DejaVu Sans} -size 0 -weight normal -slant "  \
            "roman -underline 0 -overstrike 0"
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("844x713+650+111")
        top.title("Red Neuronal - IA")
        top.configure(background="#e0ebf3")
        top.configure(highlightbackground="#912b33")
        top.configure(highlightcolor="black")



        self.menubar = Menu(top,font=font9,bg=_bgcolor,fg=_fgcolor)
        top.configure(menu = self.menubar)


        self.CostoIniLabel = Label(top)
        self.CostoIniLabel.place(relx=0.32, rely=0.43, height=38, width=156)
        self.CostoIniLabel.configure(activebackground="#f9f9f9")
        self.CostoIniLabel.configure(background="#ffffff")


        self.ClearLogButton = Button(top)
        self.ClearLogButton.place(relx=0.84, rely=0.03, height=27, width=105)
        self.ClearLogButton.configure(activebackground="#09d8f9")
        self.ClearLogButton.configure(background="#49c4ff")
        self.ClearLogButton.configure(highlightbackground="#09d8f9")
        self.ClearLogButton.configure(text='Limpiar Log')
        self.ClearLogButton.bind('<ButtonRelease-1>',button_support.ClearLog)


	#-------------------------------------------------------
	#------------ Frames Visualizacion ---------------------
        self.Frame1 = Frame(top)
        self.Frame1.place(relx=0.02, rely=0.18, relheight=0.005, relwidth=0.49)
        self.Frame1.configure(relief=GROOVE)
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief=GROOVE)
        self.Frame1.configure(background="#24306c")
        self.Frame1.configure(width=415)

        self.Frame2 = Frame(top)
        self.Frame2.place(relx=0.02, rely=0.29, relheight=0.005, relwidth=0.49)
        self.Frame2.configure(relief=GROOVE)
        self.Frame2.configure(borderwidth="2")
        self.Frame2.configure(relief=GROOVE)
        self.Frame2.configure(background="#24306c")
        self.Frame2.configure(width=415)

        self.Frame3 = Frame(top)
        self.Frame3.place(relx=0.02, rely=0.36, relheight=0.005, relwidth=0.49)
        self.Frame3.configure(relief=GROOVE)
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief=GROOVE)
        self.Frame3.configure(background="#24306c")
        self.Frame3.configure(width=415)

        self.Frame4 = Frame(top)
        self.Frame4.place(relx=0.51, rely=0.01, relheight=0.97, relwidth=0.005)
        self.Frame4.configure(relief=GROOVE)
        self.Frame4.configure(borderwidth="2")
        self.Frame4.configure(relief=GROOVE)
        self.Frame4.configure(background="#24306c")
        self.Frame4.configure(width=5)

        self.Frame5 = Frame(top)
        self.Frame5.place(relx=0.02, rely=0.5, relheight=0.005, relwidth=0.49)
        self.Frame5.configure(relief=GROOVE)
        self.Frame5.configure(borderwidth="2")
        self.Frame5.configure(relief=GROOVE)
        self.Frame5.configure(background="#24306c")
        self.Frame5.configure(width=415)

        self.Frame6 = Frame(top)
        self.Frame6.place(relx=0.02, rely=0.01, relheight=0.97, relwidth=0.005)
        self.Frame6.configure(relief=GROOVE)
        self.Frame6.configure(borderwidth="2")
        self.Frame6.configure(relief=GROOVE)
        self.Frame6.configure(background="#24306c")
        self.Frame6.configure(width=5)

        self.Frame8 = Frame(top)
        self.Frame8.place(relx=0.02, rely=0.98, relheight=0.005, relwidth=0.97)
        self.Frame8.configure(relief=GROOVE)
        self.Frame8.configure(borderwidth="2")
        self.Frame8.configure(relief=GROOVE)
        self.Frame8.configure(background="#24306c")
        self.Frame8.configure(width=815)

        self.Frame9 = Frame(top)
        self.Frame9.place(relx=0.02, rely=0.63, relheight=0.005, relwidth=0.49)
        self.Frame9.configure(relief=GROOVE)
        self.Frame9.configure(borderwidth="2")
        self.Frame9.configure(relief=GROOVE)
        self.Frame9.configure(background="#24306c")
        self.Frame9.configure(width=415)

        self.Frame10 = Frame(top)
        self.Frame10.place(relx=0.98, rely=0.01, relheight=0.97, relwidth=0.005)
        self.Frame10.configure(relief=GROOVE)
        self.Frame10.configure(borderwidth="2")
        self.Frame10.configure(relief=GROOVE)
        self.Frame10.configure(background="#24306c")
        self.Frame10.configure(width=5)

        self.Frame11 = Frame(top)
        self.Frame11.place(relx=0.02, rely=0.83, relheight=0.005, relwidth=0.49)
        self.Frame11.configure(relief=GROOVE)
        self.Frame11.configure(borderwidth="2")
        self.Frame11.configure(relief=GROOVE)
        self.Frame11.configure(background="#24306c")
        self.Frame11.configure(width=415)

        self.Frame12 = Frame(top)
        self.Frame12.place(relx=0.02, rely=0.01, relheight=0.005, relwidth=0.97)
        self.Frame12.configure(relief=GROOVE)
        self.Frame12.configure(borderwidth="2")
        self.Frame12.configure(relief=GROOVE)
        self.Frame12.configure(background="#24306c")
        self.Frame12.configure(width=815)
	#------------ Frames Visualizacion ---------------------
	#-------------------------------------------------------



	#-------------------------------------------------------
	#---------------- LABELS ESTATICAS ---------------------
        self.VarLabel1 = Label(top)
        self.VarLabel1.place(relx=0.04, rely=0.03, height=28, width=390)
        self.VarLabel1.configure(activebackground="#f9f9f9")
        self.VarLabel1.configure(background="#11cce4")
        self.VarLabel1.configure(font=font11)
        self.VarLabel1.configure(text='Variables')

        self.Label2 = Label(top)
        self.Label2.place(relx=0.04, rely=0.08, height=28, width=120)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(anchor=E)
        self.Label2.configure(background="#7cbcff")
        self.Label2.configure(highlightbackground="#b9d9d9")
        self.Label2.configure(text='Capas Entrada:')

        self.Label4 = Label(top)
        self.Label4.place(relx=0.04, rely=0.13, height=28, width=120)
        self.Label4.configure(activebackground="#f9f9f9")
        self.Label4.configure(anchor=E)
        self.Label4.configure(background="#7cbcff")
        self.Label4.configure(highlightbackground="#b9d9d9")
        self.Label4.configure(text='Capas Salida:')

        self.Label5 = Label(top)
        self.Label5.place(relx=0.3, rely=0.08, height=28, width=125)
        self.Label5.configure(activebackground="#f9f9f9")
        self.Label5.configure(anchor=E)
        self.Label5.configure(background="#7cbcff")
        self.Label5.configure(highlightbackground="#b9d9d9")
        self.Label5.configure(text='Capas escondidas:')

        self.Label9 = Label(top)
        self.Label9.place(relx=0.32, rely=0.38, height=28, width=66)
        self.Label9.configure(activebackground="#f9f9f9")
        self.Label9.configure(background="#7cbcff")
        self.Label9.configure(highlightbackground="#b9d9d9")
        self.Label9.configure(text='Lambda:')

        self.Label16 = Label(top)
        self.Label16.place(relx=0.32, rely=0.52, height=28, width=66)
        self.Label16.configure(activebackground="#f9f9f9")
        self.Label16.configure(background="#7cbcff")
        self.Label16.configure(highlightbackground="#b9d9d9")
        self.Label16.configure(text='Lambda:')

        self.Label14 = Label(top)
        self.Label14.place(relx=0.3, rely=0.2, height=26, width=84)
        self.Label14.configure(activebackground="#f9f9f9")
        self.Label14.configure(anchor=E)
        self.Label14.configure(background="#7cbcff")
        self.Label14.configure(highlightbackground="#b9d9d9")
        self.Label14.configure(text='% Training:')

        self.Label15 = Label(top)
        self.Label15.place(relx=0.3, rely=0.24, height=26, width=84)
        self.Label15.configure(activebackground="#f9f9f9")
        self.Label15.configure(anchor=E)
        self.Label15.configure(background="#7cbcff")
        self.Label15.configure(highlightbackground="#b9d9d9")
        self.Label15.configure(text='% Validation:')

        self.Label6 = Label(top)
        self.Label6.place(relx=0.04, rely=0.2, height=52, width=126)
        self.Label6.configure(activebackground="#f9f9f9")
        self.Label6.configure(background="#498bff")
        self.Label6.configure(highlightbackground="#b9d9d9")
        self.Label6.configure(text='Dataset')

        self.Label7 = Label(top)
        self.Label7.place(relx=0.04, rely=0.31, height=30, width=126)
        self.Label7.configure(activebackground="#f9f9f9")
        self.Label7.configure(background="#498bff")
        self.Label7.configure(highlightbackground="#b9d9d9")
        self.Label7.configure(text='Imagen')

        self.Label8 = Label(top)
        self.Label8.place(relx=0.04, rely=0.38, height=78, width=126)
        self.Label8.configure(activebackground="#f9f9f9")
        self.Label8.configure(background="#498bff")
        self.Label8.configure(highlightbackground="#b9d9d9")
        self.Label8.configure(text='Costo Inicial')

        self.Label10 = Label(top)
        self.Label10.place(relx=0.04, rely=0.52, height=28, width=126)
        self.Label10.configure(activebackground="#f9f9f9")
        self.Label10.configure(background="#498bff")
        self.Label10.configure(highlightbackground="#b9d9d9")
        self.Label10.configure(text='Gradiente Sigmoide')

        self.Label11 = Label(top)
        self.Label11.place(relx=0.20, rely=0.38, height=28, width=100)
        self.Label11.configure(activebackground="#f9f9f9")
        self.Label11.configure(background="#6063ff")
        self.Label11.configure(highlightbackground="#b9d9d9")
        self.Label11.configure(text='Regularizacion')

        self.Label17 = Label(top)
        self.Label17.place(relx=0.20, rely=0.52, height=28, width=100)
        self.Label17.configure(activebackground="#f9f9f9")
        self.Label17.configure(background="#6063ff")
        self.Label17.configure(highlightbackground="#b9d9d9")
        self.Label17.configure(text='Regularizacion')

        self.VarLabel3 = Label(top)
        self.VarLabel3.place(relx=0.04, rely=0.65, height=28, width=390)
        self.VarLabel3.configure(activebackground="#f9f9f9")
        self.VarLabel3.configure(background="#11cce4")
        self.VarLabel3.configure(font=font11)
        self.VarLabel3.configure(text='Entrenamiento')

        self.Label12 = Label(top)
        self.Label12.place(relx=0.04, rely=0.7, height=28, width=86)
        self.Label12.configure(activebackground="#f9f9f9")
        self.Label12.configure(background="#7cbcff")
        self.Label12.configure(highlightbackground="#b9d9d9")
        self.Label12.configure(text='Lambda:')

        self.Label12 = Label(top)
        self.Label12.place(relx=0.04, rely=0.75, height=28, width=86)
        self.Label12.configure(activebackground="#f9f9f9")
        self.Label12.configure(background="#7cbcff")
        self.Label12.configure(highlightbackground="#b9d9d9")
        self.Label12.configure(text='Iteraciones:')

        self.Label13 = Label(top)
        self.Label13.place(relx=0.24, rely=0.7, height=28, width=146)
        self.Label13.configure(activebackground="#f9f9f9")
        self.Label13.configure(background="#7cbcff")
        self.Label13.configure(highlightbackground="#b9d9d9")
        self.Label13.configure(text='Cantidad de muestras:')

        self.VarLabel4 = Label(top)
        self.VarLabel4.place(relx=0.04, rely=0.84, height=28, width=390)
        self.VarLabel4.configure(activebackground="#f9f9f9")
        self.VarLabel4.configure(background="#11cce4")
        self.VarLabel4.configure(font=font11)
        self.VarLabel4.configure(text='Guardar Datos')

        self.VarLabel2 = Label(top)
        self.VarLabel2.place(relx=0.53, rely=0.03, height=28, width=246)
        self.VarLabel2.configure(activebackground="#f9f9f9")
        self.VarLabel2.configure(background="#aed1ff")
        self.VarLabel2.configure(font=font11)
        self.VarLabel2.configure(text='Log')
        self.VarLabel2.configure(width=246)

        self.InputL_Label1 = Label(top)
        self.InputL_Label1.place(relx=0.19, rely=0.08, height=28, width=36)
        self.InputL_Label1.configure(activebackground="#f9f9f9")
        self.InputL_Label1.configure(background="#ffffff")
        self.InputL_Label1.configure(text='784')

        self.OutputL_Label1 = Label(top)
        self.OutputL_Label1.place(relx=0.19, rely=0.13, height=28, width=36)
        self.OutputL_Label1.configure(activebackground="#f9f9f9")
        self.OutputL_Label1.configure(background="#ffffff")
        self.OutputL_Label1.configure(text='10')
	#-------------------------------------------------------
	#---------------- LABELS ESTATICAS ---------------------

        self.ChangeHiddenButton1 = Button(top)
        self.ChangeHiddenButton1.place(relx=0.3, rely=0.13, height=26, width=132)

        self.ChangeHiddenButton1.configure(activebackground="#09d8f9")
        self.ChangeHiddenButton1.configure(background="#49c4ff")
        self.ChangeHiddenButton1.configure(highlightbackground="#09d8f9")
        self.ChangeHiddenButton1.configure(text='Modificar')
        self.ChangeHiddenButton1.bind('<ButtonRelease-1>',button_support.ChangeHidden)

        self.LoadDatasetButton1 = Button(top)
        self.LoadDatasetButton1.place(relx=0.20, rely=0.2, height=57, width=82)
        self.LoadDatasetButton1.configure(activebackground="#09d8f9")
        self.LoadDatasetButton1.configure(background="#49c4ff")
        self.LoadDatasetButton1.configure(highlightbackground="#09d8f9")
        self.LoadDatasetButton1.configure(text='Cargar')
        self.LoadDatasetButton1.bind('<ButtonRelease-1>',button_support.LoadDataset)

        self.LoadDatasetButton3 = Button(top)
        self.LoadDatasetButton3.place(relx=0.20, rely=0.31, height=30, width=255)

        self.LoadDatasetButton3.configure(activebackground="#09d8f9")
        self.LoadDatasetButton3.configure(background="#49c4ff")
        self.LoadDatasetButton3.configure(highlightbackground="#09d8f9")
        self.LoadDatasetButton3.configure(text='Mostrar')
        self.LoadDatasetButton3.bind('<ButtonRelease-1>',button_support.ShowRandomImage)

        self.SetPrcntButton1 = Button(top)
        self.SetPrcntButton1.place(relx=0.46, rely=0.2, height=57, width=35)
        self.SetPrcntButton1.configure(activebackground="#09d8f9")
        self.SetPrcntButton1.configure(background="#49c4ff")
        self.SetPrcntButton1.configure(font=font12)
        self.SetPrcntButton1.configure(highlightbackground="#09d8f9")
        self.SetPrcntButton1.configure(text='✓')
        self.SetPrcntButton1.bind('<ButtonRelease-1>',button_support.SetNewDistribution)

        self.CalcCostoButton1 = Button(top)
        self.CalcCostoButton1.place(relx=0.20, rely=0.43, height=37, width=100)
        self.CalcCostoButton1.configure(activebackground="#09d8f9")
        self.CalcCostoButton1.configure(background="#49c4ff")
        self.CalcCostoButton1.configure(highlightbackground="#09d8f9")
        self.CalcCostoButton1.configure(text='Calcular')
        self.CalcCostoButton1.bind('<ButtonRelease-1>',button_support.CalcCost)

        self.SigGradButton1 = Button(top)
        self.SigGradButton1.place(relx=0.4, rely=0.58, height=27, width=85)
        self.SigGradButton1.configure(activebackground="#09d8f9")
        self.SigGradButton1.configure(background="#49c4ff")
        self.SigGradButton1.configure(highlightbackground="#09d8f9")
        self.SigGradButton1.configure(text='Calcular')
        self.SigGradButton1.configure(width=85)
        self.SigGradButton1.bind('<ButtonRelease-1>',button_support.CalcSigmoidGrad)

        self.TrainStartButton1 = Button(top)
        self.TrainStartButton1.place(relx=0.22, rely=0.75, height=35, width=235)
        self.TrainStartButton1.configure(activebackground="#09d8f9")
        self.TrainStartButton1.configure(background="#49c4ff")
        self.TrainStartButton1.configure(font=font13)
        self.TrainStartButton1.configure(highlightbackground="#09d8f9")
        self.TrainStartButton1.configure(text='Entrenar')
        self.TrainStartButton1.bind('<ButtonRelease-1>',button_support.TrainingStart)

        self.SaveButton = Button(top)
        self.SaveButton.place(relx=0.04, rely=0.91, height=37, width=390)
        self.SaveButton.configure(activebackground="#09d8f9")
        self.SaveButton.configure(background="#49c4ff")
        self.SaveButton.configure(font=font13)
        self.SaveButton.configure(highlightbackground="#09d8f9")
        self.SaveButton.configure(text='Guardar Ultimos Resultados')
        self.SaveButton.bind('<ButtonRelease-1>',button_support.SaveData)

        self.ValidPrcntLabel = Label(top)
        self.ValidPrcntLabel.place(relx=0.405, rely=0.24, height=28, width=38)
        self.ValidPrcntLabel.configure(activebackground="#f9f9f9")
        self.ValidPrcntLabel.configure(background="#ffffff")
        self.ValidPrcntLabel.configure(text='40')

        self.HiddenL_Label1 = Label(top)
        self.HiddenL_Label1.place(relx=0.46, rely=0.08, height=28, width=36)
        self.HiddenL_Label1.configure(activebackground="#f9f9f9")
        self.HiddenL_Label1.configure(background="#ffffff")
        self.HiddenL_Label1.configure(text='25')

        self.HiddenEntry1 = Entry(top)
        self.HiddenEntry1.place(relx=0.46, rely=0.13, relheight=0.04
                , relwidth=0.04)
        self.HiddenEntry1.configure(background="white")
        self.HiddenEntry1.configure(font="TkFixedFont")
        self.HiddenEntry1.configure(selectbackground="#c4c4c4")
        self.HiddenEntry1.configure(textvariable=button_support.HiddenEntryVar)

        self.TrainPrcntEntry1 = Entry(top)
        self.TrainPrcntEntry1.place(relx=0.402, rely=0.2, relheight=0.04
                , relwidth=0.05)
        self.TrainPrcntEntry1.configure(background="white")
        self.TrainPrcntEntry1.configure(font="TkFixedFont")
        self.TrainPrcntEntry1.configure(selectbackground="#c4c4c4")
        self.TrainPrcntEntry1.configure(textvariable=button_support.TrainPrcntEntryVar)

        self.CostoLambdaEntry1 = Entry(top)
        self.CostoLambdaEntry1.place(relx=0.4, rely=0.38, relheight=0.04
                , relwidth=0.1)
        self.CostoLambdaEntry1.configure(background="white")
        self.CostoLambdaEntry1.configure(font="TkFixedFont")
        self.CostoLambdaEntry1.configure(selectbackground="#c4c4c4")
        self.CostoLambdaEntry1.configure(textvariable=button_support.CostoLmbdaEntryVar)

        self.GSLambdaEntry1 = Entry(top)
        self.GSLambdaEntry1.place(relx=0.4, rely=0.52, relheight=0.04
                , relwidth=0.1)
        self.GSLambdaEntry1.configure(background="white")
        self.GSLambdaEntry1.configure(font="TkFixedFont")
        self.GSLambdaEntry1.configure(selectbackground="#c4c4c4")
        self.GSLambdaEntry1.configure(textvariable=button_support.GSLambdaEntryVar)

        self.SGEntry1 = Entry(top)
        self.SGEntry1.place(relx=0.04, rely=0.58, relheight=0.04, relwidth=0.358)
        self.SGEntry1.configure(background="white")
        self.SGEntry1.configure(font="TkFixedFont")
        self.SGEntry1.configure(selectbackground="#c4c4c4")
        self.SGEntry1.configure(textvariable=button_support.SGEntryVar)

        self.TrainLambdaEntry1 = Entry(top)
        self.TrainLambdaEntry1.place(relx=0.145, rely=0.7, relheight=0.04
                , relwidth=0.09)
        self.TrainLambdaEntry1.configure(background="white")
        self.TrainLambdaEntry1.configure(font="TkFixedFont")
        self.TrainLambdaEntry1.configure(selectbackground="#c4c4c4")
        self.TrainLambdaEntry1.configure(textvariable=button_support.TrainLmbdaVar)

        self.TrainItersEntry = Entry(top)
        self.TrainItersEntry.place(relx=0.145, rely=0.75, relheight=0.04
                , relwidth=0.07)
        self.TrainItersEntry.configure(background="white")
        self.TrainItersEntry.configure(font="TkFixedFont")
        self.TrainItersEntry.configure(selectbackground="#c4c4c4")
        self.TrainItersEntry.configure(textvariable=button_support.TrainIterVar)

        self.TrainSamplesEntry1 = Entry(top)
        self.TrainSamplesEntry1.place(relx=0.41, rely=0.7, relheight=0.04
                , relwidth=0.09)
        self.TrainSamplesEntry1.configure(background="white")
        self.TrainSamplesEntry1.configure(font="TkFixedFont")
        self.TrainSamplesEntry1.configure(selectbackground="#c4c4c4")
        self.TrainSamplesEntry1.configure(textvariable=button_support.TrainSamplesVar)

        self.LogBox = ScrolledText(top)
        self.LogBox.place(relx=0.53, rely=0.08, relheight=0.88, relwidth=0.44)
        self.LogBox.configure(background="white")
        self.LogBox.configure(font="TkTextFont")
        self.LogBox.configure(highlightbackground="wheat")
        self.LogBox.configure(insertborderwidth="3")
        self.LogBox.configure(selectbackground="#c4c4c4")
        self.LogBox.configure(state=DISABLED)
        self.LogBox.configure(width=10)
        self.LogBox.configure(wrap=NONE)





# The following code is added to facilitate the Scrolled widgets you specified.
class AutoScroll(object):
    #'Configure the scrollbars for a widget.'

    def __init__(self, master):
        #  Rozen. Added the try-except clauses so that this class
        #  could be used for scrolled entry widget for which vertical
        #  scrolling is not supported. 5/7/14.
        try:
            vsb = ttk.Scrollbar(master, orient='vertical', command=self.yview)
        except:
            pass
        hsb = ttk.Scrollbar(master, orient='horizontal', command=self.xview)

        #self.configure(yscrollcommand=_autoscroll(vsb),
        #    xscrollcommand=_autoscroll(hsb))
        try:
            self.configure(yscrollcommand=self._autoscroll(vsb))
        except:
            pass
        self.configure(xscrollcommand=self._autoscroll(hsb))

        self.grid(column=0, row=0, sticky='nsew')
        try:
            vsb.grid(column=1, row=0, sticky='ns')
        except:
            pass
        hsb.grid(column=0, row=1, sticky='ew')

        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(0, weight=1)

        # Copy geometry methods of master  (taken from ScrolledText.py)
        if py3:
            methods = Pack.__dict__.keys() | Grid.__dict__.keys() \
                  | Place.__dict__.keys()
        else:
            methods = Pack.__dict__.keys() + Grid.__dict__.keys() \
                  + Place.__dict__.keys()

        for meth in methods:
            if meth[0] != '_' and meth not in ('config', 'configure'):
                setattr(self, meth, getattr(master, meth))

    @staticmethod
    def _autoscroll(sbar):
        #'Hide and show scrollbar as needed.'
        def wrapped(first, last):
            first, last = float(first), float(last)
            if first <= 0 and last >= 1:
                sbar.grid_remove()
            else:
                sbar.grid()
            sbar.set(first, last)
        return wrapped

    def __str__(self):
        return str(self.master)

def _create_container(func):
    #'Creates a ttk Frame with a given master, and use this new frame to
    #place the scrollbars and the widget.'
    def wrapped(cls, master, **kw):
        container = ttk.Frame(master)
        return func(cls, container, **kw)
    return wrapped

class ScrolledText(AutoScroll, Text):
    #'A standard Tkinter Text widget with scrollbars that will
    #automatically show/hide as needed.'
    @_create_container
    def __init__(self, master, **kw):
        Text.__init__(self, master, **kw)
        AutoScroll.__init__(self, master)

if __name__ == '__main__':
    vp_start_gui()


