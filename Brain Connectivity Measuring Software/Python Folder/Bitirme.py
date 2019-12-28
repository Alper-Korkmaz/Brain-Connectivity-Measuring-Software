import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()
import tkinter
import networkx as nx
import os
import time
import threading
import pandas as pd
from threading import Thread
from tkinter import filedialog
from scipy import stats
from scipy.stats import zscore
from numpy import linalg as LA


top = tkinter.Tk()

top.title("The Brain Functionality Visualizer")


top.geometry("860x260")



nodes = dict()

method = 5
image = cv2.imread('brain2.jpg')
image2 = cv2.imread('brain2method.jpg')
image3 = cv2.imread('brain2.jpg')
image4 = cv2.imread('brain2.jpg')
image6 = cv2.imread('brain2.jpg')
original = cv2.imread('brain2.jpg')


sem = threading.Semaphore()

def showim():
    while(True):
        sem.acquire()
        cv2.imshow("image",image)
        sem.release()
        cv2.waitKey(10)


#imWindow = Thread(target = showim, args = ()).start()


im = Image.open('brain2.jpg')
im2 = Image.open('brain2method.jpg')

width, height = im.size
width2, height2 = im2.size

x = int(width/2)
Gx=int(width/2)
Gy = int(height/2)
y = int(height/2)

w = int(width2/2)
h = int(height2/2)

def Method1():
    
    allNums = []
    with open(fileDialog(), "r+") as f:
            data = f.readlines()
            for line in data:
                allNums += line.strip().split(" ")

                CoorX1 = int(allNums[0]) + x
                CoorY1 = y - int(allNums[1])
                CoorX2 = int(allNums[2]) + x
                CoorY2 = y - int(allNums[3])

                Thickness1 = int(allNums[4])

                ArrowOrNot = int(allNums[5])



                cv2.circle(image, (CoorX1,CoorY1), 5, (0,0,255), -1)
                cv2.circle(image, (CoorX2,CoorY2), 5, (0,0,255), -1)

                pt1 = (CoorX1, CoorY1)
                pt2 = (CoorX2, CoorY2)

                if(ArrowOrNot == 1):
                    cv2.arrowedLine(image, pt1, pt2, (0,255,0), Thickness1)


                else:
                    if(ArrowOrNot == 0):
                        cv2.line(image, pt1, pt2, (0,255,0), Thickness1)



                allNums=[]
            Display1()




def Method2():
    
    allNums2 = []
    #B = ['V1', 'V2', 'V3', 'V3A','V3B', 'V4', 'V5', 'V7','LO1', 'IPS1', 'IPS2']

    CoorX1 = 290 + w #V1
    CoorY1 = h + 50
    pt1 = (CoorX1, CoorY1)


    CoorX2 = 260 + w #V2
    CoorY2 = h + 70
    pt2 = (CoorX2, CoorY2)


    CoorX3 = 230 + w #V3
    CoorY3 = h + 20
    pt3 = (CoorX3, CoorY3)
    

    CoorX4 = 190 + w #V3A
    CoorY4 = h - 10
    pt4 = (CoorX4, CoorY4)
    

    CoorX5 = 150 + w #V3B
    CoorY5 = h - 10
    pt5 = (CoorX5, CoorY5)
    

    CoorX6 = 190 + w #V4
    CoorY6 = h + 140
    pt6 = (CoorX6, CoorY6)



    CoorX7 = 160 + w #V5
    CoorY7 = h + 110
    pt7 = (CoorX7, CoorY7)
    
    
    CoorX8 = 170 + w #V7
    CoorY8 = h - 50
    pt8 = (CoorX8, CoorY8)
    

    CoorX9 = 195 + w #LO1
    CoorY9 = h + 100
    pt9 = (CoorX9, CoorY9)
    

    CoorX10 = 150 + w #IPS1
    CoorY10 = h - 120
    pt10 = (CoorX10, CoorY10)
    

    CoorX11 = 110 + w #IPS2
    CoorY11 = h - 160
    pt11 = (CoorX11, CoorY11)

    
    with open(fileDialog(), "r+") as f:
        data = f.readlines()
        for line in data:

            allNums2 += line.strip().split(" ")

            Node1 = int(allNums2[0])
            Node2 = int(allNums2[1])

            Thickness2 = int(allNums2[2])

            ArrowOrNot = int(allNums2[3])



            if(Node1 == 1):
                ptX = pt1

            if(Node1 == 2):
                ptX = pt2

            if(Node1 == 3):
                ptX = pt3

            if(Node1 == 4):
                ptX = pt4

            if(Node1 == 5):
                ptX = pt5

            if(Node1 == 6):
                ptX = pt6

            if(Node1 == 7):
                ptX = pt7

            if(Node1 == 8):
                ptX = pt8

            if(Node1 == 9):
                ptX = pt9

            if(Node1 == 10):
                ptX = pt10

            if(Node1 == 11):
                ptX = pt11






            if(Node2 == 1):
                ptY = pt1

            if(Node2 == 2):
                ptY = pt2

            if(Node2 == 3):
                ptY = pt3

            if(Node2 == 4):
                ptY = pt4

            if(Node2 == 5):
                ptY = pt5

            if(Node2 == 6):
                ptY = pt6

            if(Node2 == 7):
                ptY = pt7

            if(Node2 == 8):
                ptY = pt8

            if(Node2 == 9):
                ptY = pt9

            if(Node2 == 10):
                ptY = pt10

            if(Node2 == 11):
                ptY = pt11



            cv2.circle(image2, ptX, 5, (0,0,255), -1)
            cv2.circle(image2, ptY, 5, (0,0,255), -1)


            if(ArrowOrNot == 1):
                cv2.arrowedLine(image2, ptX, ptY, (0,255,0), Thickness2)

            else:

                if(ArrowOrNot == 0):
                    cv2.line(image2, ptX, ptY, (0,255,0), Thickness2)


            allNums2=[]
        Display2()


    

def Display1():

        blur = cv2.blur(image,(3,1))

        #blur = cv2.bilateralFilter(image,12,140,100)
        #blur = cv2.medianBlur(image,9)

        cv2.imshow('Brain', blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)



def Display2():

        blur2 = cv2.blur(image2,(3,1))

        cv2.imshow('Brain', blur2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)



def fileDialog():
    textfile = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select A File", filetype = (("text", "*.txt"), ("All Files", "*.*")))
    return textfile

def fileDialog2():
    textfile = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select A File", filetype = (("excel", "*.xlsx"), ("All Files", "*.*")))
    return textfile


def Method3():
    g = nx.Graph()

    Ex = pd.read_excel(fileDialog2(), header = None)
    A = Ex.as_matrix()
   
    B = ['V1', 'V2', 'V3', 'V3A','V3B', 'V4', 'V5', 'V7','LO1', 'IPS1', 'IPS2']
    nodes = np.arange(1.5, (len(A) * 1.5) + 1.5, 1.5)
    Local_Peak = np.random.randint(1, size=(len(A) * 5, 2))

    row = 0
    column = 0

    print(A)

    while(column < len(A.T)):
        plt.plot(nodes, A[:,column], label = B[column])
        column += 1
 
        
    plt.legend() 

    plt.savefig('Modularity_Graph.png')
    image5 = cv2.imread('Modularity_Graph.png')
    cv2.imshow('Modularity Graph', image5)
    os.remove('Modularity_Graph.png')
    plt.clf()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


    column = 0
    local_row = 0
    #local_col = 0


    while(column < len(A.T)):
        row = 1
        #local_row = 0
        while (row < (len(A) - 1)):
            if(A[row][column] > A[row - 1][column] ):
                if(A[row][column] > A[row + 1][column]):
                    Local_Peak[local_row][0] = int(column)
                    Local_Peak[local_row][1] = int(row)
                    local_row += 1
            row += 1
        column += 1
    
    print("Array is : \n", Local_Peak)


    Adj_Mtx = np.random.randint(1, size=(len(B),len(B)))

    local_row = 0
    #local_col = 0

    #term = 0
    print(Adj_Mtx)

    matrow = 0
    matcol = 0
    value = 0
    scan = 0

    while(local_row < len(Local_Peak)):
        value = Local_Peak[local_row][1]
        scan = local_row + 1
        while(scan < len(Local_Peak)):
            if(Local_Peak[scan][1] != 0):
                if(Local_Peak[scan][1] == value):
                    matrow = Local_Peak[local_row][0] 
                    matcol = Local_Peak[scan][0]
                    Adj_Mtx[matrow][matcol] += 1
                    Adj_Mtx[matcol][matrow] += 1
            scan += 1
        local_row += 1

    print(Adj_Mtx)

    matrow = 0
    matcol = 0

    while (matrow < len(Adj_Mtx)):
        matcol = 0
        while(matcol < len(Adj_Mtx)):
            if(Adj_Mtx[matrow][matcol] != 0):
                g.add_edges_from([(matrow, matcol)] , weight = Adj_Mtx[matrow][matcol])
            matcol += 1
        matrow += 1
    
    matrow = 0
    matcol = 0
    while(matrow < len(Adj_Mtx)):
        matcol = 0
        while(matcol < len(Adj_Mtx.T)):
        
            Graph = nx.betweenness_centrality(g)
            matcol += 1
        matrow += 1
    

    print("Betweenness Centrality is: \n", Graph)

    #print(Graph.items())
    print(Graph[1])
    
    index = 0

    CoorX1 = 290 + w #V1
    CoorY1 = h + 50
    #pt1 = (CoorX1, CoorY1)


    CoorX2 = 260 + w #V2
    CoorY2 = h + 70
    #pt2 = (CoorX2, CoorY2)


    CoorX3 = 230 + w #V3
    CoorY3 = h + 20
    #pt3 = (CoorX3, CoorY3)

    CoorX4 = 190 + w #V3A
    CoorY4 = h - 10
    #pt4 = (CoorX4, CoorY4)

    CoorX5 = 150 + w #V3B
    CoorY5 = h - 10
    #pt5 = (CoorX5, CoorY5)

    CoorX6 = 190 + w #V4
    CoorY6 = h + 140
    #pt6 = (CoorX6, CoorY6)


    CoorX7 = 160 + w #V5
    CoorY7 = h + 110
    #pt7 = (CoorX7, CoorY7)


    CoorX8 = 170 + w #V7
    CoorY8 = h - 50
    #pt8 = (CoorX8, CoorY8)

    CoorX9 = 195 + w #LO1
    CoorY9 = h + 100
    #pt9 = (CoorX9, CoorY9)

    CoorX10 = 150 + w #IPS1
    CoorY10 = h - 120
    #pt10 = (CoorX10, CoorY10)

    CoorX11 = 110 + w #IPS2
    CoorY11 = h - 160
    #pt11 = (CoorX11, CoorY11)


    rate = 0
    coo1x = 0
    coo1y = 0
    Rate_Arr = np.random.randint(1, size =(len(B)))
    
    while(index < len(B)):
        if(B[index] == 'V1'):
            rate = int(round(Graph[0],3) * 1000)
            coo1x = CoorX1
            coo1y = CoorY1
            Rate_Arr[index] = rate
            #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        if(B[index] == 'V2'):
            rate = int(round(Graph[1],3) * 1000)
            coo1x = CoorX2
            coo1y = CoorY2
            Rate_Arr[index] = rate
            #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        if(B[index] == 'V3'):
            rate = int(round(Graph[2],3) * 1000)
            coo1x = CoorX3
            coo1y = CoorY3
            Rate_Arr[index] = rate
            #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        if(B[index] == 'V3A'):
            rate = int(round(Graph[3],3) * 1000)
            coo1x = CoorX4
            coo1y = CoorY4
            Rate_Arr[index] = rate
            #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        if(B[index] == 'V3B'):
            rate = int(round(Graph[4],3) * 1000)
            coo1x = CoorX5
            coo1y = CoorY5
            Rate_Arr[index] = rate
            #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        if(B[index] == 'V4'):
            rate = int(round(Graph[5],3) * 1000)
            coo1x = CoorX6
            coo1y = CoorY6
            Rate_Arr[index] = rate
            #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        if(B[index] == 'V5'):
            rate = int(round(Graph[6],3) * 1000)
            coo1x = CoorX7
            coo1y = CoorY7
            Rate_Arr[index] = rate
            #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        if(B[index] == 'V7'):
            rate = int(round(Graph[7],3) * 1000)
            coo1x = CoorX8
            coo1y = CoorY8
            Rate_Arr[index] = rate
            #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        if(B[index] == 'LO1'):
            rate = int(round(Graph[8],3) * 1000)
            coo1x = CoorX9
            coo1y = CoorY9
            Rate_Arr[index] = rate
            #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        if(B[index] == 'IPS1'):
            rate = int(round(Graph[9],3) * 1000)
            coo1x = CoorX10
            coo1y = CoorY10
            Rate_Arr[index] = rate
            #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        if(B[index] == 'IPS2'):
            rate = int(round(Graph[10],3) * 1000)
            coo1x = CoorX11
            coo1y = CoorY11
            Rate_Arr[index] = rate
            #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        
        

        #Thread(target = flicker, args = (coo1x, coo1y, rate)).start()
        
        index += 1
        print(rate)
        
        
    plt.bar(B,  Rate_Arr)
    plt.xlabel('Regions of Brain')
    plt.ylabel('Centrality Scores')
    plt.title('Betweenness Centrality Score Bar Chart')
    plt.show
    plt.savefig('Betweenness.png')
    image = cv2.imread('Betweenness.png')
    cv2.imshow('x', image)
    os.remove('Betweenness.png')
    plt.clf()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
            
def Method4():
    

    Ex = pd.read_excel(fileDialog2(), header = None)
    A = Ex.as_matrix()
    
    B = np.random.rand(len(A.T))
    C = ['V1', 'V2', 'V3', 'V3A','V3B', 'V4', 'V5', 'V7','LO1', 'IPS1', 'IPS2']
    row = 0
    column = 0
    z_score = 0
    #sum = 0
    print(A)
    while(column < len(A.T)):
        row = 0
        z_score = 0
        while(row < len(A)):
            avg = A[row].mean()
            std = A[row].std()
            #print("average is", avg)
            z_score += (A[row][column] - avg) / std
            row += 1
        z_score = z_score / len(A)
        B[column] = z_score
        print("Z-score is:", z_score)
        column += 1
    

    plt.bar(C,  B)
    plt.xlabel('Regions of Brain')
    plt.ylabel('Z-Score Values')
    plt.title('Within Module Degree Z-Score Bar Chart')
    plt.show
    plt.savefig('z-score.png')
    image = cv2.imread('z-score.png')
    cv2.imshow('x', image)
    os.remove('z-score.png')
    plt.clf()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
        
    
    
def Method5():
    g = nx.DiGraph()
    Ex = pd.read_excel(fileDialog2(), header = None)
    A = Ex.as_matrix()
    B = ['V1', 'V2', 'V3', 'V3A','V3B', 'V4', 'V5', 'V7','LO1', 'IPS1', 'IPS2']
    nodes = np.arange(1.5, (len(A) * 1.5) + 1.5, 1.5)
    Local_Peak = np.random.randint(1, size=(len(A) * 10, 2))

    row = 0
    column = 0
    
    print(A)

    while(column < len(A.T)):
        plt.plot(nodes, A[:,column], label = B[column])
        column += 1

    plt.legend() 

    plt.savefig('Modularity_Graph.png')
    image5 = cv2.imread('Modularity_Graph.png')
    cv2.imshow('Modularity Graph', image5)
    os.remove('Modularity_Graph.png')
    plt.clf()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


    column = 0
    local_row = 0
    #local_col = 0


    while(column < len(A.T)):
        row = 1
        #local_row = 0
        while (row < (len(A) - 1)):
            if(A[row][column] > A[row - 1][column] ):
                if(A[row][column] > A[row + 1][column]):
                    Local_Peak[local_row][0] = int(column)
                    Local_Peak[local_row][1] = int(row)
                    local_row += 1
            row += 1
        column += 1
    
    print("Array is : \n", Local_Peak)
    

    Adj_Mtx = np.random.randint(1, size=(len(B),len(B)))
    
    local_row = 0
    #local_col = 0

    #term = 0
    print(Adj_Mtx)

    matrow = 0
    matcol = 0
    value = 0
    scan = 0
    
    while(local_row < len(Local_Peak)):
        value = Local_Peak[local_row][1]
        scan = local_row + 1
        while(scan < len(Local_Peak)):
            if(Local_Peak[scan][1] != 0):
                if(Local_Peak[scan][1] == value):
                    matrow = Local_Peak[local_row][0] 
                    matcol = Local_Peak[scan][0]
                    Adj_Mtx[matrow][matcol] += 1
                    Adj_Mtx[matcol][matrow] += 1
            scan += 1
        local_row += 1
                    
    print(Adj_Mtx)
        
        
                
    matrow = 0
    matcol = 0

    while (matrow < len(Adj_Mtx)):
        matcol = 0
        while(matcol < len(Adj_Mtx)):
            if(Adj_Mtx[matrow][matcol] != 0):
                g.add_edges_from([(matrow, matcol)], weight = Adj_Mtx[matrow][matcol])
            matcol += 1
        matrow += 1

    matrow = 0
    matcol = 0

    Graph = nx.directed_modularity_matrix(g)


    print("Modularity is: \n", Graph)

    eigval,eigvec = LA.eig(Graph)
    print("Eigenvalues:", eigval)
    print("Eigenvectors:", eigvec)
    

    Frame = pd.DataFrame(eigvec, index= B, columns=B)
    val = 0

    CoorX1 = 290 + w #V1
    CoorY1 = h + 50
    pt1 = (CoorX1, CoorY1)


    CoorX2 = 260 + w #V2
    CoorY2 = h + 70
    pt2 = (CoorX2, CoorY2)


    CoorX3 = 230 + w #V3
    CoorY3 = h + 20
    pt3 = (CoorX3, CoorY3)

    CoorX4 = 190 + w #V3A
    CoorY4 = h - 10
    pt4 = (CoorX4, CoorY4)

    CoorX5 = 150 + w #V3B
    CoorY5 = h - 10
    pt5 = (CoorX5, CoorY5)

    CoorX6 = 190 + w #V4
    CoorY6 = h + 140
    pt6 = (CoorX6, CoorY6)


    CoorX7 = 160 + w #V5
    CoorY7 = h + 110
    pt7 = (CoorX7, CoorY7)


    CoorX8 = 170 + w #V7
    CoorY8 = h - 50
    pt8 = (CoorX8, CoorY8)

    CoorX9 = 195 + w #LO1
    CoorY9 = h + 100
    pt9 = (CoorX9, CoorY9)

    CoorX10 = 150 + w #IPS1
    CoorY10 = h - 120
    pt10 = (CoorX10, CoorY10)

    CoorX11 = 110 + w #IPS2
    CoorY11 = h - 160
    pt11 = (CoorX11, CoorY11)

    while(val < len(B)):

        if(B[val] == 'V1'):
            maxValuesObj = Frame[B[val]].max()
            maxValuesObj = (int((round(maxValuesObj, 1) * 10))) * 2
            #maxValuesObj2 = int((round(Frame.max(),1) * 10)*4)
            if(maxValuesObj <= 0):
                maxValuesObj = 8
            cv2.circle(image4, pt1, maxValuesObj, (0,0,255), -1)
        if(B[val] == 'V2'):
            maxValuesObj = Frame[B[val]].max()
            maxValuesObj = (int((round(maxValuesObj, 1) * 10)))* 2
            #maxValuesObj2 = int((round(Frame.max(),1) * 10)*4)
            if(maxValuesObj <= 0):
                maxValuesObj = 8
            cv2.circle(image4, pt2, maxValuesObj, (0,0,255), -1)
        if(B[val] == 'V3'):
            maxValuesObj = Frame[B[val]].max()
            maxValuesObj = (int((round(maxValuesObj, 1) * 10)))* 2
            #maxValuesObj2 = int((round(Frame.max(),1) * 10)*4)
            if(maxValuesObj <= 0):
                maxValuesObj = 8
            cv2.circle(image4, pt3, maxValuesObj, (0,0,255), -1)
        if(B[val] == 'V3A'):
            maxValuesObj = Frame[B[val]].max()
            maxValuesObj = (int((round(maxValuesObj, 1) * 10)))* 2
            #maxValuesObj2 = int((round(Frame.max(),1) * 10)*4)
            if(maxValuesObj <= 0):
                maxValuesObj = 8
            cv2.circle(image4, pt4, maxValuesObj, (0,0,255), -1)
        if(B[val] == 'V3B'):
            maxValuesObj = Frame[B[val]].max()
            maxValuesObj = (int((round(maxValuesObj, 1) * 10)))* 2
            if(maxValuesObj <= 0):
                maxValuesObj = 8
            #maxValuesObj2 = int((round(Frame.max(),1) * 10)*4)
            cv2.circle(image4, pt5, maxValuesObj, (0,0,255), -1)
        if(B[val] == 'V4'):
            maxValuesObj = Frame[B[val]].max()
            maxValuesObj = (int((round(maxValuesObj, 1) * 10)))* 2
            if(maxValuesObj <= 0):
                maxValuesObj = 8
            #maxValuesObj2 = int((round(Frame.max(),1) * 10)*4)
            cv2.circle(image4, pt6, maxValuesObj, (0,0,255), -1)
        if(B[val] == 'V5'):
            maxValuesObj = Frame[B[val]].max()
            maxValuesObj = (int((round(maxValuesObj, 1) * 10)))* 2
            if(maxValuesObj <= 0):
                maxValuesObj = 8
            #maxValuesObj2 = int((round(Frame.max(),1) * 10)*4)
            cv2.circle(image4, pt7, maxValuesObj, (0,0,255), -1)
        if(B[val] == 'V7'):
            maxValuesObj = Frame[B[val]].max()
            maxValuesObj = (int((round(maxValuesObj, 1) * 10)))* 2
            if(maxValuesObj <= 0):
                maxValuesObj = 8
            #maxValuesObj2 = int((round(Frame.max(),1) * 10)*4)
            cv2.circle(image4, pt8, maxValuesObj, (0,0,255), -1)
        if(B[val] == 'LO1'):
            maxValuesObj = Frame[B[val]].max()
            maxValuesObj = (int((round(maxValuesObj, 1) * 10)))* 2
            if(maxValuesObj <= 0):
                maxValuesObj = 8
            #maxValuesObj2 = int((round(Frame.max(),1) * 10)*4)
            cv2.circle(image4, pt9, maxValuesObj, (0,0,255), -1)
        if(B[val] == 'IPS1'):
            maxValuesObj = Frame[B[val]].max()
            maxValuesObj = (int((round(maxValuesObj, 1) * 10)))* 2
            if(maxValuesObj <= 0):
                maxValuesObj = 8
            #maxValuesObj2 = int((round(Frame.max(),1) * 10)*4)
            cv2.circle(image4, pt10, maxValuesObj, (0,0,255), -1)
        if(B[val] == 'IPS2'):
            maxValuesObj = Frame[B[val]].max()
            maxValuesObj = (int((round(maxValuesObj, 1) * 10)))* 2
            #maxValuesObj2 = int((round(Frame.max(),1) * 10)*4)
            if(maxValuesObj <= 0):
                maxValuesObj = 8
            cv2.circle(image4, pt11, maxValuesObj, (0,0,255), -1)
        val += 1
    
    maxValuesObj2 = (round(Frame.max(),1) * 10)*4
    print("aaa",maxValuesObj2)
    cv2.imshow('Brain', image4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    maxValuesObj2 = (round(Frame.max(),1) * 10)*2
    print(maxValuesObj2)


        
def Method6():
    g = nx.Graph()
    nodes = dict()
    #file = fileDialog()
    
    CoorX1 = 290 + w #V1
    CoorY1 = h + 50
    pt1 = (CoorX1, CoorY1)


    CoorX2 = 260 + w #V2
    CoorY2 = h + 70
    pt2 = (CoorX2, CoorY2)


    CoorX3 = 230 + w #V3
    CoorY3 = h + 20
    pt3 = (CoorX3, CoorY3)

    CoorX4 = 190 + w #V3A
    CoorY4 = h - 10
    pt4 = (CoorX4, CoorY4)

    CoorX5 = 150 + w #V3B
    CoorY5 = h - 10
    pt5 = (CoorX5, CoorY5)

    CoorX6 = 190 + w #V4
    CoorY6 = h + 140
    pt6 = (CoorX6, CoorY6)


    CoorX7 = 160 + w #V5
    CoorY7 = h + 110
    pt7 = (CoorX7, CoorY7)


    CoorX8 = 170 + w #V7
    CoorY8 = h - 50
    pt8 = (CoorX8, CoorY8)

    CoorX9 = 195 + w #LO1
    CoorY9 = h + 100
    pt9 = (CoorX9, CoorY9)

    CoorX10 = 150 + w #IPS1
    CoorY10 = h - 120
    pt10 = (CoorX10, CoorY10)

    CoorX11 = 110 + w #IPS2
    CoorY11 = h - 160
    pt11 = (CoorX11, CoorY11)
  


    CoorX12 = 36 + w
    CoorY12 = h - 41


    CoorX13 = -60 + w
    CoorY13 = h - 150


    CoorX14 = 120 + w
    CoorY14 = h + 80


    with open(fileDialog(), "r+") as f:
        
        lines = f.readlines()
        
        
        for l in lines:
            
            if(len(l.split(' '))==3):
                
                node1 = int(l.split(' ')[0].strip())
                node2 = int(l.split(' ')[1].strip())
                thickness = int(l.split(' ')[2].strip())
                g.add_edges_from([(node1,node2)])
                
                if (node1 not in nodes):
                    nodes.update({node1:thickness})
                
                elif (thickness > nodes[node1]):
                    nodes.update({node1:thickness})
                
                if (node2 not in nodes):
                    nodes.update({node2:thickness})
                
                elif (thickness > nodes[node2]):
                    nodes.update({node2:thickness})


        b=nx.betweenness_centrality(g)
        
        max = 0.0
        node = g.nodes(0)

        for x,y in b.items():
            
            if(max<y):
               
                max=y
                node=x


        max = 0
        
        for l in lines:
            
            if(len(l.split(' '))==3):
               
                node1 = int(l.split(' ')[0].strip())
                node2 = int(l.split(' ')[1].strip())
                thickness = int(l.split(' ')[2].strip())
                
                if(node1 == node or node2 == node ):
                    
                    if( max < thickness):
                        
                        max = thickness

        
        nodes.update({node:max})
        
  


        allNums = []

        nodeno=0
        coo1x = 300
        coo1y = 300
        coo2x = 0
        coo2y = 0
    

            
    
          
        for line in lines:
                
            allNums += line.strip().split(" ")

                
            Node1 = int(allNums[0])
            Node2 = int(allNums[1])
            #rate = int(allNums[2])
            if(Node1 == 1):
                coo1x = CoorX1
                coo1y = CoorY1
            if(Node1 == 2):
                coo1x = CoorX2
                coo1y = CoorY2
            if(Node1 == 3):
                coo1x = CoorX3
                coo1y = CoorY3
            if(Node1 == 4):
                coo1x = CoorX4
                coo1y = CoorY4
            if(Node1 == 5):
                coo1x = CoorX5
                coo1y = CoorY5
            if(Node1 == 6):
                coo1x = CoorX6
                coo1y = CoorY6
            if(Node1 == 7):
                coo1x = CoorX7
                coo1y = CoorY7
            if(Node1 == 8):
                coo1x = CoorX8
                coo1y = CoorY8
            if(Node1 == 9):
                coo1x = CoorX9
                coo1y = CoorY9
            if(Node1 == 10):
                coo1x = CoorX10
                coo1y = CoorY10
            if(Node1 == 11):
                coo1x = CoorX11
                coo1y = CoorX11
            if(Node1 == 12):
                coo1x = CoorX12
                coo1y = CoorY12
            if(Node1 == 13):
                coo1x = CoorX13
                coo1y = CoorY13
            if(Node1 == 14):
                coo1x = CoorX14
                coo1y = CoorY14
                

                
                
            if(Node2 == 1):
                coo2x = CoorX1
                coo2y = CoorY1
            if(Node2 == 2):
                coo2x = CoorX2
                coo2y = CoorY2
            if(Node2 == 3):
                coo2x = CoorX3
                coo2y = CoorY3
            if(Node2 == 4):
                coo2x = CoorX4
                coo2y = CoorY4
            if(Node2 == 5):
                coo2x = CoorX5
                coo2y = CoorY5
            if(Node2 == 6):
                coo2x = CoorX6
                coo2y = CoorY6
            if(Node2 == 7):
                coo2x = CoorX7
                coo2y = CoorY7
            if(Node2 == 8):
                coo2x = CoorX8
                coo2y = CoorY8
            if(Node2 == 9):
                coo2x = CoorX9
                coo2y = CoorY9
            if(Node2 == 10):
                coo2x = CoorX10
                coo2y = CoorY10
            if(Node2 == 11):
                coo2x = CoorX11
                coo2y = CoorY11
            if(Node2 == 12):
                coo2x = CoorX12
                coo2y = CoorY12
            if(Node2 == 13):
                coo2x = CoorX13
                coo2y = CoorY13
            if(Node2 == 14):
                coo2x = CoorX14
                coo2y = CoorY14
           
            
                  
            
            Thread(target = flicker, args = (coo1x, coo1y,nodes[nodeno])).start()
        
            #nodeno=nodeno+1
        
            Thread(target = flicker, args = (coo2x, coo2y,nodes[nodeno])).start()
                    
            nodeno=nodeno+1
            

            #flicker(coo1x, coo1y, nodes[nodeno])
                
            allNums=[]



def flicker(coo1x,coo1y,rate):
    
        x = True

        while(x):
            
            cv2.circle(image3, (int(coo1x),int(coo1y)), 5, (0,0,255), -1)    
            sem.acquire()
            cv2.imshow("Betweenness Centrality", original) 
            
   
            sem.release()
            cv2.waitKey(rate*8) 
            
            for i in range (coo1x-10, coo1x+10):
                for j in range(coo1y-10, coo1y+10):
                    image3[i,j] = original[i,j]
            sem.acquire()
            cv2.imshow("Betweenness Centrality",image3)
            
            
            sem.release()
            cv2.waitKey(rate*4)
            
            

            
def Method7():
    allNums2 = []
    #B = ['V1', 'V2', 'V3', 'V3A','V3B', 'V4', 'V5', 'V7','LO1', 'IPS1', 'IPS2']

    CoorX1 = 290 + w #V1
    CoorY1 = h + 50
    pt1 = (CoorX1, CoorY1)


    CoorX2 = 260 + w #V2
    CoorY2 = h + 70
    pt2 = (CoorX2, CoorY2)


    CoorX3 = 230 + w #V3
    CoorY3 = h + 20
    pt3 = (CoorX3, CoorY3)
    

    CoorX4 = 190 + w #V3A
    CoorY4 = h - 10
    pt4 = (CoorX4, CoorY4)
    

    CoorX5 = 150 + w #V3B
    CoorY5 = h - 10
    pt5 = (CoorX5, CoorY5)
    

    CoorX6 = 190 + w #V4
    CoorY6 = h + 140
    pt6 = (CoorX6, CoorY6)



    CoorX7 = 160 + w #V5
    CoorY7 = h + 110
    pt7 = (CoorX7, CoorY7)
    
    
    CoorX8 = 170 + w #V7
    CoorY8 = h - 50
    pt8 = (CoorX8, CoorY8)
    

    CoorX9 = 195 + w #LO1
    CoorY9 = h + 100
    pt9 = (CoorX9, CoorY9)
    

    CoorX10 = 150 + w #IPS1
    CoorY10 = h - 120
    pt10 = (CoorX10, CoorY10)
    

    CoorX11 = 110 + w #IPS2
    CoorY11 = h - 160
    pt11 = (CoorX11, CoorY11)

    
    with open(fileDialog(), "r+") as f:
        data = f.readlines()
        for line in data:

            allNums2 += line.strip().split(" ")

            Node = int(allNums2[0])
            Betweenness = int(allNums2[1])

            if(Node == 1):
                CoorX = CoorX1
                CoorY = CoorY1
          

            if(Node == 2):
                CoorX = CoorX2
                CoorY = CoorY2
        

            if(Node == 3):
                CoorX = CoorX3
                CoorY = CoorY3               


            if(Node == 4):
                CoorX = CoorX4
                CoorY = CoorY4                


            if(Node == 5):
                CoorX = CoorX5
                CoorY = CoorY5                


            if(Node == 6):
                CoorX = CoorX6
                CoorY = CoorY6               


            if(Node == 7):
                CoorX = CoorX7
                CoorY = CoorY7                


            if(Node == 8):
                CoorX = CoorX8
                CoorY = CoorY8                


            if(Node == 9):
                CoorX = CoorX9
                CoorY = CoorY9                


            if(Node == 10):
                CoorX = CoorX10
                CoorY = CoorY10                


            if(Node == 11):
                CoorX = CoorX11
                CoorY = CoorY11               
               
            Thread(target = flicker2, args = (CoorX, CoorY,Betweenness)).start()
            print(Betweenness)
            allNums2=[]



def flicker2(coo1x,coo1y,rate):
    
        x = True

        while(x):
            
            cv2.circle(image3, (int(coo1x),int(coo1y)), 5, (0,0,255), -1)    
            sem.acquire()
            cv2.imshow("Betweenness Centrality", image3) 
            
   
            sem.release()
            cv2.waitKey(rate*8) 
            
            for i in range (coo1x, coo1x):
                for j in range(coo1y, coo1y):
                    image3[i,j] = original[i,j]
            sem.acquire()
            cv2.imshow("Betweenness Centrality",original)
            
            
            sem.release()
            cv2.waitKey(rate*4)
            


def _quit():
    os._exit(1)




B = tkinter.Button(top, text ="Microscale Directed Graph", fg = "blue", command = Method1, height = 5, width = 40)
B.grid(column=0, row=0)


D = tkinter.Button(top, text ="Macroscale Directed Graph", fg = "blue", command = Method2, height = 5, width = 40)
D.grid(column=1, row=0)


F = tkinter.Button(top, text ="Centrality Score Bar Chart", fg = "blue", command = Method3, height = 5, width = 40)
F.grid(column=2, row=0)


H = tkinter.Button(top, text ="Z-Score Bar Chart", fg = "blue", command = Method4, height = 5, width = 40)
H.grid(column=0, row=1)

H = tkinter.Button(top, text ="Modularity Graph", fg = "blue", command = Method5, height = 5, width = 40)
H.grid(column=1, row=1)

J = tkinter.Button(top, text="Centrality Score Video", fg="blue", command=Method6, height = 5, width = 40)
J.grid(column=2, row=1)

J = tkinter.Button(top, text="QUIT", fg="red", command=_quit, height = 5, width = 40)
J.grid(column=1, row=2)


#J = tkinter.Button(top, text="aaa", fg="blue", command=Method7, height = 5, width = 40)
#J.grid(column=0, row=2)



top.mainloop()





