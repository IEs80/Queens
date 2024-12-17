import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# ***** Function definitions *****#
"""
@fn:        empty_diagonals    
@params:    grid "coodinates"
@brief:     checks if the diagonals of any given squeare in the grid are empty
@author:    I.Sz.
@version:   1.0
@date:      11-2024
"""
def empty_diagonals(y,x):

    if y-1>=0 and x-1>=0 and y-1<cols_rows and x-1<cols_rows:
        if grid[1,y-1,x-1]==1:
            return False

    if y+1>=0 and x+1>=0 and y+1<cols_rows and x+1<cols_rows:
        if grid[1,y+1,x+1]==1:
            return False

    if y-1>=0 and x+1>=0 and y-1<cols_rows and x+1<cols_rows:
        if grid[1,y-1,x+1]==1:
            return False
        
    if y+1>=0 and x-1>=0 and y+1<cols_rows and x-1<cols_rows:
        if grid[1,y+1,x-1]==1:
            return False
            
    return True


"""
@fn:        get_board    
@params:    number of squares present on the map, contours to use, a grid, a parameter to only perform the operations over empty squares
@brief:     gets the color of each square of the map, and translates it into an id to fill the grid
@author:    I.Sz.
@version:   1.0
@date:      09-2024
"""
def get_board(sqr_cant,contour,plus,selective):
    global grid
    global board

    for cnts in range(sqr_cant+plus):

        print(f"Canny cnts = {cnts}")

        cnt = contour[cnts]
        x,y,w,h = cv.boundingRect(cnt)
        x_center = x+(w/2)
        y_center = y+(h/2)

        col_index = int(((x_center*cols_rows)/col_nbr))
        row_index = int(((y_center*cols_rows)/row_nbr))

        
        if selective == 1:
            #only proceed if the row|col of the grid is 0 (has no color associated)
            if grid[0][row_index][col_index] == 0:

                filled = np.zeros(img.shape[:2],dtype='uint8')
                filled = cv.rectangle(filled, (x,y), (x+w,y+h), (255,255,255) , -1)

                
                masked = cv.bitwise_and(img,img,mask=filled) #
                
                print(f"masked shape {masked.shape}")
                print(f"x_center = {int(x_center)}")
                print(f"y_center = {int(y_center)}")

                pix_values = masked[int(y_center),int(x_center)] 
                
                #sum all to get an unique value (group id)
                group_id = float(pix_values[0]) + float(pix_values[1]) +float(pix_values[2]) #los valores son uint8_t (0-255), entonces si quiero guardar la suma podría haber overflow. Por eso, casteo a una variable de mayor tamaño

                print(f"B value sub-{row_index} {col_index} is: {pix_values[0]}")
                print(f"G value sub-{row_index} {col_index} is: {pix_values[1]}")
                print(f"R value sub-{row_index} {col_index} is: {pix_values[2]}")

                
                #store the corresponding value only if the possition was empty
                grid[0][row_index][col_index] = round(group_id) #
      
                #draw a board to verify the results
                board = cv.rectangle(board, (x,y), (x+w,y+h), (int(pix_values[0]),int(pix_values[1]),int(pix_values[2])) , -1)
        else:
                filled = np.zeros(img.shape[:2],dtype='uint8')
                filled = cv.rectangle(filled, (x,y), (x+w,y+h), (255,255,255) , -1)
                
                masked = cv.bitwise_and(img,img,mask=filled) #
                
                print(f"masked shape {masked.shape}")
                print(f"x_center = {int(x_center)}")
                print(f"y_center = {int(y_center)}")

                pix_values = masked[int(y_center),int(x_center)] 
                

                #sum all to get an unique value (group id)
                group_id = float(pix_values[0]) + float(pix_values[1]) +float(pix_values[2]) #he pix_values are uint8_t (0-255), si if we sum all in a uint8_t there colud be overflow, so we cast them as floats

                print(f"B value sub-{row_index} {col_index} is: {pix_values[0]}")
                print(f"G value sub-{row_index} {col_index} is: {pix_values[1]}")
                print(f"R value sub-{row_index} {col_index} is: {pix_values[2]}")

                #store the corresponding value
                grid[0][row_index][col_index] = round(group_id) #

                #draw a board to verify the results
                board = cv.rectangle(board, (x,y), (x+w,y+h), (int(pix_values[0]),int(pix_values[1]),int(pix_values[2])) , -1)


"""
@fn:        user_input    
@params:    -
@brief:     Validates user input
@author:    I.Sz.
@version:   1.0
@date:      09-2024
"""
def user_input_char():
    try:                    
        usr_in = str(input())
        if not(usr_in.isalpha()):
            raise TypeError
          
    except TypeError:
        print("Please, enter a valid input, numbers are not allowed")
        usr_in = user_input_char()

    print(f"usr_in = {usr_in}")
    if usr_in == 'S' or usr_in == 'N':
        return usr_in
    else:
        print("Please, enter a valid input")
        usr_in = user_input_char()


# ***** Main Script Starts Here *****#

#variable that enables de output of control prints of intermediate images
debug_cv = 0

img = cv.imread(r'.\QueensBoard.jpg')
#img = cv.imread(r'.\QueensBoard_3.jpg')
#img = cv.imread(r'.\QueensBoard_4.jpg')
#img = cv.imread(r'.\QueensBoard_5.jpg')

#Crop the image to take out blank borders and help with detection
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white

if debug_cv == 1:
    cv.imshow("Queens Inverted",gray)



coords = cv.findNonZero(gray) # Find all non-zero points (text)
x, y, w, h = cv.boundingRect(coords) # Find minimum spanning bounding box
img = img[y:y+h, x:x+w] # Croppen image


cv.imshow("Original board Cropped",img)


#First, we need to define how many squares we have on the image, to be able to define the grid
#A grayscale image can be used to determine the groups (without really using color, in a simpler way)
#1. convert to grayscale
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY )


#2. Apply a Threshold to get an image that consists only in pixels with values 0-255: what is black, stays black (0), everything else turns white (255)
ret, thresh = cv.threshold(img_gray,125,255,cv.THRESH_BINARY)

if debug_cv == 1:
    cv.imshow('Queens thresh', thresh)



#canny = cv.Canny(img,125,175)
#thresh = thresh+canny
if debug_cv == 1:
    cv.imshow('Queens thresh+Canny', thresh)

#Now we have only a square grid, we must determine the number of squares
#contour detection
countour_t, hierarchies_t = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

#We can visualize contours drawing over the image
if debug_cv == 1:
    cv.imshow('Contours Tresh',thresh)


#Cannny: we also apply a Canny proces, this can give a better result
#Canny(imagen a analizar, minVal, maxVal, tamaño del kernel para sacar gradiente (default = 3))
canny = cv.Canny(img,125,175)

ret, canny = cv.threshold(canny,10,255,cv.THRESH_BINARY)
if debug_cv == 1:
    cv.imshow('Img canny', canny)

#erode y dilate to get rid of noise
#kernel can be 3, 5  o 7
kernel_dilat = np.ones((5, 5), np.uint8)
#we do the opeartions
canny_dilation = cv.dilate(canny, kernel_dilat, iterations=2)
if debug_cv == 1:
    cv.imshow('canny_dilation', canny_dilation)

#search contours on Canny image
countour_c, hierarchies_canny = cv.findContours(canny_dilation,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

#Draw cotours obtained with Canny
aux = np.zeros(img.shape[:2],dtype="uint8")
cv.drawContours(aux,countour_c,-1,(255,255,255),1)
if debug_cv == 1:
    cv.imshow('Canny countours', aux)

filled_countours_c = np.zeros(img.shape[:2],dtype='uint8')
for cnts in range(1,len(countour_c)):
    cnt = countour_c[cnts]
    x,y,w,h = cv.boundingRect(cnt)
    x_center = x+(w/2)
    y_center = y+(h/2)

    filled_countours_c = cv.rectangle(filled_countours_c, (x,y), (x+w,y+h), (255,255,255) , -1)


if debug_cv == 1:
    cv.imshow("filled_countours_c",filled_countours_c)

print(f'{len(countour_t)} threshold countour(s) fount!')
print(f'{len(countour_c)} canny countour(s) fount!')


total_image = cv.bitwise_or(thresh,filled_countours_c)
if debug_cv == 1:
    cv.imshow("total_image",total_image)

countour_tc, hierarchies_canny = cv.findContours(total_image,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
aux = np.zeros(img.shape[:2],dtype="uint8")
cv.drawContours(aux,countour_tc,-1,(255,255,255),1)
if debug_cv == 1:
    cv.imshow("total_image countours",aux)

plus_tresh = 2
plus_canny = 1
plus_ct = 3


#cv.waitKey(0)
#el detector de contornos, encuentra dos contornos de más: el de la imagen en general y el de el cuadrado grande => le resto dos a la cantidad para obtener el número de cuadrados presentes
sqr_cant = int(len(countour_t)-plus_tresh)
print(f'{sqr_cant} squares present in the map with thresh')
sqr_cant = int(len(countour_c)-plus_canny)
print(f'{sqr_cant} squares present in the map with canny')
sqr_cant = int(len(countour_tc)-plus_ct)
print(f'{sqr_cant} squares present in the map with thresh+canny')


#define the size of the grid (columns and rows)
cols_rows = int(np.sqrt(sqr_cant))
sqr_cant = cols_rows*cols_rows
print(f'{sqr_cant} squares present in the map')
#define a np.array of cols_rows x cols_rows
grid = np.zeros(shape=[2,cols_rows,cols_rows]) #two dimensions: one that represents "free-occupied" and other for the color of the square
print("\n\tTha map: \n\n")
print(grid)

x = range(cols_rows,0)
print(x)

#print(countour_t)
print(img.shape[:2])
#number of columns and rows
row_nbr = img.shape[0]
col_nbr = img.shape[1]

board = np.zeros(img.shape[:2],dtype="uint8")
board = cv.cvtColor(board,cv.COLOR_GRAY2BGR)

#call function to recreate the provided board
get_board(sqr_cant=sqr_cant,contour=countour_tc,plus=plus_ct,selective=0)

#if one or more id is equal to 0, then a portion of the map was left black. So, we use the canny contours to fill that part of the map
if 0 in np.unique(grid[0,:,:]):
    print("Some squares were left in blank, parsing with Canny only...")
    get_board(sqr_cant=sqr_cant,contour=countour_c,plus=plus_canny,selective=1)


print(grid[0,:,:])
valid_ids = np.unique(grid[0,:,:]) #obtengo todos los distintos ids (colores) que tengo en mi mapa

if 0 in valid_ids:
    print(f"\*** The board couldn't be parsed. You may try again cropping your image better. Execution aborted.")
else:

    print(f"\nGroup ids: {valid_ids}")

    #From here, we must resolve the game
    #Queens rules: 
    # 1) each row, column and coloured region must contain exactly one crown (queen).
    # 2) the crowns, can't be adjacent to each other
    #cv.imshow(f"Real board", img)
    cv.imshow(f"Recreated board", board)


    #*********** Resolver *************
    print('Start Resolver? Enter "S" to start or "N" to exit')
    data = user_input_char()

    if data == "S":


        #lista de columas y filas válidas
        valid_rows = np.arange(0,cols_rows) #filas disponibles para usar
        valid_cols = np.arange(0,cols_rows) #columnas disponibles para usar

        probabilities = (1/cols_rows)*np.ones((cols_rows,cols_rows)) #distribucion unfirme

        decrement = (1/cols_rows)*0.1 #tasa de decremento de las probabilidades

        #Enables the use of variable probabilities. Leave this value at 0
        use_decrement = 0

        print(probabilities[1])

        #columns used in the iteration
        iteration_col = []

        #columns that con no longer be used in this iteration
        invalid_cols = [] 

        #While i have available color to iterate
        while valid_ids.size>0:
            #star with the first row, and go down
            for i in range(cols_rows):

                #if an invalid column for this iteration (i.e.: an already used column) was selected, then choose again
                while len(invalid_cols) < cols_rows:
                    column = np.random.choice(valid_cols, size=1, p=probabilities[i]) #choose an available column from the list and based on the probabilities 
                    if invalid_cols.count(column)==0: #check if the selected column is on the list of already used ones
                        break
               
                #Validations
                # row and column must be empty
                # all the rows from that column must be empty
                # all the columns from that row must be empty
                # the color from the picked columns-row is unused
                # thera are no occupied spaces around the selected column-row
                if grid[1,i,column]==0 and grid[1,i,0:cols_rows-1].sum()==0 and grid[1,0:cols_rows,column].sum()==0 and grid[0,i,column] in valid_ids and (i in valid_rows) and empty_diagonals(i,column):
                    grid[1,i,column]=1
                    #valid_cols = np.delete(valid_cols,np.argwhere(valid_cols==column))
                    #valid_rows = np.delete(valid_rows,np.argwhere(valid_rows==i))
                    valid_ids = np.delete(valid_ids,np.argwhere(valid_ids==grid[0,i,column]))
                    iteration_col.append([i,column[0]])
                    invalid_cols.append(column[0])
                    print(f"invalid_col = {invalid_cols}")
                    print(f"len(invalid_cols) = {len(invalid_cols)}")
                    print(f"cols_rows = {cols_rows}")
                if valid_cols.size<=0:
                    break
            if valid_ids.size<=0 or len(invalid_cols) == cols_rows:
                print(" * Solución encontrada *")
                break
            
            
            #if we exited the for loop, but still got availables ids, we must do another try
            if valid_ids.size>0:
                
                if use_decrement == 1:
                    for j in range(len(iteration_col)):
                        y = iteration_col[j][0] #row
                        x = iteration_col[j][1] #col 
                        probabilities[y,x] = probabilities[y,x]-decrement
                        if probabilities[y,x]<0:
                            probabilities[y,x] = 0.1 #no uso probabilidad 0, para que siempre alguna chance tengan de salir. 
                        probabilities[y] /= np.sum(probabilities[y]) 

                        print(f"probabilites = {probabilities}")

                valid_ids = np.unique(grid[0,:,:])
                invalid_cols = [] #reinicio invalid cols
                grid[1,:,:] = 0 #reinicio la grilla
            
            
        print(grid[1,:,:])
        print("\n\n")
        print(grid[0,:,:])
        #print(iteration_col)

        cv.waitKey(0)

    else:
        print("*** Press any key to close the program ***")
        cv.imshow(f"Recreated board", board)
        cv.waitKey(0)
