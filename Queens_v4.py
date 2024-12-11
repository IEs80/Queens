import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# ***** Function definitions *****#
"""
@fn:        plot_image    
@params:    image and title
@brief:     plots an image with matplotlib
@author:    I.Sz.
@version:   1.0
@date:      09-2024
"""
def plot_image(img,title):

    plt.title(title) 
    plt.xlabel("x axis") 
    plt.ylabel("y axis") 
    plt.imshow(img)
    plt.show()

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
@params:    number of squares present on the map, contours to use, grid
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
                #cv.imshow("filled", filled)
                #filled = cv.cvtColor(filled,cv.COLOR_GRAY2BGR)
                
                masked = cv.bitwise_and(img,img,mask=filled) #
                
                #cv.circle(masked,coord,10,255,-1)
                #cv.imshow("masked", masked)


                print(f"masked shape {masked.shape}")
                print(f"x_center = {int(x_center)}")
                print(f"y_center = {int(y_center)}")

                pix_values = masked[int(y_center),int(x_center)] 
                

                #sumo para obtener un valor único (identificador de grupo)
                group_id = float(pix_values[0]) + float(pix_values[1]) +float(pix_values[2]) #los valores son uint8_t (0-255), entonces si quiero guardar la suma podría haber overflow. Por eso, casteo a una variable de mayor tamaño

                print(f"B value sub-{row_index} {col_index} is: {pix_values[0]}")
                print(f"G value sub-{row_index} {col_index} is: {pix_values[1]}")
                print(f"R value sub-{row_index} {col_index} is: {pix_values[2]}")

                #guardo el valor correspondiente en la grilla, sólo si estaba vacía
                grid[0][row_index][col_index] = round(group_id) #
                #print(grid[0,:,:])
                #cv.imshow(f"Countour {col}", filled)
                #cv.imshow(f"Square b {col_index}", masked[:,:,0])
                #cv.imshow(f"Square g {col_index}", masked[:,:,1])
                #cv.imshow(f"Square r {col_index}", masked[:,:,2])
                #cv.waitKey(0)

                #dibujo un tablero para verificar los resultados
                board = cv.rectangle(board, (x,y), (x+w,y+h), (int(pix_values[0]),int(pix_values[1]),int(pix_values[2])) , -1)
        else:
                filled = np.zeros(img.shape[:2],dtype='uint8')
                filled = cv.rectangle(filled, (x,y), (x+w,y+h), (255,255,255) , -1)
                #cv.imshow("filled", filled)
                #filled = cv.cvtColor(filled,cv.COLOR_GRAY2BGR)
                
                masked = cv.bitwise_and(img,img,mask=filled) #
                
                #cv.circle(masked,coord,10,255,-1)
                #cv.imshow("masked", masked)


                print(f"masked shape {masked.shape}")
                print(f"x_center = {int(x_center)}")
                print(f"y_center = {int(y_center)}")

                pix_values = masked[int(y_center),int(x_center)] 
                

                #sumo para obtener un valor único (identificador de grupo)
                group_id = float(pix_values[0]) + float(pix_values[1]) +float(pix_values[2]) #los valores son uint8_t (0-255), entonces si quiero guardar la suma podría haber overflow. Por eso, casteo a una variable de mayor tamaño

                print(f"B value sub-{row_index} {col_index} is: {pix_values[0]}")
                print(f"G value sub-{row_index} {col_index} is: {pix_values[1]}")
                print(f"R value sub-{row_index} {col_index} is: {pix_values[2]}")

                #guardo el valor correspondiente en la grilla, sólo si estaba vacía
                grid[0][row_index][col_index] = round(group_id) #
                #print(grid[0,:,:])
                #cv.imshow(f"Countour {col}", filled)
                #cv.imshow(f"Square b {col_index}", masked[:,:,0])
                #cv.imshow(f"Square g {col_index}", masked[:,:,1])
                #cv.imshow(f"Square r {col_index}", masked[:,:,2])
                #cv.waitKey(0)

                #dibujo un tablero para verificar los resultados
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
    try:                    #se intenta ejecutar lo que está dentro de la clausula "try"
        usr_in = str(input())
        if not(usr_in.isalpha()):
            raise TypeError
    #si lo que está dentro de "try" produce error, entonces se ejecuta esto            
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


#Lo que me interesa en primer lugar, es poder definir cuántos cuadrados tengo en la imagen. Para poder armar la grilla
#La imagen gris también puede utilizarse para hacer los grupos (de una manera más simple, sin utilizar realmente color)

#1. Paso a blanco y negro
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY )
#cv.imshow("Queens Gray",img_gray)
#cv.waitKey(0)



#2. Aplico Threshold para lograr imagen 0-255: todo lo que es negro, queda negro (0), todo lo que no sea negro, queda en 255
ret, thresh = cv.threshold(img_gray,125,255,cv.THRESH_BINARY)

if debug_cv == 1:
    cv.imshow('Queens thresh', thresh)
#cv.waitKey(0)


#canny = cv.Canny(img,125,175)
#thresh = thresh+canny
if debug_cv == 1:
    cv.imshow('Queens thresh+Canny', thresh)

#ahora que tenemos sólo una grilla de cuadrados, debemos determinar cuántos cuadrados tenemos en la imagen
# deteccion de contornos?
countour_t, hierarchies_t = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

#podemos visualziar los contornos, dibujando sobre la imagen
#thresh = cv.cvtColor(thresh,cv.COLOR_GRAY2BGR)
#cv.drawContours(thresh,countour_t,-1,(0,255,0),1)
if debug_cv == 1:
    cv.imshow('Contours Tresh',thresh)


#Cannny: hago un procesamiento con Canny en lugar de threshold, ya que por el tipo de imagen, da un mejor resultado

#Canny(imagen a analizar, minVal, maxVal, tamaño del kernel para sacar gradiente (3 por default))
canny = cv.Canny(img,125,175)

ret, canny = cv.threshold(canny,10,255,cv.THRESH_BINARY)
if debug_cv == 1:
    cv.imshow('Img canny', canny)

#erosion y dilatación para eliminar ruido
# definimos el kernel que puede ser de tamaño 3, 5  o 7
kernel_dilat = np.ones((5, 5), np.uint8)
#realizamos las operaciones
canny_dilation = cv.dilate(canny, kernel_dilat, iterations=2)
if debug_cv == 1:
    cv.imshow('canny_dilation', canny_dilation)

#busco contornos en la imagen canny
countour_c, hierarchies_canny = cv.findContours(canny_dilation,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

#Dibujo los contonrs obtenidos con Canny
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


#defino el tamaño de columnas y filas
cols_rows = int(np.sqrt(sqr_cant))
sqr_cant = cols_rows*cols_rows
print(f'{sqr_cant} squares present in the map')
#defino un np.array de cols_rows x cols_rows
grid = np.zeros(shape=[2,cols_rows,cols_rows]) #dos dimensiones: una para utilizar como guía de espacio "libre-ocupado" y otra para llevar los colores
print("\n\tTha map: \n\n")
print(grid)

x = range(cols_rows,0)
print(x)

#print(countour_t)
print(img.shape[:2])
#cantidad columnas
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
print(f"\nGroup ids: {valid_ids}")

#A partir de acá, es resolver el problema
#Reglas de Queens:
# 1) cada fila, columna y región coloreada debe contener exactamente un símbolo de corona (reina).
# 2) Los símbolos de corona no se pueden colocar en celdas adyacentes, ni siquiera en diagonal.
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

    use_decrement = 0
    #column = np.random.choice(valid_cols)
    #row = np.random.choice(valid_rows)

    print(probabilities[1])

    #columna utilizada por iteración
    iteration_col = []
    
    #columans que no puedo utilziar más en una iteración
    invalid_cols = [] 

    #mientras aún tenga colores diponibles para completar
    while valid_ids.size>0:
        #comienzo por la primera fila y voy bajando
        for i in range(cols_rows):

            #si seleccioné una columna que sé que es inválida, vuelvo a seleccionar
            while True and len(invalid_cols) < cols_rows:
                column = np.random.choice(valid_cols, size=1, p=probabilities[i]) #selecciono la columna en base a las que tengo disponibles para utilizar y la probabilidad
                if invalid_cols.count(column)==0: #reviso si la columna seleccionada está en la lista de las columnas ya utilizadas
                    break
            #column = np.random.choice(valid_cols,size=1,p=probabilities[i]) #esta función tiene un parámetro que es un array de prbabilidades. Lo que habría que hacer es, para cada i, en sucesivas pasadas, ir modificando estas probabilidades


            #print(grid[1,i,0:cols_rows-1].sum()) #así reviso filas
            #print(grid[1,0:cols_rows,column].sum()) #así reviso columnas
            
            #Verificaciones
            # fila y columna vacía
            # todas las columnas de la fila, vacías
            # todas las filas de la columna vacías
            # el color de la fila columna está sin usar
            # TODO: adyacentes vacías! 
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
                #print(valid_ids)
                #print(f"columns = {column}")
            if valid_cols.size<=0:
                break
        if valid_ids.size<=0 or len(invalid_cols) == cols_rows:
            print(" * Solución encontrada *")
            break
        
        #si salí del for, pero aún me quedan ids desocupados, tengo que hacer otro intento
        if valid_ids.size>0:
            #print("probando otra solución")
            #print(f"iteration_col = {iteration_col}")
            #disminuyo las probabilidades de las columnas seleccionadas para cada fila
            
            if use_decrement == 1:
                for j in range(len(iteration_col)):
                    #print(f"iteration_col[j] = {iteration_col[j]}")
                    y = iteration_col[j][0] #row
                    x = iteration_col[j][1] #col 
                    #print(f"x = {x}")
                    #print(f"y = {y}")
                    probabilities[y,x] = probabilities[y,x]-decrement
                    if probabilities[y,x]<0:
                        probabilities[y,x] = 0.1 #no uso probabilidad 0, para que siempre alguna chance tengan de salir. 
                    probabilities[y] /= np.sum(probabilities[y]) 

                    print(f"probabilites = {probabilities}")
                    #probabilities[iteration_col[j]] /= np.sum(probabilities[iteration_col[j]])

            valid_ids = np.unique(grid[0,:,:])
            invalid_cols = [] #reinicio invalid cols
            #valid_rows = np.arange(0,cols_rows-1)
            #valid_cols = np.arange(0,cols_rows-1) 
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
