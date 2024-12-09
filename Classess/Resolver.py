import numpy as np
import cv2 as cv


class Resolver:

    #def __init__():
        


    """
    @fn:        open_board    
    @params:    opens the board image
    @brief:     opens and cropps the image of the board
    @author:    I.Sz.
    @version:   1.0
    @date:      09-2024
    """    
    def open_board(self,dir):
        img = cv.imread(dir)
        
        #Crop the image to take out blank borders and help with detection
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
        cv.imshow("Queens Inverted",gray)

        #Take the largest countour and use it as a mask for a first cropp
        countour_g, hierarchies_g = cv.findContours(gray,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)


        values = []
        for cnts in range(1,len(countour_g)):
            cnt = countour_g[cnts]
            values.append(cv.contourArea(cnt))


        #get de maximun area value
        max_area = max(values)
        #get the position where the max area is (+1 because index starts from 0)
        max_area_contour = values.index(max_area)+1

        print(f"max_area = {max_area}")
        print(f"max_area_contour = {max_area_contour}")

        #draw the mask in an empty image
        mask = np.zeros(gray.shape[:2],dtype='uint8')
        x,y,w,h = cv.boundingRect(countour_g[max_area_contour])
        x_center = x+(w/2)
        y_center = y+(h/2)
        mask = cv.rectangle(mask, (x,y), (x+w,y+h), (255,255,255) , -1)
        cv.imshow('mask crop',mask)

        cropped_image = cv.bitwise_and(gray,mask)
        cv.imshow('cropped_image',cropped_image)
        cv.waitKey(0)


        coords = cv.findNonZero(mask) # Find all non-zero points (text)
        x, y, w, h = cv.boundingRect(coords) # Find minimum spanning bounding box
        img = img[y:y+h, x:x+w] # Croppen image
        cv.imshow("Queens Cropped",img)


        #Lo que me interesa en primer lugar, es poder definir cuántos cuadrados tengo en la imagen. Para poder armar la grilla
        #La imagen gris también puede utilizarse para hacer los grupos (de una manera más simple, sin utilizar realmente color)

        #1. Paso a blanco y negro
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY )
        #cv.imshow("Queens Gray",img_gray)
        #cv.waitKey(0)



        #2. Aplico Threshold para lograr imagen 0-255: todo lo que es negro, queda negro (0), todo lo que no sea negro, queda en 255
        ret, thresh = cv.threshold(img_gray,125,255,cv.THRESH_BINARY)
        cv.imshow('Queens thresh', thresh)
        #cv.waitKey(0)


    """
    @fn:        get_contours    
    @params:    img
    @brief:     gets the contour of the image thats being analyzed
    @author:    I.Sz.
    @version:   1.0
    @date:      09-2024
    """
    def get_contours(self,img):
        #ahora que tenemos sólo una grilla de cuadrados, debemos determinar cuántos cuadrados tenemos en la imagen
        # deteccion de contornos?
        countour_t, hierarchies_t = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

        #podemos visualziar los contornos, dibujando sobre la imagen
        #thresh = cv.cvtColor(thresh,cv.COLOR_GRAY2BGR)
        #cv.drawContours(thresh,countour_t,-1,(0,255,0),1)
        cv.imshow('Contours Tresh',thresh)


        #Cannny: hago un procesamiento con Canny en lugar de threshold, ya que por el tipo de imagen, da un mejor resultado

        #Canny(imagen a analizar, minVal, maxVal, tamaño del kernel para sacar gradiente (3 por default))
        canny = cv.Canny(img,125,175)

        ret, canny = cv.threshold(canny,10,255,cv.THRESH_BINARY)
        cv.imshow('Img canny', canny)

        #erosion y dilatación para eliminar ruido
        # definimos el kernel que puede ser de tamaño 3, 5  o 7
        kernel_dilat = np.ones((5, 5), np.uint8)
        #realizamos las operaciones
        canny_dilation = cv.dilate(canny, kernel_dilat, iterations=2)
        cv.imshow('canny_dilation', canny_dilation)

        #busco contornos en la imagen canny
        countour_c, hierarchies_canny = cv.findContours(canny_dilation,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        #Dibujo los contonrs obtenidos con Canny
        aux = np.zeros(img.shape[:2],dtype="uint8")
        cv.drawContours(aux,countour_c,-1,(255,255,255),1)
        cv.imshow('Canny countours', aux)

        filled_countours_c = np.zeros(img.shape[:2],dtype='uint8')
        for cnts in range(1,len(countour_c)):
            cnt = countour_c[cnts]
            x,y,w,h = cv.boundingRect(cnt)
            x_center = x+(w/2)
            y_center = y+(h/2)

            filled_countours_c = cv.rectangle(filled_countours_c, (x,y), (x+w,y+h), (255,255,255) , -1)



        cv.imshow("filled_countours_c",filled_countours_c)

        print(f'{len(countour_t)} threshold countour(s) fount!')
        print(f'{len(countour_c)} canny countour(s) fount!')


        total_image = cv.bitwise_or(thresh,filled_countours_c)
        cv.imshow("total_image",total_image)

        countour_tc, hierarchies_canny = cv.findContours(total_image,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        aux = np.zeros(img.shape[:2],dtype="uint8")
        cv.drawContours(aux,countour_tc,-1,(255,255,255),1)
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



    """
    @fn:        get_colours    
    @params:    number of squares present on the map, contours to use, grid
    @brief:     gets the color of each square of the map, and translates it into an id to fill the grid
    @author:    I.Sz.
    @version:   1.0
    @date:      09-2024
    """
    def get_colours(self,sqr_cant,contour,plus):
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

            #only proceed if the row|col of the grid is 0 (i.e.: has no color associated)
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



