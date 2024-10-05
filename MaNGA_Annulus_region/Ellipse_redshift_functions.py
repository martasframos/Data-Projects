import math
import numpy as np
import pandas as pd


def point_inside_rotated_ellipse(x0, y0, cx, cy, a, b, angle_degrees):
    
    """
    Is the point inside of an ellipse? 
    ..................................

    Inputs
    ----------
    x0, y0 : float
           x and y coordinates of the point of interest (i.e, does it lie inside of the ellipse?)
        
    cx, cy : float
           x and y coordinates for the centre of the ellipse
           
    a : float
      major axis of the ellipse
      
    b : float
      minor axis of the ellipse
      
    angle_degrees : float 
                  Angle of the ellipse, this is going to be the "phi" value from the NSA catalogue

    Method
    -------
    1 - Converts degrees to radians
    2 - Rotates the ellipse based on the angle in radians
    3 - Calculates if point is on ellipse using the formula: (x^2/a^2) + (y^2/b^2) <= 1
    
    Output
    ------
    Boolean : True if inside of ellipse, False if outside of ellipse
    
    """
    
    # Convert the angle to radians
    angle_radians = math.radians(angle_degrees)
    
    # Rotate the point and ellipse to align with the x-axis
    x_rot = math.cos(-angle_radians) * (x0 - cx) - math.sin(-angle_radians) * (y0 - cy)
    y_rot = math.sin(-angle_radians) * (x0 - cx) + math.cos(-angle_radians) * (y0 - cy)
    
    # Calculate the value of the unrotated ellipse equation
    result = ((x_rot / a) ** 2) + ((y_rot / b) ** 2)
    
    # Check if the point is inside the unrotated ellipse
    if result <= 1:
        return True
    else:
        return False



def ell_constrains(x, y, major, minor):
    '''
    This function is used to define the ellipse border
    ..................................................
    
    Inputs
    ------
    
    x, y : float
         x and y coordinates of the point
         
    major, minor : float 
                 major and minor axis of the ellipse, respectively
    
    Outputs
    -------
    ellipse_border : float
                   Allow for the ellipse to be drawn based on the inputs given
    '''
    
    ellipse_border = ((math.pow((x), 2) / math.pow(major, 2)) + (math.pow((y), 2) / math.pow(minor, 2)))
    
    return  ellipse_border


def Ellipse_(radius, minor):
    
    '''
    This function uses the previous two functions and returns the coordinates that make up the ellipse which
    will be used combine all the spaxels and obtain the spectra up to a certain radius. This takes as input
    the major axis (radius), minor axis and the position angle.

    '''
    
    length = np.arange(-radius - 0.25, radius + 0.25,0.5) #Major axis of the ellipse. Spaxels have side 0.25"
    
    width = np.arange(-minor, minor + 0.25, 0.5) #Minor axis of the ellipse. Spaxels have side 0.25"
    
    x = []; y = []; x_array = []; y_array = []; outside = []
    
    for width_values in width:
        
        for length_values in length:
            
            x.append(length_values + 0.25)
            
            y.append(width_values)
            
    for counter, values in enumerate(x):
        
        values = values + 0.25
        
        y_vals = y[counter] + 0.25
        
        if values >- radius and values < radius:
            
            if (ell_constrains(values, y_vals, radius, minor) > 1 ):
                
                outside.append(values)
                
            else:
                
                x_array.append(values)
                
                y_array.append(y_vals)
        
    return x_array, y_array

def ellipse_points(cx, cy, a, b, angle, num_points=100):
    
    '''
    Drwaing an ellipse at an angle
    ..............................
    
    Inputs:
    -------
    
    cx, cy : float 
           These are the x and y coordinates of the centre of the ellipse
           
    a, b : float 
         Major and minor axis, respectively (this might be change to input the axis ratio in the future)
    
    angle : float
          This is the position angle of the ellipse, in degrees 
    
    num_points : int
               Number of points to be plotted in the ellipse
    
    Outputs:
    --------
    x, y : float
         x and y coordinates of the ellipse
    
    
    '''
    
    
    t = np.linspace(0, 2 * np.pi, num_points)
    
    x = cx + a * np.cos(t) * np.cos(np.radians(angle)) - b * np.sin(t) * np.sin(np.radians(angle))
    
    y = cy + a * np.cos(t) * np.sin(np.radians(angle)) + b * np.sin(t) * np.cos(np.radians(angle))
    
    return x, y


def ploting_annulus_points (x_ellipse, y_ellipse, cx, cy, a0, b0, a1, b1, angle_degrees):
    
    '''
    Getting the coorditates for the ellipses. This function can be used in two ways. An earlier version of this
    is writted above and is called ploting_annulus_points_v1, where the goal was to plot only the points between 
    two annulus regions. With this function, all annulus regions are taken at the same time, to get the coordinates.
    ................................................................................................................
    
    Inputs:
    -------
    
    x_ellipse : float
              x coordinate of the ellipse (this is the ellipse that has been drawn using the function from 1st year)
              
    y_ellipse : float
              y coordinate of the ellipse (this is the ellipse that has been drawn using the function from 1st year)
              
    cx, cy : float
           x and y coordinates of the centre of the ellipse
           
    a0 : float
        major axis of the smalles ellipse
        
    b0 : float
        minor axis of the smaller ellipse
        
    a1 : float
        major axis of the larger ellipse
        
    b1 : float 
        minor axis of the large ellipse
    
    angle_degrees : float
                  Position angle of the ellipse in degrees
    
    Method: 
    ------
    
    Veryfies that the point is inside of the ellipses. 
    
    Output: 
    -------
    
    X_values : list
             List of x coordinates that fall in the middle of the spaxel and inside of the desired ellipse
    Y_values : list
             List of y coordinates that fall in the middle of the spaxel and inside of the desired ellipse

    '''
    
    X_values = []
    
    Y_values = []
    
    if a1 == 0 and b1 == 0 : #The first only takes the centre, no need to take other annulus regions into account
        
        for position, x_value in enumerate(x_ellipse):
    
            x0 = x_value

            y0 = y_ellipse[position]

            if point_inside_rotated_ellipse(x0, y0, cx, cy, a0, b0, angle_degrees):
                
                X_values.append(x0)
                Y_values.append(y0)
                
    else:
        
    
        for counter, x_value in enumerate(x_ellipse):

            x0 = x_value

            y0 = y_ellipse[counter]

            if point_inside_rotated_ellipse(x0, y0, cx, cy, a0, b0, angle_degrees):
                
                X_values.append(x0)
                Y_values.append(y0)

            else: 

                if point_inside_rotated_ellipse(x0, y0, cx, cy, a1, b1, angle_degrees):
                    #points between the two ellipses
                    X_values.append(x0)
                    Y_values.append(y0)
                    
    return X_values, Y_values


def remove_common_coordinates (zipped_x_y_values):
    
    '''
    Removing coordinates, to plot annulus region and not the full ellipses. 
    .......................................................................
    
    Inputs:
    -------
    zipped_x_y_values : list of touples 
                      list of (x,y) coordinates that make up each full ellipse
    Method:
    -------
    
    1- removes the coordinates that are common between dajacent ellipses, so that only those between the
    ellipses remain.
    
    2 - Computes individual x and y arrays to then plot. 
    
    Output:
    -------
    x_ : list of lists 
       Each position corresponds to each annulus region, inside each position in the list there are the 
       x coordinates for the annulus region in question. len(x_) is equal to the number of annulus regions
       that are mapped inside of the galaxy.
    
    y_ : list of lists 
       Each position corresponds to each annulus region, inside each position in the list there are the 
       y coordinates for the annulus region in question. len(y_) is equal to the number of annulus regions
       that are mapped inside of the galaxy.
    
    '''

    clean_coords = []

    for ct, ell_coords in enumerate(zipped_x_y_values):

        if ct == 0: 

            clean_coords.append(ell_coords)

        else:

            inner_array = []

            for coords in ell_coords:

                if coords not in zipped_x_y_values[ct-1]:

                    inner_array.append(coords)

            clean_coords.append(inner_array)
            
    x_ = []
    y_ = []

    for coordinates in clean_coords:

        inner_x = []; inner_y = []

        for x_y_vals in coordinates:

            inner_x.append(x_y_vals[0])

            inner_y.append(x_y_vals[1])

        x_.append(inner_x)

        y_.append(inner_y)
    
    return x_, y_
            
    

