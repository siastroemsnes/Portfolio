# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:05:36 2021

@author: Sia Stroemsnes & Marie Staveland
"""

from math import floor
import numpy as np

def raytrace(xA,yA,xB,yB):
    """ Return all cells of the unit grid crossed by the line segment between
        A and B.
    """
    
    (dx, dy) = (xB - xA, yB - yA)
        
    sx=np.where(dx>0,1,-1)
    sy=np.where(dy>0,1,-1)

    grid_A = (floor(xA), floor(yA))
    grid_B = (floor(xB), floor(yB))
    (x, y) = grid_A
    traversed=[[grid_A[0],grid_A[1],xA,yA,xB,yB]]

    tIx = np.abs(dy * (x + sx - xA)) if dx != 0 else float("+inf")
    tIy = np.abs(dx * (y + sy - yA)) if dy != 0 else float("+inf")

    
    while (x,y) != grid_B:
        # NB if tIx == tIy we increment both x and y
        (movx, movy) = (tIx <= tIy, tIy <= tIx)

        if movx:
            # intersection is at (x + sx, yA + tIx / dx^2)
            x += sx
            tIx = np.abs(dy * (x + sx - xA))

        if movy:
            # intersection is at (xA + tIy / dy^2, y + sy)
            y += sy
            tIy = np.abs(dx * (y + sy - yA))

        traversed.append([x,y,xA,yA,xB,yB])

    return traversed

# xA=425
# yA=65
# xB=429
# yB=68

ss=raytrace( 425,65 , 429,68 )


