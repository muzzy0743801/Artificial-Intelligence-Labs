# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:07:18 2022

@author: muxxa
"""

from random import randint
env=["A","B"]
acctuator=["This room is clean.","This room is dirty."]
def A(i):
 index=i
 for a in env:
     if index==0:
         print("Room A:")
         print(acctuator[index])
         print("The vacuum cleaner is now moving in Room B...\n")
         return B(randint(0, 1))
     elif index==1:
         print("Room A:")
         print(acctuator[index])
         print("Room A is getting cleaned...\n")
         index=index-1
def B(i):
 index=i
 for b in env:
     if index==0:
         print("Room B:")
         return(acctuator[index])
     elif index==1:
         print("Room B:")
         print(acctuator[index])
         print("Room B is getting cleaned...\n")
         index=index-1
A(randint(0, 1))
