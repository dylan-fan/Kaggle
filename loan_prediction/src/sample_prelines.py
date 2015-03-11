'''
Created on 2013/09/20

@author: dylan
'''

"""
sample rawfile to sample file by previous lines:

sample_prelines.py input.txt out.txt -l lines


"""



import sys
import argparse

import string

import random


print __doc__


parser = argparse.ArgumentParser()
parser.add_argument( "input_file" )
parser.add_argument( "output_file" )

parser.add_argument( "-l", "--pre-lines", help = " sample previous lines", default = None )

args = parser.parse_args()
i_f = open(args.input_file,'r')
o_f = open(args.output_file, 'w')



lines_num = args.pre_lines

lines_num = string.atoi(lines_num)




line = i_f.readline()

o_f.write(line)

i = 0
for line in i_f:
    o_f.write(line)
    i = i + 1
    if i == lines_num:
        break

i_f.close()

o_f.close()