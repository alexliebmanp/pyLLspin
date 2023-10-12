'''
This file contains functions specific to interfacting python and Mathematica
'''

import re
match_dec = '-?[0-9]+.?[0-9]*'
match_int = r'[-+]?\d+'

def mathematica_num_to_python(number):
    '''
    handles conversion from mathematica output string to python. for example, a number might be a float as a string or it may be some scientific notation such as:

    Plus[Times[3.6906886435872366, Global`e], -9]
    '''

    try:
        python_number = float(number)
    except:
        base = float(re.search(match_dec, re.search(f'Plus\[Times\[{match_dec}, Global`e\],', number)[0])[0])
        exp = float(re.search(match_int, re.search(f', {match_int}\]', number)[0])[0])
        python_number = base*10**exp

    return python_number

def python_num_to_mathematica(number):
    '''
    convert python float to mathematica appropriate string such as

    Plus[Times[3.6906886435872366, Global`e], -9]
    '''
    #mantissa, exponent = scientific_notation_split(number)
    #math_num = 'Plus[Times['+mantissa+', Global`e], '+exponent+']'

    math_num = '{:.50f}'.format(number)
    math_num = math_num.rstrip('0').rstrip('.')

    return math_num

def scientific_notation_split(number):
    # Convert the input number to scientific notation
    notation = format(number, '.16e')  # '.16e' formats the number in scientific notation with 16 decimal places

    # Split the scientific notation string into mantissa and exponent parts
    mantissa, exponent = notation.split('e')

    # Convert the exponent part to an integer
    exponent = str(int(exponent))

    return mantissa, exponent