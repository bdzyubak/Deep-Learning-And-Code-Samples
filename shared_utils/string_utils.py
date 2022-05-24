import tensorflow as tf
import re
import string


def custom_standardization(input_data): 
    lowercase = tf.strings.lower(input_data)
    stripped_html_newlines = tf.strings.regex_replace(lowercase, '<br />', ' ')
    stripped_html_punctuation = tf.strings.regex_replace(stripped_html_newlines,
                                  '[%s]' % re.escape(string.punctuation), '')
    return stripped_html_punctuation


def get_trailing_digits(str_digits:str): 
    if not isinstance(str_digits,str): 
        raise(ValueError('Input must be a string'))
    flipped_digits = ''
    for c in str_digits[::-1]: 
        if c.isdigit(): 
            flipped_digits += c
        else: 
            break
    if flipped_digits: 
        trailing_digits = flipped_digits[::-1]
    else: 
        trailing_digits = ''
    return trailing_digits

def permute_array(arr,prefix=''): 
    if isinstance(arr,str): 
        arr = list(arr)
    elif not isinstance(arr,list): 
        raise(ValueError('Input must be an array (list) or string.'))
    
    all_perms = list()
    if not prefix: 
        all_perms.append(arr)
    for i in range(1,len(arr)): 
        perm_arr = arr[i:] + arr[:i]
        if prefix: 
            perm_arr = prefix + perm_arr
        print(perm_arr)
        all_perms.append(perm_arr)
        all_perms.append(permute_array(arr[i:],prefix=arr[:i]))
    return all_perms

if __name__ == '__main__': 
    arr = 'ABC'
    permute_array(arr)
    # assert permute_array(arr) == [['ABC'],['BCA'],['CAB'],['CBA'],['ACB'],['BAC']]
    