import tensorflow as tf
import re
import string


def custom_standardization(input_data): 
    lowercase = tf.strings.lower(input_data)
    stripped_html_newlines = tf.strings.regex_replace(lowercase, '<br />', ' ')
    stripped_html_punctuation = tf.strings.regex_replace(stripped_html_newlines,
                                  '[%s]' % re.escape(string.punctuation), '')
    return stripped_html_punctuation