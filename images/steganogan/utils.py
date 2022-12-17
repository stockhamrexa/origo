'''
Common utility functions used in the steganogan library.
'''

import zlib
from reedsolo import RSCodec

rs = RSCodec(250) # Each block contains 256 bits, 250 of which are error correcting codes.

def text_to_bits(text, compress=True):
    '''
    Convert text to a list of ints in {0, 1}. Text can be a string or a length N list of strings.
    '''

    if isinstance(text, str):
        return bytearray_to_bits(text_to_bytearray(text, compress))

    elif isinstance(text, list):
        results = []

        for i in text:
            if isinstance(i, str):
                results.append(bytearray_to_bits(text_to_bytearray(i, compress)))

            else:
                raise ValueError(
                    'All elements in input must be a string.'
                )

        return results

    else:
        raise ValueError(
            'Input must be a string or a list of strings.'
        )

def text_to_bytearray(text, compress=True):
    '''
    Convert text to bytearray and add error correction. Optionally compress the text.
    '''

    text = text.encode('utf-8')

    if compress:
        text = zlib.compress(text)

    results = rs.encode(bytearray(text))

    return results

def bytearray_to_bits(x):
    '''
    Convert bytearray to a list of bits.
    '''

    result = []

    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])

    return result

def bits_to_text(bits):
    '''
    Convert a list of ints in {0, 1} to text. Bits can be a list of ints or a length N list of lists of ints.
    '''

    if isinstance(bits, list):
        if isinstance(bits[0], int):
            return bytearray_to_text(bits_to_bytearray(bits))

        elif isinstance(bits[0], list):
            results = []

            for i in bits:
                if not isinstance(i, list):
                    raise ValueError(
                        'All elements in input must be a list of ints.'
                    )

                results.append(bytearray_to_text(bits_to_bytearray(i)))

            return results

    else:
        raise ValueError(
            'Input must be a list of ints or a list of lists.'
        )

def bits_to_bytearray(bits):
    '''
    Convert a list of bits to a bytearray.
    '''

    ints = []

    for i in range(len(bits) // 8):
        byte = bits[i * 8:(i + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))

    return bytearray(ints)

def bytearray_to_text(x):
    '''
    Convert bytearray to text and apply error correction. Decompresses text if it was compressed.
    '''

    try:
        text = rs.decode(x)[0]

        try:
            text = zlib.decompress(text)

        except zlib.error:
            pass

        return text.decode('utf-8')

    except BaseException:
        return False

def int_to_string(x, num_bits=32):
    '''
    Converts an int to its num_bits bit binary representation and casts it as a string.
    '''

    if not isinstance(x, int) or not isinstance(num_bits, int):
        raise ValueError(
            'Input and the num_bits parameter must be ints.'
        )

    if x > (2 ** num_bits) - 1:
        raise ValueError(
            'Input must be able to be represented in 32 bits.'
        )

    return format(x, '0' + str(num_bits) + 'b')

def validate_file_type(filename):
    '''
    Determines whether or not the file extension for a given file is valid for use by SteganoGAN.

    :param filename: The string representation of a given file path.
    :return: True if the filename has a valid extension, else False. Always returns the file extension.
    '''

    if not isinstance(filename, str):
        raise ValueError(
            'The parameter filename1 must be a string.'
        )

    valid_types = ['png', 'jpg', 'jpeg', 'jpe', 'jfif', 'jif']

    filename = filename.split('.')

    if len(filename) == 1:
        raise ValueError(
            'The filename parameter must have a file type with the format: filename.type.'
        )

    type = filename[-1].lower()

    return type in valid_types, type