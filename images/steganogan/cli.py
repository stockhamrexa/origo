'''
A command line interface for all tools included in the origo package.
'''

import argparse

from model import SteganoGAN

def _init_steganogan(args):
    '''
    Initialize and return a SteganoGan object.
    '''

    kwargs = {
        'cuda': args.cuda,
        'verbose': args.verbose
    }

    if args.depth:
        kwargs['data_depth'] = args.depth

    if args.model:
        kwargs['model_type'] = args.model

    if args.log:
        kwargs['log_dir'] = args.log

    return SteganoGAN(**kwargs)

def _critique(args):
    '''
    Call the critique function of the SteganoGAN class.
    '''

    gan = _init_steganogan(args)
    gan.critique(args.file)

def _decode(args):
    '''
    Call the decode function of the SteganoGAN class.
    '''

    gan = _init_steganogan(args)

    kwargs = {
        'image': args.file,
        'greedy': args.greedy
    }

    if args.k:
        kwargs['k'] = args.k

    if args.threshold:
        kwargs['var_threshold'] = args.threshold

    gan.decode(**kwargs)

def _display(args):
    '''
    Print the SteganoGAN object.
    '''
    gan = _init_steganogan(args)
    print(gan)

def _encode(args):
    '''
    Call the encode function of the SteganoGAN class.
    '''

    gan = _init_steganogan(args)

    kwargs = {
        'cover': args.file,
        'output': args.output,
        'text': args.text
    }

    if args.pad == False:
        kwargs['pad'] = args.pad

    if args.compress == False:
        kwargs['compress'] = args.compress

    gan.encode(**kwargs)

def _get_metrics(args):
    '''
    Call the get_avg_time function of the SteganoGAN class.
    '''

    gan = _init_steganogan(args)

    kwargs = {
        'cover': args.file,
        'greedy': args.greedy
    }

    if args.iters:
        kwargs['num_iters'] = args.iters

    if args.k:
        kwargs['k'] = args.k

    if args.threshold:
        kwargs['var_threshold'] = args.threshold

    if args.pad == False:
        kwargs['pad'] = args.pad

    if args.compress == False:
        kwargs['compress'] = args.compress

    gan.get_metrics(**kwargs)

def _get_parser():
    '''
    Returns an ArgumentParser for the origo command line interface.
    '''

    # Flags that are used by all actions
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument('-d', '--depth', choices={1, 2, 3, 4, 5, 6, 7, 8}, type=int, help='Data depth.')
    parent.add_argument('-m', '--model', choices={'custom', 'div2k', 'mscoco'}, help='Model architecture.')
    parent.add_argument('-c', '--cuda', action='store_true', help='Use a CUDA device if available.')
    parent.add_argument('-l', '--log', help='Create a log directory at the given path.')
    parent.add_argument('-v', '--verbose', action='store_true', help='Be verbose.')

    # Decode flags that are used by other actions
    decode_parent = argparse.ArgumentParser(add_help=False)
    decode_parent.add_argument('-g', '--greedy', action='store_true', help='Stop after the first message is successfully decoded. Requires the -t flag.')
    decode_parent.add_argument('-k', type=int, help='The number of elements to attempt to decode.')
    decode_parent.add_argument('-t', '--threshold', type=float, help='Variance threshold. Ignore all results with a higher variance.')

    # Encode flags that are used by other actions
    encode_parent = argparse.ArgumentParser(add_help=False)
    encode_parent.add_argument('-p', '--pad', action='store_true', help='Add padding after every replication of the payload.')
    encode_parent.add_argument('-s', '--compress', action='store_true', help='Compress the payload.')

    parser = argparse.ArgumentParser(description='Image Steganography Using SteganoGAN')
    parser.set_defaults(action=None)
    subparsers = parser.add_subparsers(title='action', help='Action to perform.')

    critique = subparsers.add_parser('critique', parents=[parent], help='Compute the Wasserstein distance between the natural image distribution and a given image.')
    critique.add_argument('-f', '--file', required=True, help='Path to the file or directory to perform an action on.')
    critique.set_defaults(action=_critique)

    decode = subparsers.add_parser('decode', parents=[parent, decode_parent], help='Find the message in a steganographic image.')
    decode.add_argument('-f', '--file', required=True, help='Path to the file or directory to perform an action on.')
    decode.set_defaults(action=_decode)

    display = subparsers.add_parser('display', parents=[parent], help='Print the SteganoGAN object.')
    display.set_defaults(action=_display, file=None)

    encode = subparsers.add_parser('encode', parents=[parent, encode_parent], help='Hide a message into a steganographic image.')
    encode.add_argument('-f', '--file', required=True, help='Path to the file or directory to perform an action on.')
    encode.add_argument('-o', '--output', required=True, help='Path to the file the steganographic image should be saved as.')
    encode.add_argument('-t', '--text', required=True, help='The text to embed in an image.')
    encode.set_defaults(action=_encode)

    time = subparsers.add_parser('metrics', parents=[parent, decode_parent, encode_parent], help='Compute the average time it takes to perform each action.')
    time.add_argument('-f', '--file', required=True, help='Path to the file or directory to perform an action on.')
    time.add_argument('-i', '--iters', required=True, type=int, help='The number of iterations to average operation time over.')
    time.set_defaults(action=_get_metrics)

    return parser

def main():
    parser = _get_parser()
    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        parser.exit()

    if args.action == _decode:
        if args.greedy and not args.threshold:
            parser.error('--threshold is required when using the --greedy flag.')

    args.action(args)

if __name__ == '__main__':
    main()