from os.path import join, expanduser


root = expanduser("")
imagesets = join(root, 'Dataset', 'IMAGE')
models = join(root, 'Models')
trained = join(root, 'Models', 'trained')
plots = join(root, 'Plots')
