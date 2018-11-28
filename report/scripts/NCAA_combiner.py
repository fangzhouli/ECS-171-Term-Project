from PIL import Image
import os


def combine(type):
    """ Combine plots in /mlp/NCAA_preGame/plots/learningRate10e-1
        Input:
          - @type: either 'accuracy' or 'weights'
    """
    rootdir = './mlp/NCAA_preGame/plots/learningRate10e-1'
    files = []
    for dir, subdirs, fs in os.walk(rootdir):
        for f in fs:
            if f.split('_')[-1] == type + '.png':
                files += [os.path.join(dir,f)]
    fs.sort()
    fs.remove(fs[0]) # remove NCAA_0_1_compact*

    # Coordinates
    x = 0
    y = 0
    # Output canvas
    out = Image.new("RGB", (2400, 3600))

    for i in range(len(files)):
        img = Image.open(files[i])
        img = img.resize([800, 600], Image.ANTIALIAS)
        if i % 6 == 0 and i != 0:
            x+=800
        y += 600
        if i % 6 == 0:
            y=0
        w, h = img.size
        out.paste(img, (x, y, x+w, y+h))

    out.save('./report/images/NCAA_18_' + type + '.jpg')
