import sys

sys.path.append('C:\\Users\\Xinran\\Desktop\\cnn\\src\\')

import net
from layer_proto import Layer


def main():
    n = net.Net([1,2])
    n.fit(None, None, 100)
    n.add_layer(Layer('Conv'))
    print(n)



if __name__ == '__main__':
    main()