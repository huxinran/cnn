def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def toImg(rawArray):
    #print(rawArray.shape)
    H, W = 32, 32
    data = np.zeros((32, 32, 3), np.uint8)
    t = 0
    for c in range(3):
        for i in range(32):
            for j in range(32):
                #print(rawArray[t], i, j,  c)

                data[i][j][c] = rawArray[t]
                t += 1

    
    return Image.fromarray(data, 'RGB')


    d = unpickle("./data/data_batch_1")
    
    for k in d.keys():
        d[k.decode()] = d.pop(k)
    
    for i in range(10000):
        img = toImg(d['data'][i])
        img.show()
        input('next') 