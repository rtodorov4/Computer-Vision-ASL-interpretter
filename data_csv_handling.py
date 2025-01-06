import pandas as pd 

file = '/Users/ryanzhu/Downloads/Coding (suffering)/Project ASL/ASL Gesture data.csv'
xs = [2,2,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
ys = [3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
gest = 'filler'

def storeData(file, xs, ys, gest):
    df = pd.read_csv(file)
    dic = df.to_dict(orient='list')

    for i in range(21):
        key = str(i)
        coords = [xs[i], ys[1]]
        dic[key].append(coords)

    dic['gest'].append(gest)

    df = pd.DataFrame(dic)
    df.to_csv(file, index=False)
    return None

storeData(file, xs, ys, gest)