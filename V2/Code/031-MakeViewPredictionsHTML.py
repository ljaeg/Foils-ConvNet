import numpy as np
import pandas as pd
import argparse
import os, sys

parser = argparse.ArgumentParser()
# parser.add_argument('--PredictionsFile', default='predictions.csv')
# parser.add_argument('--PredictionsFileHTML', default='predictions.html')
parser.add_argument('--MaxImages', default=-1)
parser.add_argument('--PredictionsFile', default=os.path.join('..', 'Data', 'Predict', 'amazon20k_predictions.csv'))
parser.add_argument('--PredictionsFileHTML', default=os.path.join('..', 'Data', 'Predict', 'amazon20k_predictions.html'))
parser.add_argument('--RelativePath', default='amazon20k')
args = parser.parse_args()

print(args.PredictionsFile)
d = pd.read_csv(args.PredictionsFile)
d.sort_values('Prediction', ascending=False, inplace=True)
d.reset_index(inplace=True)
# del d['index']
# print(d.head())

# If we are limiting out output to the top N images then crop the dataframe.
if int(args.MaxImages) != -1:
    d = d.head(int(args.MaxImages))

with open(args.PredictionsFileHTML, 'w') as f:
    f.write('<body>\n')
    f.write('<table>\n')
    f.write('<tr><td>Number</td><td>Prediction</td><td>Image</td><td>Image name</td></tr>\n')
    for i, r in d.iterrows():
        FileName = os.path.join(args.RelativePath, os.path.split(r.FileName)[-1])
        if 'id' in r:
            f.write(f'<tr><td>{i}</td><td>{r.Prediction}</td><td><a href="{FileName}"><img src="{FileName}"/></a></td><td>{os.path.split(r.FileName)[-1]}<br>{r.id}</td></tr>\n')
        else:
            f.write(f'<tr><td>{i}</td><td>{r.Prediction}</td><td><a href="{FileName}"><img src="{FileName}"/></a></td><td>{os.path.split(r.FileName)[-1]}</td></tr>\n')
    f.write('</table>\n')
    f.write('</body>\n')

print(f'Wrote predictions output to {os.path.join(os.getcwd(), args.PredictionsFileHTML)}')
print('(Copy and paste that into a new tab to view.)')
print('Done')
