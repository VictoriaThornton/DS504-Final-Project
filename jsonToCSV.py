import pandas as pd
import gzip
import simplejson as json

reviews_filename = 'Video_Games'
metadata_filename = 'meta_Video_Games'

#parsing adapted from Amazon Reviews sample
def parse(path):
  g = gzip.open(path, 'rb')

  for l in g:
    yield json.loads(l)

#convert json.gz to pandas dataframe
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

#convert dataframe to CSV
df = getDF(reviews_filename + '.json.gz')
df.to_csv(reviews_filename + '.csv', index = False)
print("finished converting reviews")
print("numRows: " , df.shape[0])
print("numCol: " , df.shape[1])

df = getDF(metadata_filename + '.json.gz')
df.to_csv(metadata_filename + '.csv', index = False, header = False, doublequote= False, escapechar='\\')
print("finished converting metadata")
print("numRows: " , df.shape[0])
print("numCol: " , df.shape[1])



