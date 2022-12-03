###
from glob2 import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def get_list_image_to_csv(lst_train, output_csv):
  csv_content = []
  for img_path in lst_train:
    split_img = img_path.split("/")
    csv_content.append({
        'filepath': img_path,
        'image_name':split_img[-1],
        'id': split_img[-2][:-1]
    })

  df = pd.DataFrame(csv_content)
  df.to_csv(output_csv, index=False)
  print("DONE")
def parse_args():

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, required=True)
  parser.add_argument('--output_path', type=str, required=True)
  parser.add_argument('--split_test', action='store_true')
  args, _ = parser.parse_known_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  lst_img = glob(f"{args.data_path}/**/*.jpg")

  os.makedirs(args.output_path, exist_ok = True)
  lst_train, lst_val = train_test_split(lst_img, test_size = .2, random_state = 42)

  if args.split_test:
    lst_train, lst_test = train_test_split(lst_train, test_size=.1 ,random_state = 42)
    get_list_image_to_csv(lst_test, f'{args.output_path}/test.csv')

  get_list_image_to_csv(lst_train, f'{args.output_path}/train.csv')
  get_list_image_to_csv(lst_val, f'{args.output_path}/val.csv')